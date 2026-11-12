from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table
from sentence_transformers import CrossEncoder, SentenceTransformer


INPUT_DIR = Path("output")
DOCUMENT_STEM = "attention"
DEFAULT_STRATEGY = "structured"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = 5
CANDIDATE_POOL = 20
RRF_K = 60

SAMPLE_QUERIES = [
    "How many attention heads does the base model use?",
    "Why do they scale the dot products?",
    "What hardware was used for training?",
    "How does the model know the order of words?",
    "What makes the Transformer faster to train than an RNN?",
]


console = Console()


@dataclass
class Hit:
    position: int
    chunk_id: str
    section: str
    pages: list[int]
    text: str
    score: float
    fusion_rank: int | None = None


def tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token.isalnum() or "-" in token]


def load_index(strategy: str) -> tuple[dict, np.ndarray]:
    metadata_path = INPUT_DIR / f"{DOCUMENT_STEM}.index.{strategy}.json"
    vector_path = INPUT_DIR / f"{DOCUMENT_STEM}.vectors.{strategy}.npy"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vectors = np.load(vector_path)
    return metadata, vectors


def dense_search(
    model: SentenceTransformer,
    query: str,
    vectors: np.ndarray,
    limit: int,
) -> list[tuple[int, float]]:
    query_vector = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores = vectors @ query_vector
    order = np.argsort(-scores)[:limit]
    return [(int(position), float(scores[position])) for position in order]


def sparse_search(bm25: BM25Okapi, query: str, limit: int) -> list[tuple[int, float]]:
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(-scores)[:limit]
    return [(int(position), float(scores[position])) for position in order]


def reciprocal_rank_fusion(
    dense: list[tuple[int, float]],
    sparse: list[tuple[int, float]],
    limit: int,
) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}

    for rank, (position, _) in enumerate(dense, start=1):
        fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)
    for rank, (position, _) in enumerate(sparse, start=1):
        fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda pair: -pair[1])[:limit]
    return [(position, score) for position, score in ordered]


def rerank(
    reranker: CrossEncoder,
    query: str,
    chunks: list[dict],
    candidates: list[tuple[int, float]],
    limit: int,
) -> tuple[list[tuple[int, float]], float]:
    pairs = [(query, chunks[position]["text"]) for position, _ in candidates]

    started = time.perf_counter()
    scores = reranker.predict(pairs, show_progress_bar=False)
    elapsed = time.perf_counter() - started

    scored = [
        (position, float(score))
        for (position, _), score in zip(candidates, scores)
    ]
    scored.sort(key=lambda pair: -pair[1])
    return scored[:limit], elapsed


def build_hits(
    chunks: list[dict],
    results: list[tuple[int, float]],
    fusion_order: list[int] | None = None,
) -> list[Hit]:
    ranks = (
        {position: rank for rank, position in enumerate(fusion_order, start=1)}
        if fusion_order
        else {}
    )

    hits = []
    for position, score in results:
        chunk = chunks[position]
        hits.append(
            Hit(
                position=position,
                chunk_id=chunk["chunk_id"],
                section=chunk["section"],
                pages=chunk["pages"],
                text=chunk["text"],
                score=score,
                fusion_rank=ranks.get(position),
            )
        )
    return hits


def print_hits(title: str, hits: list[Hit], show_movement: bool) -> None:
    table = Table(title=title)
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    if show_movement:
        table.add_column("Was", justify="right")
        table.add_column("Move", justify="right")
    table.add_column("Section")
    table.add_column("Pages", justify="right")

    for position, hit in enumerate(hits, start=1):
        row = [str(position), f"{hit.score:.4f}"]
        if show_movement:
            previous = hit.fusion_rank
            row.append(str(previous) if previous else "-")
            if previous is None:
                row.append("-")
            elif previous > position:
                row.append(f"+{previous - position}")
            elif previous < position:
                row.append(f"-{position - previous}")
            else:
                row.append("=")
        row.append(hit.section)
        row.append(",".join(str(page) for page in hit.pages))
        table.add_row(*row)

    console.print(table)


def run_query(
    model: SentenceTransformer,
    reranker: CrossEncoder,
    bm25: BM25Okapi,
    chunks: list[dict],
    vectors: np.ndarray,
    query: str,
) -> None:
    console.print()
    console.print(f"[bold cyan]Query:[/] {query}")

    dense = dense_search(model, query, vectors, CANDIDATE_POOL)
    sparse = sparse_search(bm25, query, CANDIDATE_POOL)
    fused = reciprocal_rank_fusion(dense, sparse, CANDIDATE_POOL)
    fusion_order = [position for position, _ in fused]

    fusion_hits = build_hits(chunks, fused[:TOP_K])
    print_hits("Before reranking (hybrid RRF)", fusion_hits, show_movement=False)

    reranked, elapsed = rerank(reranker, query, chunks, fused, TOP_K)
    reranked_hits = build_hits(chunks, reranked, fusion_order)
    print_hits(
        f"After reranking ({elapsed * 1000:.0f} ms)",
        reranked_hits,
        show_movement=True,
    )

    if reranked_hits:
        best = reranked_hits[0]
        body = best.text.split("\n\n", 1)[-1]
        console.print(f"[bold green]Top result[/] ({best.section}, page {best.pages}):")
        console.print(body[:500])


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()

    strategy = DEFAULT_STRATEGY
    metadata_path = INPUT_DIR / f"{DOCUMENT_STEM}.index.{strategy}.json"
    if not metadata_path.exists():
        console.print(f"[bold red]Index not found:[/] {metadata_path.resolve()}")
        console.print("Run embed_chunks.py first.")
        return

    metadata, vectors = load_index(strategy)
    chunks = metadata["chunks"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold cyan]Device:[/] {device}")
    console.print(f"[bold cyan]Index:[/] {strategy}, {len(chunks)} chunks")

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    reranker = CrossEncoder(RERANKER_MODEL, device=device)
    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    if query:
        run_query(model, reranker, bm25, chunks, vectors, query)
        return

    for sample in SAMPLE_QUERIES:
        run_query(model, reranker, bm25, chunks, vectors, sample)


if __name__ == "__main__":
    main()