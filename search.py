from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer


INPUT_DIR = Path("output")
DOCUMENT_STEM = "attention"
DEFAULT_STRATEGY = "structured"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
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
    chunk_id: str
    section: str
    pages: list[int]
    text: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


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
) -> list[tuple[int, float, int | None, int | None]]:
    dense_ranks = {position: rank for rank, (position, _) in enumerate(dense, start=1)}
    sparse_ranks = {position: rank for rank, (position, _) in enumerate(sparse, start=1)}

    fused: dict[int, float] = {}
    for position, rank in dense_ranks.items():
        fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)
    for position, rank in sparse_ranks.items():
        fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda pair: -pair[1])[:limit]
    return [
        (position, score, dense_ranks.get(position), sparse_ranks.get(position))
        for position, score in ordered
    ]


def build_hits(
    chunks: list[dict],
    results: list[tuple[int, float, int | None, int | None]],
) -> list[Hit]:
    hits = []
    for position, score, dense_rank, sparse_rank in results:
        chunk = chunks[position]
        hits.append(
            Hit(
                chunk_id=chunk["chunk_id"],
                section=chunk["section"],
                pages=chunk["pages"],
                text=chunk["text"],
                score=score,
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
            )
        )
    return hits


def print_hits(title: str, hits: list[Hit], show_ranks: bool) -> None:
    table = Table(title=title)
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    if show_ranks:
        table.add_column("Dense", justify="right")
        table.add_column("BM25", justify="right")
    table.add_column("Section")
    table.add_column("Pages", justify="right")

    for position, hit in enumerate(hits, start=1):
        row = [str(position), f"{hit.score:.4f}"]
        if show_ranks:
            row.append(str(hit.dense_rank) if hit.dense_rank else "-")
            row.append(str(hit.sparse_rank) if hit.sparse_rank else "-")
        row.append(hit.section)
        row.append(",".join(str(page) for page in hit.pages))
        table.add_row(*row)

    console.print(table)


def run_query(
    model: SentenceTransformer,
    bm25: BM25Okapi,
    chunks: list[dict],
    vectors: np.ndarray,
    query: str,
    verbose: bool,
) -> None:
    console.print()
    console.print(f"[bold cyan]Query:[/] {query}")

    dense = dense_search(model, query, vectors, CANDIDATE_POOL)
    sparse = sparse_search(bm25, query, CANDIDATE_POOL)
    fused = reciprocal_rank_fusion(dense, sparse, TOP_K)

    if verbose:
        dense_hits = build_hits(
            chunks, [(position, score, None, None) for position, score in dense[:TOP_K]]
        )
        sparse_hits = build_hits(
            chunks, [(position, score, None, None) for position, score in sparse[:TOP_K]]
        )
        print_hits("Dense only", dense_hits, show_ranks=False)
        print_hits("BM25 only", sparse_hits, show_ranks=False)

    hits = build_hits(chunks, fused)
    print_hits("Hybrid (RRF)", hits, show_ranks=True)

    if hits:
        best = hits[0]
        body = best.text.split("\n\n", 1)[-1]
        console.print(f"[bold green]Top result[/] ({best.section}, page {best.pages}):")
        console.print(body[:500])


def main() -> None:
    arguments = sys.argv[1:]
    verbose = "--verbose" in arguments
    arguments = [item for item in arguments if item != "--verbose"]
    query = " ".join(arguments).strip()

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

    model = SentenceTransformer(MODEL_NAME, device=device)
    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    if query:
        run_query(model, bm25, chunks, vectors, query, verbose)
        return

    for sample in SAMPLE_QUERIES:
        run_query(model, bm25, chunks, vectors, sample, verbose)


if __name__ == "__main__":
    main()