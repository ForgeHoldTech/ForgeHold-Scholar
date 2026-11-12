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


INDEX_PATH = Path("output/corpus.index.json")
VECTOR_PATH = Path("output/corpus.vectors.npy")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = 8
CANDIDATE_POOL = 40
RRF_K = 60
MAX_PER_PAPER = 3
DIVERSITY_SCORE_MARGIN = 4.0

SAMPLE_QUERIES = [
    "What is retrieval augmented generation?",
    "How does BERT handle bidirectional context?",
    "What chunking strategy works best for retrieval?",
    "How do dense retrievers scale with model size?",
]


console = Console()


@dataclass
class Hit:
    position: int
    chunk_id: str
    paper_id: str
    paper_title: str
    published: str
    section: str
    pages: list[int]
    text: str
    score: float
    fusion_rank: int | None = None


def tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token.isalnum() or "-" in token]


def load_index() -> tuple[dict, list[dict], np.ndarray]:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    vectors = np.load(VECTOR_PATH)
    return index, index["chunks"], vectors


def allowed_positions(
    chunks: list[dict], paper: str | None, year: str | None
) -> set[int] | None:
    if not paper and not year:
        return None

    allowed = set()
    needle = paper.lower() if paper else ""

    for position, chunk in enumerate(chunks):
        if paper:
            matches = (
                needle in chunk["paper_id"].lower()
                or needle in chunk["paper_title"].lower()
            )
            if not matches:
                continue
        if year and not chunk.get("published", "").startswith(year):
            continue
        allowed.add(position)

    return allowed


def dense_search(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    limit: int,
    allowed: set[int] | None,
) -> list[int]:
    scores = vectors @ query_vector
    order = np.argsort(-scores)

    results = []
    for position in order:
        index = int(position)
        if allowed is not None and index not in allowed:
            continue
        results.append(index)
        if len(results) >= limit:
            break
    return results


def sparse_search(
    bm25: BM25Okapi, query: str, limit: int, allowed: set[int] | None
) -> list[int]:
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(-scores)

    results = []
    for position in order:
        index = int(position)
        if allowed is not None and index not in allowed:
            continue
        results.append(index)
        if len(results) >= limit:
            break
    return results


def reciprocal_rank_fusion(dense: list[int], sparse: list[int], limit: int) -> list[int]:
    fused: dict[int, float] = {}
    for rank, position in enumerate(dense, start=1):
        fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)
    for rank, position in enumerate(sparse, start=1):
        fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda pair: -pair[1])[:limit]
    return [position for position, _ in ordered]


def diversify(
    chunks: list[dict],
    ranked: list[tuple[int, float]],
    cap: int,
    limit: int,
    margin: float,
) -> list[tuple[int, float]]:
    if not ranked:
        return []

    best_score = ranked[0][1]
    threshold = best_score - margin

    counts: dict[str, int] = {}
    kept: list[tuple[int, float]] = []
    deferred: list[tuple[int, float]] = []

    for position, score in ranked:
        paper_id = chunks[position]["paper_id"]
        over_cap = counts.get(paper_id, 0) >= cap

        if over_cap and score >= threshold:
            deferred.append((position, score))
            continue

        counts[paper_id] = counts.get(paper_id, 0) + 1
        kept.append((position, score))

        if len(kept) >= limit:
            return kept

    combined = kept + deferred
    combined.sort(key=lambda pair: -pair[1])
    return combined[:limit]


def build_hits(
    chunks: list[dict],
    ranked: list[tuple[int, float]],
    fusion_order: list[int],
) -> list[Hit]:
    ranks = {position: rank for rank, position in enumerate(fusion_order, start=1)}

    hits = []
    for position, score in ranked:
        chunk = chunks[position]
        hits.append(
            Hit(
                position=position,
                chunk_id=chunk["chunk_id"],
                paper_id=chunk["paper_id"],
                paper_title=chunk["paper_title"],
                published=chunk.get("published", ""),
                section=chunk["section"],
                pages=chunk["pages"],
                text=chunk["text"],
                score=score,
                fusion_rank=ranks.get(position),
            )
        )
    return hits


def print_hits(title: str, hits: list[Hit]) -> None:
    table = Table(title=title)
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Was", justify="right")
    table.add_column("Paper")
    table.add_column("Year", justify="right")
    table.add_column("Section")
    table.add_column("Pg", justify="right")

    for position, hit in enumerate(hits, start=1):
        table.add_row(
            str(position),
            f"{hit.score:.3f}",
            str(hit.fusion_rank) if hit.fusion_rank else "-",
            hit.paper_title[:40],
            hit.published[:4],
            hit.section[:28],
            ",".join(str(page) for page in hit.pages),
        )

    console.print(table)


def run_query(
    model: SentenceTransformer,
    reranker: CrossEncoder,
    bm25: BM25Okapi,
    chunks: list[dict],
    vectors: np.ndarray,
    query: str,
    paper: str | None,
    year: str | None,
    top_k: int,
) -> None:
    console.print()
    console.print(f"[bold cyan]Query:[/] {query}")
    if paper or year:
        console.print(f"[dim]filter: paper={paper or 'any'} year={year or 'any'}[/dim]")

    allowed = allowed_positions(chunks, paper, year)
    if allowed is not None and not allowed:
        console.print("[bold yellow]No chunks match that filter.[/]")
        return

    query_vector = model.encode(
        QUERY_PREFIX + query, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    started = time.perf_counter()
    dense = dense_search(query_vector, vectors, CANDIDATE_POOL, allowed)
    sparse = sparse_search(bm25, query, CANDIDATE_POOL, allowed)
    fused = reciprocal_rank_fusion(dense, sparse, CANDIDATE_POOL)

    if not fused:
        console.print("[bold yellow]No results.[/]")
        return

    pairs = [(query, chunks[position]["text"]) for position in fused]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        ((position, float(score)) for position, score in zip(fused, scores)),
        key=lambda pair: -pair[1],
    )
    elapsed = time.perf_counter() - started

    selected = diversify(
        chunks, ranked, MAX_PER_PAPER, top_k, DIVERSITY_SCORE_MARGIN
    )
    hits = build_hits(chunks, selected, fused)

    print_hits(f"Results ({elapsed * 1000:.0f} ms)", hits)

    papers = sorted({hit.paper_title for hit in hits})
    console.print(f"[dim]{len(hits)} results from {len(papers)} papers[/dim]")

    if hits:
        best = hits[0]
        body = best.text.split("\n\n", 1)[-1]
        console.print()
        console.print(f"[bold green]Top result[/] ({best.paper_title}, page {best.pages}):")
        console.print(body[:450])


def main() -> None:
    arguments = sys.argv[1:]

    paper = None
    year = None
    top_k = TOP_K
    remaining = []
    skip = False

    for position, argument in enumerate(arguments):
        if skip:
            skip = False
            continue
        if argument == "--paper" and position + 1 < len(arguments):
            paper = arguments[position + 1]
            skip = True
        elif argument == "--year" and position + 1 < len(arguments):
            year = arguments[position + 1]
            skip = True
        elif argument == "--top" and position + 1 < len(arguments):
            value = arguments[position + 1]
            if value.isdigit():
                top_k = max(1, min(int(value), CANDIDATE_POOL))
            skip = True
        else:
            remaining.append(argument)

    query = " ".join(remaining).strip()

    if not INDEX_PATH.exists():
        console.print(f"[bold red]Index not found:[/] {INDEX_PATH.resolve()}")
        console.print("Run build_index.py first.")
        return

    index, chunks, vectors = load_index()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold cyan]Device:[/] {device}")
    console.print(
        f"[bold cyan]Corpus:[/] {index['document_count']} papers, "
        f"{index['chunk_count']:,} chunks"
    )

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    reranker = CrossEncoder(RERANKER_MODEL, device=device)
    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    queries = [query] if query else SAMPLE_QUERIES
    for item in queries:
        run_query(
            model, reranker, bm25, chunks, vectors, item, paper, year, top_k
        )


if __name__ == "__main__":
    main()