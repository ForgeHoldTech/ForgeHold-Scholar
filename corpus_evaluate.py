from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table
from sentence_transformers import CrossEncoder, SentenceTransformer

from corpus_answer import (
    CANDIDATE_POOL,
    EMBEDDING_MODEL,
    QUERY_PREFIX,
    RERANKER_MODEL,
    dense_search,
    load_index,
    reciprocal_rank_fusion,
    sparse_search,
    tokenize,
)


QUESTIONS_PATH = Path("data/corpus_questions.json")
REPORT_PATH = Path("output/corpus_evaluation_report.json")

EVAL_DEPTH = 10
MAX_FAILURES_SHOWN = 10

CONFIGURATIONS = [
    ("bm25", "bm25", False),
    ("dense", "dense", False),
    ("hybrid", "hybrid", False),
    ("hybrid + rerank", "hybrid", True),
]


console = Console()


@dataclass
class Outcome:
    question_id: str
    question: str
    expected_paper: str
    expected_chunk: str
    chunk_rank: int | None
    paper_rank: int | None
    top_paper: str


@dataclass
class Result:
    name: str
    chunk_recall_1: float
    chunk_recall_5: float
    chunk_recall_10: float
    chunk_mrr: float
    paper_recall_1: float
    paper_recall_5: float
    seconds: float
    outcomes: list[Outcome] = field(default_factory=list)


def rank_candidates(
    retriever: str,
    use_reranker: bool,
    question: str,
    query_vector: np.ndarray,
    chunks: list[dict],
    vectors: np.ndarray,
    bm25: BM25Okapi,
    reranker: CrossEncoder,
) -> list[int]:
    if retriever == "dense":
        candidates = dense_search(query_vector, vectors, CANDIDATE_POOL, None)
    elif retriever == "bm25":
        candidates = sparse_search(bm25, question, CANDIDATE_POOL, None)
    else:
        candidates = reciprocal_rank_fusion(
            dense_search(query_vector, vectors, CANDIDATE_POOL, None),
            sparse_search(bm25, question, CANDIDATE_POOL, None),
            CANDIDATE_POOL,
        )

    if use_reranker and candidates:
        pairs = [(question, chunks[position]["text"]) for position in candidates]
        scores = reranker.predict(pairs, show_progress_bar=False)
        ordered = sorted(zip(candidates, scores), key=lambda pair: -float(pair[1]))
        candidates = [position for position, _ in ordered]

    return candidates[:EVAL_DEPTH]


def first_rank(values: list[str], target: str) -> int | None:
    for position, value in enumerate(values, start=1):
        if value == target:
            return position
    return None


def share(outcomes: list[Outcome], attribute: str, depth: int) -> float:
    hits = sum(
        1
        for outcome in outcomes
        if getattr(outcome, attribute) is not None
        and getattr(outcome, attribute) <= depth
    )
    return hits / len(outcomes)


def evaluate(
    name: str,
    retriever: str,
    use_reranker: bool,
    questions: list[dict],
    query_vectors: np.ndarray,
    chunks: list[dict],
    vectors: np.ndarray,
    bm25: BM25Okapi,
    reranker: CrossEncoder,
) -> Result:
    outcomes: list[Outcome] = []
    started = time.perf_counter()

    for position, entry in enumerate(questions):
        top = rank_candidates(
            retriever,
            use_reranker,
            entry["question"],
            query_vectors[position],
            chunks,
            vectors,
            bm25,
            reranker,
        )

        chunk_ids = [chunks[index]["chunk_id"] for index in top]
        paper_ids = [chunks[index]["paper_id"] for index in top]

        outcomes.append(
            Outcome(
                question_id=entry["id"],
                question=entry["question"],
                expected_paper=entry["paper_title"],
                expected_chunk=entry["chunk_id"],
                chunk_rank=first_rank(chunk_ids, entry["chunk_id"]),
                paper_rank=first_rank(paper_ids, entry["paper_id"]),
                top_paper=chunks[top[0]]["paper_title"] if top else "",
            )
        )

    elapsed = time.perf_counter() - started

    return Result(
        name=name,
        chunk_recall_1=share(outcomes, "chunk_rank", 1),
        chunk_recall_5=share(outcomes, "chunk_rank", 5),
        chunk_recall_10=share(outcomes, "chunk_rank", 10),
        chunk_mrr=sum(
            1.0 / outcome.chunk_rank for outcome in outcomes if outcome.chunk_rank
        ) / len(outcomes),
        paper_recall_1=share(outcomes, "paper_rank", 1),
        paper_recall_5=share(outcomes, "paper_rank", 5),
        seconds=elapsed,
        outcomes=outcomes,
    )


def main() -> None:
    if not QUESTIONS_PATH.exists():
        console.print(f"[bold red]Questions not found:[/] {QUESTIONS_PATH.resolve()}")
        console.print("Run make_eval_set.py first.")
        return

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    if not questions:
        console.print("[bold red]The question file is empty.[/]")
        return

    index, chunks, vectors = load_index()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold cyan]Device:[/] {device}")
    console.print(
        f"[bold cyan]Corpus:[/] {index['document_count']} papers, "
        f"{index['chunk_count']:,} chunks"
    )
    console.print(f"[bold cyan]Questions:[/] {len(questions)}")

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    reranker = CrossEncoder(RERANKER_MODEL, device=device)
    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    query_vectors = model.encode(
        [QUERY_PREFIX + entry["question"] for entry in questions],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    results: list[Result] = []
    for name, retriever, use_reranker in CONFIGURATIONS:
        console.print(f"Evaluating {name}")
        results.append(
            evaluate(
                name,
                retriever,
                use_reranker,
                questions,
                query_vectors,
                chunks,
                vectors,
                bm25,
                reranker,
            )
        )

    table = Table(title=f"Corpus retrieval evaluation ({len(questions)} questions)")
    table.add_column("Configuration")
    table.add_column("Chunk R@1", justify="right")
    table.add_column("Chunk R@5", justify="right")
    table.add_column("Chunk R@10", justify="right")
    table.add_column("Chunk MRR", justify="right")
    table.add_column("Paper R@1", justify="right")
    table.add_column("Paper R@5", justify="right")
    table.add_column("Sec", justify="right")

    for result in results:
        table.add_row(
            result.name,
            f"{result.chunk_recall_1:.3f}",
            f"{result.chunk_recall_5:.3f}",
            f"{result.chunk_recall_10:.3f}",
            f"{result.chunk_mrr:.3f}",
            f"{result.paper_recall_1:.3f}",
            f"{result.paper_recall_5:.3f}",
            f"{result.seconds:.1f}",
        )

    console.print()
    console.print(table)

    best = max(results, key=lambda result: result.chunk_mrr)
    failures = [outcome for outcome in best.outcomes if outcome.chunk_rank is None]

    if failures:
        failure_table = Table(
            title=f"Missed questions in {best.name} "
            f"({len(failures)} of {len(questions)})"
        )
        failure_table.add_column("ID")
        failure_table.add_column("Question")
        failure_table.add_column("Expected paper")
        failure_table.add_column("Top paper returned")

        for outcome in failures[:MAX_FAILURES_SHOWN]:
            failure_table.add_row(
                outcome.question_id,
                outcome.question[:55],
                outcome.expected_paper[:28],
                outcome.top_paper[:28],
            )

        console.print()
        console.print(failure_table)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "question_count": len(questions),
        "document_count": index["document_count"],
        "chunk_count": index["chunk_count"],
        "eval_depth": EVAL_DEPTH,
        "candidate_pool": CANDIDATE_POOL,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "results": [
            {
                "configuration": result.name,
                "chunk_recall_at_1": round(result.chunk_recall_1, 4),
                "chunk_recall_at_5": round(result.chunk_recall_5, 4),
                "chunk_recall_at_10": round(result.chunk_recall_10, 4),
                "chunk_mrr": round(result.chunk_mrr, 4),
                "paper_recall_at_1": round(result.paper_recall_1, 4),
                "paper_recall_at_5": round(result.paper_recall_5, 4),
                "seconds": round(result.seconds, 2),
                "outcomes": [
                    {
                        "question_id": outcome.question_id,
                        "question": outcome.question,
                        "expected_paper": outcome.expected_paper,
                        "expected_chunk": outcome.expected_chunk,
                        "chunk_rank": outcome.chunk_rank,
                        "paper_rank": outcome.paper_rank,
                        "top_paper": outcome.top_paper,
                    }
                    for outcome in result.outcomes
                ],
            }
            for result in results
        ],
    }
    REPORT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    console.print()
    console.print(f"[bold green]Saved:[/] {REPORT_PATH}")
    console.print(
        f"[bold green]Best configuration:[/] {best.name} "
        f"(chunk MRR {best.chunk_mrr:.3f}, chunk R@5 {best.chunk_recall_5:.3f})"
    )


if __name__ == "__main__":
    main()