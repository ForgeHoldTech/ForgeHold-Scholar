from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from math import log2
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table
from sentence_transformers import CrossEncoder, SentenceTransformer


INPUT_DIR = Path("output")
OUTPUT_DIR = Path("output")
DATASET_PATH = Path("data/golden_questions.json")
DOCUMENT_STEM = "attention"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = 5
CANDIDATE_POOL = 20
RRF_K = 60

STRATEGIES = ["fixed", "paragraph", "section", "structured"]
RETRIEVERS = ["dense", "bm25", "hybrid"]


console = Console()


@dataclass
class QueryOutcome:
    question_id: str
    question: str
    category: str
    difficulty: str
    expected: list[str]
    retrieved: list[str]
    hit_rank: int | None
    reciprocal_rank: float
    ndcg_at_5: float


@dataclass
class Result:
    strategy: str
    retriever: str
    reranked: bool
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    seconds: float
    by_category: dict[str, float] = field(default_factory=dict)
    outcomes: list[QueryOutcome] = field(default_factory=list)


def tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token.isalnum() or "-" in token]


def section_key(chunk: dict) -> str:
    number = chunk.get("section_number", "")
    if number:
        return number
    heading = chunk.get("section", "")
    return heading.split()[0] if heading else ""


def load_index(strategy: str) -> tuple[list[dict], np.ndarray] | None:
    metadata_path = INPUT_DIR / f"{DOCUMENT_STEM}.index.{strategy}.json"
    vector_path = INPUT_DIR / f"{DOCUMENT_STEM}.vectors.{strategy}.npy"
    if not metadata_path.exists() or not vector_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vectors = np.load(vector_path)
    return metadata["chunks"], vectors


def dense_search(query_vector: np.ndarray, vectors: np.ndarray, limit: int) -> list[int]:
    scores = vectors @ query_vector
    return [int(position) for position in np.argsort(-scores)[:limit]]


def sparse_search(bm25: BM25Okapi, query: str, limit: int) -> list[int]:
    scores = bm25.get_scores(tokenize(query))
    return [int(position) for position in np.argsort(-scores)[:limit]]


def reciprocal_rank_fusion(dense: list[int], sparse: list[int], limit: int) -> list[int]:
    fused: dict[int, float] = {}
    for rank, position in enumerate(dense, start=1):
        fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)
    for rank, position in enumerate(sparse, start=1):
        fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda pair: -pair[1])[:limit]
    return [position for position, _ in ordered]


def recall_at(flags: list[bool], k: int) -> float:
    return 1.0 if any(flags[:k]) else 0.0


def reciprocal_rank(flags: list[bool]) -> float:
    for position, is_relevant in enumerate(flags, start=1):
        if is_relevant:
            return 1.0 / position
    return 0.0


def ndcg_at(flags: list[bool], k: int, relevant_available: int) -> float:
    gain = sum(
        1.0 / log2(position + 1)
        for position, is_relevant in enumerate(flags[:k], start=1)
        if is_relevant
    )
    ideal_count = min(k, relevant_available)
    ideal = sum(1.0 / log2(position + 1) for position in range(1, ideal_count + 1))
    return gain / ideal if ideal else 0.0


def evaluate_configuration(
    strategy: str,
    retriever: str,
    use_reranker: bool,
    chunks: list[dict],
    vectors: np.ndarray,
    bm25: BM25Okapi,
    query_vectors: dict[str, np.ndarray],
    reranker: CrossEncoder | None,
    questions: list[dict],
) -> Result:
    keys = [section_key(chunk) for chunk in chunks]

    outcomes: list[QueryOutcome] = []
    started = time.perf_counter()

    for entry in questions:
        question = entry["question"]
        relevant = set(entry["relevant_sections"])

        if retriever == "dense":
            candidates = dense_search(query_vectors[question], vectors, CANDIDATE_POOL)
        elif retriever == "bm25":
            candidates = sparse_search(bm25, question, CANDIDATE_POOL)
        else:
            candidates = reciprocal_rank_fusion(
                dense_search(query_vectors[question], vectors, CANDIDATE_POOL),
                sparse_search(bm25, question, CANDIDATE_POOL),
                CANDIDATE_POOL,
            )

        if use_reranker and reranker is not None and candidates:
            pairs = [(question, chunks[position]["text"]) for position in candidates]
            scores = reranker.predict(pairs, show_progress_bar=False)
            ordered = sorted(zip(candidates, scores), key=lambda pair: -float(pair[1]))
            candidates = [position for position, _ in ordered]

        top = candidates[:TOP_K]
        flags = [keys[position] in relevant for position in top]

        relevant_in_index = sum(1 for key in keys if key in relevant)
        relevant_available = min(relevant_in_index, TOP_K)

        hit_rank = next(
            (position for position, flag in enumerate(flags, start=1) if flag), None
        )

        outcomes.append(
            QueryOutcome(
                question_id=entry["id"],
                question=question,
                category=entry["category"],
                difficulty=entry["difficulty"],
                expected=sorted(relevant),
                retrieved=[keys[position] for position in top],
                hit_rank=hit_rank,
                reciprocal_rank=reciprocal_rank(flags),
                ndcg_at_5=ndcg_at(flags, TOP_K, relevant_available),
            )
        )

    elapsed = time.perf_counter() - started
    count = len(outcomes)

    by_category: dict[str, float] = {}
    for category in sorted({outcome.category for outcome in outcomes}):
        rows = [outcome for outcome in outcomes if outcome.category == category]
        by_category[category] = sum(
            1.0 for row in rows if row.hit_rank is not None
        ) / len(rows)

    return Result(
        strategy=strategy,
        retriever=retriever,
        reranked=use_reranker,
        recall_at_1=sum(
            1.0 for row in outcomes if row.hit_rank == 1
        ) / count,
        recall_at_3=sum(
            1.0 for row in outcomes if row.hit_rank is not None and row.hit_rank <= 3
        ) / count,
        recall_at_5=sum(1.0 for row in outcomes if row.hit_rank is not None) / count,
        mrr=sum(row.reciprocal_rank for row in outcomes) / count,
        ndcg_at_5=sum(row.ndcg_at_5 for row in outcomes) / count,
        seconds=elapsed,
        by_category=by_category,
        outcomes=outcomes,
    )


def main() -> None:
    if not DATASET_PATH.exists():
        console.print(f"[bold red]Dataset not found:[/] {DATASET_PATH.resolve()}")
        console.print("Run eval_dataset.py first.")
        return

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    questions = dataset["questions"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold cyan]Device:[/] {device}")
    console.print(f"[bold cyan]Questions:[/] {len(questions)}")

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    reranker = CrossEncoder(RERANKER_MODEL, device=device)

    texts = [QUERY_PREFIX + entry["question"] for entry in questions]
    encoded = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)
    query_vectors = {
        entry["question"]: encoded[position] for position, entry in enumerate(questions)
    }

    results: list[Result] = []

    for strategy in STRATEGIES:
        loaded = load_index(strategy)
        if loaded is None:
            console.print(f"[yellow]Missing index for strategy:[/] {strategy}")
            continue

        chunks, vectors = loaded
        bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

        for retriever in RETRIEVERS:
            for use_reranker in (False, True):
                console.print(
                    f"Evaluating {strategy} / {retriever}"
                    f"{' + rerank' if use_reranker else ''}"
                )
                results.append(
                    evaluate_configuration(
                        strategy,
                        retriever,
                        use_reranker,
                        chunks,
                        vectors,
                        bm25,
                        query_vectors,
                        reranker,
                        questions,
                    )
                )

    if not results:
        console.print("[bold red]No indexes found. Run embed_chunks.py first.[/]")
        return

    results.sort(key=lambda item: (-item.ndcg_at_5, -item.mrr))

    table = Table(title="Retrieval evaluation")
    table.add_column("Strategy")
    table.add_column("Retriever")
    table.add_column("Rerank")
    table.add_column("R@1", justify="right")
    table.add_column("R@3", justify="right")
    table.add_column("R@5", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("nDCG@5", justify="right")
    table.add_column("Sec", justify="right")

    for result in results:
        table.add_row(
            result.strategy,
            result.retriever,
            "yes" if result.reranked else "no",
            f"{result.recall_at_1:.3f}",
            f"{result.recall_at_3:.3f}",
            f"{result.recall_at_5:.3f}",
            f"{result.mrr:.3f}",
            f"{result.ndcg_at_5:.3f}",
            f"{result.seconds:.1f}",
        )

    console.print()
    console.print(table)

    lookup = {
        (result.strategy, result.retriever, result.reranked): result
        for result in results
    }

    impact = Table(title="Reranker impact on nDCG@5")
    impact.add_column("Strategy")
    impact.add_column("Retriever")
    impact.add_column("Without", justify="right")
    impact.add_column("With", justify="right")
    impact.add_column("Delta", justify="right")

    for strategy in STRATEGIES:
        for retriever in RETRIEVERS:
            before = lookup.get((strategy, retriever, False))
            after = lookup.get((strategy, retriever, True))
            if not before or not after:
                continue
            delta = after.ndcg_at_5 - before.ndcg_at_5
            impact.add_row(
                strategy,
                retriever,
                f"{before.ndcg_at_5:.3f}",
                f"{after.ndcg_at_5:.3f}",
                f"{delta:+.3f}",
            )

    console.print()
    console.print(impact)

    best = results[0]

    categories = sorted({entry["category"] for entry in questions})
    breakdown = Table(
        title=f"Recall@5 by category ({best.strategy} / {best.retriever}"
        f"{' + rerank' if best.reranked else ''})"
    )
    breakdown.add_column("Category")
    breakdown.add_column("Recall@5", justify="right")

    for category in categories:
        value = best.by_category.get(category)
        breakdown.add_row(category, f"{value:.3f}" if value is not None else "-")

    console.print()
    console.print(breakdown)

    failures = [outcome for outcome in best.outcomes if outcome.hit_rank is None]
    if failures:
        failure_table = Table(title="Failed queries in best configuration")
        failure_table.add_column("ID")
        failure_table.add_column("Question")
        failure_table.add_column("Expected")
        failure_table.add_column("Retrieved")

        for outcome in failures:
            failure_table.add_row(
                outcome.question_id,
                outcome.question[:60],
                ",".join(outcome.expected),
                ",".join(outcome.retrieved),
            )

        console.print()
        console.print(failure_table)

    payload = {
        "document": dataset["document"],
        "question_count": len(questions),
        "top_k": TOP_K,
        "candidate_pool": CANDIDATE_POOL,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "results": [
            {
                "strategy": result.strategy,
                "retriever": result.retriever,
                "reranked": result.reranked,
                "recall_at_1": round(result.recall_at_1, 4),
                "recall_at_3": round(result.recall_at_3, 4),
                "recall_at_5": round(result.recall_at_5, 4),
                "mrr": round(result.mrr, 4),
                "ndcg_at_5": round(result.ndcg_at_5, 4),
                "seconds": round(result.seconds, 2),
                "by_category": {
                    category: round(value, 4)
                    for category, value in result.by_category.items()
                },
                "outcomes": [
                    {
                        "question_id": outcome.question_id,
                        "question": outcome.question,
                        "category": outcome.category,
                        "difficulty": outcome.difficulty,
                        "expected": outcome.expected,
                        "retrieved": outcome.retrieved,
                        "hit_rank": outcome.hit_rank,
                        "reciprocal_rank": round(outcome.reciprocal_rank, 4),
                        "ndcg_at_5": round(outcome.ndcg_at_5, 4),
                    }
                    for outcome in result.outcomes
                ],
            }
            for result in results
        ],
    }

    report_path = OUTPUT_DIR / "evaluation_report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    console.print()
    console.print(f"[bold green]Saved:[/] {report_path}")
    console.print(
        f"[bold green]Best configuration:[/] {best.strategy} / {best.retriever}"
        f"{' + rerank' if best.reranked else ''} "
        f"(nDCG@5 {best.ndcg_at_5:.3f}, R@5 {best.recall_at_5:.3f})"
    )


if __name__ == "__main__":
    main()