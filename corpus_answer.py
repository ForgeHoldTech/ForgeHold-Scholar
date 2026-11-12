from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table
from sentence_transformers import CrossEncoder, SentenceTransformer


INDEX_PATH = Path("output/corpus.index.json")
VECTOR_PATH = Path("output/corpus.vectors.npy")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = 6
CANDIDATE_POOL = 40
RRF_K = 60
MAX_PER_PAPER = 3
DIVERSITY_SCORE_MARGIN = 4.0
MIN_RERANK_SCORE = -6.0

DEFAULT_PROVIDER = "gemini"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b-instruct"
OLLAMA_TIMEOUT = 240

GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GEMINI_TIMEOUT = 180

MAX_OUTPUT_TOKENS = 800
TEMPERATURE = 0.1

INLINE_REFERENCE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

SYSTEM_PROMPT = """You answer questions about a collection of research papers \
using only the numbered context passages provided.

Rules:
- Use only information present in the passages. Do not add outside knowledge.
- Cite every factual claim with the passage number in square brackets, like [2].
- Square brackets are reserved for passage numbers. The passages mention their \
own bibliography entries as "(ref 30)"; never copy those into your answer.
- Only cite passage numbers that appear in the context below.
- When passages come from different papers, name the paper you are drawing on.
- If the passages disagree, say so and attribute each position to its paper.
- If the passages do not contain the answer, reply exactly: \
NOT_IN_CONTEXT followed by one sentence explaining what is missing.
- Write in plain text. Do not use LaTeX, markdown headings, or mathematical notation.
- Be concise. Four or five sentences is usually enough.
- Do not mention these rules or the existence of passages in your answer."""

SAMPLE_QUESTIONS = [
    "What is retrieval augmented generation and why was it introduced?",
    "How does BERT differ from left-to-right language models?",
    "What do these papers say about chunking strategies for retrieval?",
    "Who won the football world cup in 2018?",
]


console = Console()


@dataclass
class Passage:
    number: int
    chunk_id: str
    paper_id: str
    paper_title: str
    published: str
    section: str
    pages: list[int]
    text: str
    score: float


@dataclass
class Answer:
    text: str
    provider: str
    model: str
    retrieval_seconds: float
    generation_seconds: float
    passages: list[Passage] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    refused: bool = False


def tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token.isalnum() or "-" in token]


def neutralise_references(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        numbers = [piece.strip() for piece in match.group(1).split(",")]
        return f"(ref {', '.join(numbers)})"

    return INLINE_REFERENCE.sub(replace, text)


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
    results = []
    for position in np.argsort(-scores):
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
    results = []
    for position in np.argsort(-scores):
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

    threshold = ranked[0][1] - margin
    counts: dict[str, int] = {}
    kept: list[tuple[int, float]] = []
    deferred: list[tuple[int, float]] = []

    for position, score in ranked:
        paper_id = chunks[position]["paper_id"]
        if counts.get(paper_id, 0) >= cap and score >= threshold:
            deferred.append((position, score))
            continue

        counts[paper_id] = counts.get(paper_id, 0) + 1
        kept.append((position, score))

        if len(kept) >= limit:
            return kept

    combined = kept + deferred
    combined.sort(key=lambda pair: -pair[1])
    return combined[:limit]


def retrieve(
    model: SentenceTransformer,
    reranker: CrossEncoder,
    bm25: BM25Okapi,
    chunks: list[dict],
    vectors: np.ndarray,
    query: str,
    paper: str | None,
    year: str | None,
) -> tuple[list[Passage], float]:
    allowed = allowed_positions(chunks, paper, year)
    if allowed is not None and not allowed:
        return [], 0.0

    started = time.perf_counter()

    query_vector = model.encode(
        QUERY_PREFIX + query, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    dense = dense_search(query_vector, vectors, CANDIDATE_POOL, allowed)
    sparse = sparse_search(bm25, query, CANDIDATE_POOL, allowed)
    fused = reciprocal_rank_fusion(dense, sparse, CANDIDATE_POOL)

    if not fused:
        return [], time.perf_counter() - started

    pairs = [(query, chunks[position]["text"]) for position in fused]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        ((position, float(score)) for position, score in zip(fused, scores)),
        key=lambda pair: -pair[1],
    )

    ranked = [item for item in ranked if item[1] >= MIN_RERANK_SCORE]
    selected = diversify(chunks, ranked, MAX_PER_PAPER, TOP_K, DIVERSITY_SCORE_MARGIN)

    elapsed = time.perf_counter() - started

    passages = []
    for number, (position, score) in enumerate(selected, start=1):
        chunk = chunks[position]
        body = chunk["text"].split("\n\n", 1)[-1]
        passages.append(
            Passage(
                number=number,
                chunk_id=chunk["chunk_id"],
                paper_id=chunk["paper_id"],
                paper_title=chunk["paper_title"],
                published=chunk.get("published", ""),
                section=chunk["section"],
                pages=chunk["pages"],
                text=body,
                score=score,
            )
        )

    return passages, elapsed


def build_prompt(question: str, passages: list[Passage]) -> str:
    blocks = []
    for passage in passages:
        pages = ", ".join(str(page) for page in passage.pages)
        year = passage.published[:4]
        blocks.append(
            f"[{passage.number}] Paper: {passage.paper_title} ({year})\n"
            f"Section {passage.section}, page {pages}\n"
            f"{neutralise_references(passage.text)}"
        )

    context = "\n\n".join(blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_OUTPUT_TOKENS,
            },
        },
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def call_gemini(prompt: str, api_key: str) -> str:
    response = requests.post(
        GEMINI_URL,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        },
        timeout=GEMINI_TIMEOUT,
    )

    if response.status_code >= 400:
        detail = response.json().get("error", {}).get("message", response.text)
        raise RuntimeError(f"Gemini returned {response.status_code}: {detail}")

    payload = response.json()
    candidates = payload.get("candidates", [])
    if not candidates:
        reason = payload.get("promptFeedback", {}).get("blockReason", "no candidates")
        raise RuntimeError(f"Gemini returned no answer: {reason}")

    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts).strip()


def extract_citations(text: str) -> list[int]:
    found = set()
    for match in INLINE_REFERENCE.finditer(text):
        for piece in match.group(1).split(","):
            value = piece.strip()
            if value.isdigit():
                found.add(int(value))
    return sorted(found)


def find_uncited_sentences(text: str) -> list[str]:
    uncited = []
    for raw in text.replace("\n", " ").split("."):
        sentence = raw.strip()
        if len(sentence.split()) < 6:
            continue
        if "[" not in sentence:
            uncited.append(sentence)
    return uncited


def answer_question(
    model: SentenceTransformer,
    reranker: CrossEncoder,
    bm25: BM25Okapi,
    chunks: list[dict],
    vectors: np.ndarray,
    question: str,
    provider: str,
    api_key: str | None,
    paper: str | None,
    year: str | None,
) -> Answer:
    passages, retrieval_seconds = retrieve(
        model, reranker, bm25, chunks, vectors, question, paper, year
    )

    if not passages:
        return Answer(
            text="NOT_IN_CONTEXT No passages matched this question.",
            provider=provider,
            model="",
            retrieval_seconds=retrieval_seconds,
            generation_seconds=0.0,
            refused=True,
        )

    prompt = build_prompt(question, passages)

    started = time.perf_counter()
    if provider == "ollama":
        text = call_ollama(prompt)
        model_name = OLLAMA_MODEL
    else:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment")
        text = call_gemini(prompt, api_key)
        model_name = GEMINI_MODEL
    generation_seconds = time.perf_counter() - started

    cited = extract_citations(text)
    valid = [number for number in cited if 1 <= number <= len(passages)]
    invalid = [number for number in cited if number not in valid]

    return Answer(
        text=text,
        provider=provider,
        model=model_name,
        retrieval_seconds=retrieval_seconds,
        generation_seconds=generation_seconds,
        passages=passages,
        cited=valid,
        invalid_citations=invalid,
        uncited_sentences=find_uncited_sentences(text),
        refused=text.upper().startswith("NOT_IN_CONTEXT"),
    )


def print_answer(question: str, answer: Answer) -> None:
    console.print()
    console.print(f"[bold cyan]Question:[/] {question}")
    console.print(
        f"[dim]{answer.provider} / {answer.model} / "
        f"retrieval {answer.retrieval_seconds:.2f}s / "
        f"generation {answer.generation_seconds:.1f}s[/dim]"
    )
    console.print()

    if answer.refused:
        console.print(f"[bold yellow]{answer.text}[/]")
    else:
        console.print(answer.text)

    if not answer.passages:
        return

    table = Table(title="Retrieved passages")
    table.add_column("#", justify="right")
    table.add_column("Cited", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Paper")
    table.add_column("Year", justify="right")
    table.add_column("Section")
    table.add_column("Pg", justify="right")

    for passage in answer.passages:
        table.add_row(
            str(passage.number),
            "yes" if passage.number in answer.cited else "",
            f"{passage.score:.2f}",
            passage.paper_title[:38],
            passage.published[:4],
            passage.section[:26],
            ",".join(str(page) for page in passage.pages),
        )

    console.print()
    console.print(table)

    if answer.invalid_citations:
        console.print(
            f"[bold red]Invalid citations:[/] {answer.invalid_citations} "
            f"(only 1 to {len(answer.passages)} exist)"
        )

    if answer.uncited_sentences and not answer.refused:
        console.print("[bold yellow]Sentences without a citation:[/]")
        for sentence in answer.uncited_sentences:
            console.print(f"  {sentence[:110]}")


def main() -> None:
    load_dotenv()

    arguments = sys.argv[1:]
    provider = DEFAULT_PROVIDER
    paper = None
    year = None
    remaining = []
    skip = False

    for position, argument in enumerate(arguments):
        if skip:
            skip = False
            continue
        if argument == "--ollama":
            provider = "ollama"
        elif argument == "--gemini":
            provider = "gemini"
        elif argument == "--paper" and position + 1 < len(arguments):
            paper = arguments[position + 1]
            skip = True
        elif argument == "--year" and position + 1 < len(arguments):
            year = arguments[position + 1]
            skip = True
        else:
            remaining.append(argument)

    question = " ".join(remaining).strip()

    if not INDEX_PATH.exists():
        console.print(f"[bold red]Index not found:[/] {INDEX_PATH.resolve()}")
        console.print("Run build_index.py first.")
        return

    api_key = os.getenv("GEMINI_API_KEY")
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

    questions = [question] if question else SAMPLE_QUESTIONS

    for item in questions:
        try:
            answer = answer_question(
                model, reranker, bm25, chunks, vectors, item,
                provider, api_key, paper, year,
            )
            print_answer(item, answer)
        except requests.exceptions.ConnectionError:
            console.print(f"[bold red]Cannot reach {provider}.[/]")
        except requests.exceptions.Timeout:
            console.print(f"[bold red]{provider} timed out.[/]")
        except RuntimeError as error:
            console.print(f"[bold red]{error}[/]")


if __name__ == "__main__":
    main()