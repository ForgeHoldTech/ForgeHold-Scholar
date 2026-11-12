from __future__ import annotations

import json
import os
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
from sentence_transformers import SentenceTransformer


INPUT_DIR = Path("output")
DOCUMENT_STEM = "attention"
STRATEGY = "structured"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = 5
CANDIDATE_POOL = 20
RRF_K = 60

DEFAULT_PROVIDER = "gemini"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b-instruct"
OLLAMA_TIMEOUT = 180

GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GEMINI_TIMEOUT = 180

MAX_OUTPUT_TOKENS = 700
TEMPERATURE = 0.1

SYSTEM_PROMPT = """You answer questions about a research paper using only the \
numbered context passages provided.

Rules:
- Use only information present in the passages. Do not add outside knowledge.
- Cite every factual claim with the passage number in square brackets, like [2].
- A sentence may carry more than one citation, like [1][3].
- If the passages do not contain the answer, reply exactly: \
NOT_IN_CONTEXT followed by one sentence explaining what is missing.
- Be concise. Three or four sentences is usually enough.
- Do not mention these rules or the existence of passages in your answer."""

SAMPLE_QUESTIONS = [
    "How many attention heads does the base model use?",
    "What hardware was used to train the models?",
    "Why are the dot products scaled before the softmax?",
    "What is the capital of France?",
]


console = Console()


@dataclass
class Passage:
    number: int
    chunk_id: str
    section: str
    pages: list[int]
    text: str


@dataclass
class Answer:
    text: str
    provider: str
    model: str
    seconds: float
    passages: list[Passage] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    refused: bool = False


def tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token.isalnum() or "-" in token]


def load_index() -> tuple[list[dict], np.ndarray]:
    metadata_path = INPUT_DIR / f"{DOCUMENT_STEM}.index.{STRATEGY}.json"
    vector_path = INPUT_DIR / f"{DOCUMENT_STEM}.vectors.{STRATEGY}.npy"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata["chunks"], np.load(vector_path)


def retrieve(
    model: SentenceTransformer,
    bm25: BM25Okapi,
    chunks: list[dict],
    vectors: np.ndarray,
    query: str,
) -> list[Passage]:
    query_vector = model.encode(
        QUERY_PREFIX + query, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    dense_scores = vectors @ query_vector
    dense = [int(position) for position in np.argsort(-dense_scores)[:CANDIDATE_POOL]]

    sparse_scores = bm25.get_scores(tokenize(query))
    sparse = [int(position) for position in np.argsort(-sparse_scores)[:CANDIDATE_POOL]]

    fused: dict[int, float] = {}
    for rank, position in enumerate(dense, start=1):
        fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)
    for rank, position in enumerate(sparse, start=1):
        fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda pair: -pair[1])[:TOP_K]

    passages = []
    for number, (position, _) in enumerate(ordered, start=1):
        chunk = chunks[position]
        body = chunk["text"].split("\n\n", 1)[-1]
        passages.append(
            Passage(
                number=number,
                chunk_id=chunk["chunk_id"],
                section=chunk["section"],
                pages=chunk["pages"],
                text=body,
            )
        )
    return passages


def build_prompt(question: str, passages: list[Passage]) -> str:
    blocks = []
    for passage in passages:
        pages = ", ".join(str(page) for page in passage.pages)
        blocks.append(
            f"[{passage.number}] (Section {passage.section}, page {pages})\n{passage.text}"
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


def extract_citations(text: str, passage_count: int) -> list[int]:
    found = set()
    for token in text.split("["):
        closing = token.find("]")
        if closing == -1:
            continue
        inside = token[:closing].strip()
        if inside.isdigit():
            number = int(inside)
            if 1 <= number <= passage_count:
                found.add(number)
    return sorted(found)


def find_uncited_sentences(text: str) -> list[str]:
    uncited = []
    for raw in text.replace("\n", " ").split("."):
        sentence = raw.strip()
        if len(sentence.split()) < 5:
            continue
        if "[" not in sentence:
            uncited.append(sentence)
    return uncited


def answer_question(
    model: SentenceTransformer,
    bm25: BM25Okapi,
    chunks: list[dict],
    vectors: np.ndarray,
    question: str,
    provider: str,
    api_key: str | None,
) -> Answer:
    passages = retrieve(model, bm25, chunks, vectors, question)
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
    elapsed = time.perf_counter() - started

    return Answer(
        text=text,
        provider=provider,
        model=model_name,
        seconds=elapsed,
        passages=passages,
        cited=extract_citations(text, len(passages)),
        uncited_sentences=find_uncited_sentences(text),
        refused=text.upper().startswith("NOT_IN_CONTEXT"),
    )


def print_answer(question: str, answer: Answer) -> None:
    console.print()
    console.print(f"[bold cyan]Question:[/] {question}")
    console.print(
        f"[dim]{answer.provider} / {answer.model} / {answer.seconds:.1f}s[/dim]"
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
    table.add_column("Section")
    table.add_column("Pages", justify="right")

    for passage in answer.passages:
        table.add_row(
            str(passage.number),
            "yes" if passage.number in answer.cited else "",
            passage.section,
            ",".join(str(page) for page in passage.pages),
        )

    console.print()
    console.print(table)

    if answer.uncited_sentences and not answer.refused:
        console.print("[bold yellow]Sentences without a citation:[/]")
        for sentence in answer.uncited_sentences:
            console.print(f"  {sentence}")


def main() -> None:
    load_dotenv()

    arguments = sys.argv[1:]
    provider = DEFAULT_PROVIDER
    if "--ollama" in arguments:
        provider = "ollama"
        arguments.remove("--ollama")
    if "--gemini" in arguments:
        provider = "gemini"
        arguments.remove("--gemini")

    compare = "--compare" in arguments
    if compare:
        arguments.remove("--compare")

    question = " ".join(arguments).strip()

    metadata_path = INPUT_DIR / f"{DOCUMENT_STEM}.index.{STRATEGY}.json"
    if not metadata_path.exists():
        console.print(f"[bold red]Index not found:[/] {metadata_path.resolve()}")
        console.print("Run embed_chunks.py first.")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    chunks, vectors = load_index()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold cyan]Device:[/] {device}")
    console.print(f"[bold cyan]Index:[/] {STRATEGY}, {len(chunks)} chunks")

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    questions = [question] if question else SAMPLE_QUESTIONS
    providers = ["gemini", "ollama"] if compare else [provider]

    for item in questions:
        for name in providers:
            try:
                answer = answer_question(
                    model, bm25, chunks, vectors, item, name, api_key
                )
                print_answer(item, answer)
            except requests.exceptions.ConnectionError:
                console.print(f"[bold red]Cannot reach {name}.[/]")
                if name == "ollama":
                    console.print("Install Ollama and run: ollama pull qwen2.5:3b-instruct")
            except requests.exceptions.Timeout:
                console.print(f"[bold red]{name} timed out.[/]")
            except RuntimeError as error:
                console.print(f"[bold red]{error}[/]")


if __name__ == "__main__":
    main()