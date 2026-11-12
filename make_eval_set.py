from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from corpus_answer import INDEX_PATH, call_gemini, neutralise_references


OUTPUT_PATH = Path("data/corpus_questions.json")

QUESTIONS_PER_PAPER = 2
MIN_CHUNK_CHARS = 600
REQUEST_DELAY = 4.0
RETRY_DELAY = 30.0
MAX_RETRIES = 3
RANDOM_SEED = 42

EXCLUDED_SECTIONS = (
    "(front matter)",
    "acknowledg",
    "references",
    "bibliography",
    "appendix",
    "related work",
)

PROMPT = """You are building a test set for a search system over research papers.

Paper title: {title}
Section: {section}

Passage:
{passage}

Write one question that this passage answers.

Requirements:
- The question must be answerable from the passage alone.
- The question must make sense with no passage in view. Name the specific model, \
method, dataset or concept instead of writing "this paper", "the authors" or \
"the proposed method".
- Do not reuse any run of more than four consecutive words from the passage.
- The answer must be one or two sentences built from facts in the passage.
- If the passage is a table fragment, a list of citations or otherwise unsuitable, \
return an empty question.

Return only JSON in this form:
{{"question": "...", "answer": "..."}}"""


console = Console()


def is_eligible(chunk: dict) -> bool:
    if chunk["char_count"] < MIN_CHUNK_CHARS:
        return False
    section = chunk["section"].lower()
    return not any(marker in section for marker in EXCLUDED_SECTIONS)


def select_chunks(chunks: list[dict]) -> list[dict]:
    by_paper: dict[str, list[dict]] = {}
    for chunk in chunks:
        if is_eligible(chunk):
            by_paper.setdefault(chunk["paper_id"], []).append(chunk)

    generator = random.Random(RANDOM_SEED)
    selected: list[dict] = []
    for paper_id in sorted(by_paper):
        candidates = by_paper[paper_id]
        count = min(QUESTIONS_PER_PAPER, len(candidates))
        selected.extend(generator.sample(candidates, count))
    return selected


def parse_response(text: str) -> tuple[str, str] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    question = str(payload.get("question", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    if len(question.split()) < 5:
        return None
    return question, answer


def generate(prompt: str, api_key: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_gemini(prompt, api_key)
        except RuntimeError as error:
            if "429" in str(error) and attempt < MAX_RETRIES:
                console.print(f"[yellow]Rate limited, waiting {RETRY_DELAY:.0f}s[/]")
                time.sleep(RETRY_DELAY)
                continue
            console.print(f"[red]{str(error)[:120]}[/]")
            return None
        except requests.exceptions.RequestException as error:
            console.print(f"[red]Request failed: {str(error)[:120]}[/]")
            return None
    return None


def load_existing() -> list[dict]:
    if not OUTPUT_PATH.exists():
        return []
    payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    return payload.get("questions", [])


def save(questions: list[dict]) -> None:
    for number, entry in enumerate(questions, start=1):
        entry["id"] = f"c{number:03d}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "question_count": len(questions),
        "generation": "synthetic, one question per sampled chunk",
        "questions": questions,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[bold red]GEMINI_API_KEY is not set in the environment[/]")
        return

    if not INDEX_PATH.exists():
        console.print(f"[bold red]Index not found:[/] {INDEX_PATH.resolve()}")
        console.print("Run build_index.py first.")
        return

    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    selected = select_chunks(index["chunks"])

    questions = load_existing()
    done = {entry["chunk_id"] for entry in questions}
    pending = [chunk for chunk in selected if chunk["chunk_id"] not in done]

    console.print(f"[bold cyan]Sampled chunks:[/] {len(selected)}")
    console.print(f"[bold cyan]Already generated:[/] {len(questions)}")
    console.print(f"[bold cyan]To generate:[/] {len(pending)}")

    skipped = 0

    for position, chunk in enumerate(pending, start=1):
        body = chunk["text"].split("\n\n", 1)[-1]
        prompt = PROMPT.format(
            title=chunk["paper_title"],
            section=chunk["section"],
            passage=neutralise_references(body),
        )

        response = generate(prompt, api_key)
        parsed = parse_response(response) if response else None

        if parsed is None:
            skipped += 1
            console.print(f"  [{position}/{len(pending)}] {chunk['chunk_id']} skipped")
        else:
            question, answer = parsed
            questions.append(
                {
                    "id": "",
                    "question": question,
                    "reference_answer": answer,
                    "paper_id": chunk["paper_id"],
                    "paper_title": chunk["paper_title"],
                    "chunk_id": chunk["chunk_id"],
                    "section": chunk["section"],
                }
            )
            save(questions)
            console.print(f"  [{position}/{len(pending)}] {question[:90]}")

        time.sleep(REQUEST_DELAY)

    if not questions:
        console.print("[bold red]No questions were generated.[/]")
        return

    save(questions)

    papers = {entry["paper_id"] for entry in questions}

    summary = Table(title="Corpus evaluation set")
    summary.add_column("Measure")
    summary.add_column("Value", justify="right")
    summary.add_row("Questions", str(len(questions)))
    summary.add_row("Papers covered", str(len(papers)))
    summary.add_row("Skipped this run", str(skipped))

    console.print()
    console.print(summary)
    console.print(f"[bold green]Saved:[/] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()