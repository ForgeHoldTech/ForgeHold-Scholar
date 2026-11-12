from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import requests
import torch
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table
from sentence_transformers import CrossEncoder, SentenceTransformer

from corpus_answer import (
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    INDEX_PATH,
    RERANKER_MODEL,
    answer_question,
    call_gemini,
    load_index,
    neutralise_references,
    tokenize,
)


QUESTIONS_PATH = Path("data/corpus_questions.json")
RESULTS_PATH = Path("output/answer_evaluation.json")

DEFAULT_LIMIT = 30
RANDOM_SEED = 42
REQUEST_DELAY = 4.0
RETRY_DELAY = 30.0
MAX_RETRIES = 3
MAX_PROBLEMS_SHOWN = 8

GRADES = {"yes", "partial", "no"}

JUDGE_PROMPT = """You are grading the answer of a question answering system \
over research papers.

Question:
{question}

Reference answer:
{reference}

Context passages given to the system:
{context}

System answer:
{answer}

Grade two things.

correctness: does the system answer agree with the reference answer?
- "yes" if it conveys the same key facts
- "partial" if it conveys only some of them or adds a minor error
- "no" if it contradicts the reference or misses the point

faithfulness: is every factual claim in the system answer supported by the \
context passages?
- "yes" if all claims are supported
- "partial" if most are supported and one is not
- "no" if several claims are unsupported

Return only JSON in this form:
{{"correctness": "yes", "faithfulness": "yes", "unsupported_claims": []}}"""


console = Console()


def with_retry(action):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return action()
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


def pick_questions(questions: list[dict], limit: int) -> list[dict]:
    if limit >= len(questions):
        return questions
    generator = random.Random(RANDOM_SEED)
    chosen = generator.sample(questions, limit)
    return sorted(chosen, key=lambda entry: entry["id"])


def parse_verdict(text: str) -> dict | None:
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

    correctness = str(payload.get("correctness", "")).strip().lower()
    faithfulness = str(payload.get("faithfulness", "")).strip().lower()
    if correctness not in GRADES or faithfulness not in GRADES:
        return None

    claims = payload.get("unsupported_claims", [])
    if not isinstance(claims, list):
        claims = []

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "unsupported_claims": [str(claim) for claim in claims],
    }


def build_judge_prompt(entry: dict, answer) -> str:
    blocks = [
        f"[{passage.number}] {passage.paper_title}\n"
        f"{neutralise_references(passage.text)}"
        for passage in answer.passages
    ]
    return JUDGE_PROMPT.format(
        question=entry["question"],
        reference=entry["reference_answer"],
        context="\n\n".join(blocks),
        answer=answer.text,
    )


def load_existing() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return payload.get("records", [])


def save(records: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator_model": GEMINI_MODEL,
        "judge_model": GEMINI_MODEL,
        "record_count": len(records),
        "records": records,
    }
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def percent(part: int, whole: int) -> str:
    if not whole:
        return "-"
    return f"{part} ({100 * part / whole:.0f}%)"


def print_summary(records: list[dict]) -> None:
    total = len(records)
    answered = [record for record in records if not record["refused"]]
    refused = [record for record in records if record["refused"]]
    false_refusals = [record for record in refused if record["source_retrieved"]]
    graded = [record for record in answered if record["correctness"] in GRADES]

    summary = Table(title=f"Answer evaluation ({total} questions)")
    summary.add_column("Measure")
    summary.add_column("Value", justify="right")

    summary.add_row("Answered", percent(len(answered), total))
    summary.add_row("Refused", percent(len(refused), total))
    summary.add_row(
        "Refused although source passage was retrieved",
        percent(len(false_refusals), total),
    )

    for grade in ("yes", "partial", "no"):
        count = sum(1 for record in graded if record["correctness"] == grade)
        summary.add_row(f"Correctness: {grade}", percent(count, len(graded)))

    for grade in ("yes", "partial", "no"):
        count = sum(1 for record in graded if record["faithfulness"] == grade)
        summary.add_row(f"Faithfulness: {grade}", percent(count, len(graded)))

    summary.add_row(
        "Answers with invalid citations",
        percent(
            sum(1 for record in answered if record["invalid_citations"]),
            len(answered),
        ),
    )
    summary.add_row(
        "Answers with every sentence cited",
        percent(
            sum(1 for record in answered if record["uncited_sentences"] == 0),
            len(answered),
        ),
    )
    summary.add_row(
        "Answers citing the source passage",
        percent(
            sum(1 for record in answered if record["source_cited"]),
            len(answered),
        ),
    )

    if answered:
        mean_seconds = sum(
            record["generation_seconds"] for record in answered
        ) / len(answered)
        summary.add_row("Mean generation time", f"{mean_seconds:.1f}s")

    console.print()
    console.print(summary)

    problems = [
        record
        for record in graded
        if record["correctness"] == "no" or record["faithfulness"] == "no"
    ]
    if not problems:
        return

    table = Table(title=f"Problem answers ({len(problems)})")
    table.add_column("ID")
    table.add_column("Question")
    table.add_column("Correct")
    table.add_column("Faithful")
    table.add_column("Unsupported claim")

    for record in problems[:MAX_PROBLEMS_SHOWN]:
        claims = record["unsupported_claims"]
        table.add_row(
            record["question_id"],
            record["question"][:48],
            record["correctness"],
            record["faithfulness"],
            claims[0][:48] if claims else "",
        )

    console.print()
    console.print(table)


def main() -> None:
    load_dotenv()

    limit = DEFAULT_LIMIT
    arguments = sys.argv[1:]
    for position, argument in enumerate(arguments):
        if argument == "--limit" and position + 1 < len(arguments):
            value = arguments[position + 1]
            if value.isdigit():
                limit = max(1, int(value))

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[bold red]GEMINI_API_KEY is not set in the environment[/]")
        return

    if not QUESTIONS_PATH.exists():
        console.print(f"[bold red]Questions not found:[/] {QUESTIONS_PATH.resolve()}")
        console.print("Run make_eval_set.py first.")
        return

    if not INDEX_PATH.exists():
        console.print(f"[bold red]Index not found:[/] {INDEX_PATH.resolve()}")
        console.print("Run build_index.py first.")
        return

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    selected = pick_questions(questions, limit)

    records = load_existing()
    done = {record["question_id"] for record in records}
    pending = [entry for entry in selected if entry["id"] not in done]

    index, chunks, vectors = load_index()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold cyan]Device:[/] {device}")
    console.print(f"[bold cyan]Selected questions:[/] {len(selected)}")
    console.print(f"[bold cyan]Already evaluated:[/] {len(selected) - len(pending)}")
    console.print(f"[bold cyan]To evaluate:[/] {len(pending)}")

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    reranker = CrossEncoder(RERANKER_MODEL, device=device)
    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    for position, entry in enumerate(pending, start=1):
        answer = with_retry(
            lambda: answer_question(
                model, reranker, bm25, chunks, vectors, entry["question"],
                "gemini", api_key, None, None,
            )
        )
        time.sleep(REQUEST_DELAY)

        if answer is None:
            console.print(f"  [{position}/{len(pending)}] {entry['id']} failed")
            continue

        source_number = next(
            (
                passage.number
                for passage in answer.passages
                if passage.chunk_id == entry["chunk_id"]
            ),
            None,
        )

        record = {
            "question_id": entry["id"],
            "question": entry["question"],
            "reference_answer": entry["reference_answer"],
            "answer": answer.text,
            "refused": answer.refused,
            "source_retrieved": source_number is not None,
            "source_cited": source_number is not None and source_number in answer.cited,
            "valid_citations": len(answer.cited),
            "invalid_citations": answer.invalid_citations,
            "uncited_sentences": len(answer.uncited_sentences),
            "correctness": "refused" if answer.refused else "ungraded",
            "faithfulness": "refused" if answer.refused else "ungraded",
            "unsupported_claims": [],
            "retrieval_seconds": round(answer.retrieval_seconds, 3),
            "generation_seconds": round(answer.generation_seconds, 3),
        }

        if not answer.refused:
            prompt = build_judge_prompt(entry, answer)
            response = with_retry(lambda: call_gemini(prompt, api_key))
            time.sleep(REQUEST_DELAY)

            verdict = parse_verdict(response) if response else None
            if verdict:
                record.update(verdict)

        records.append(record)
        save(records)

        console.print(
            f"  [{position}/{len(pending)}] {entry['id']} "
            f"correct={record['correctness']} faithful={record['faithfulness']}"
        )

    selected_ids = {entry["id"] for entry in selected}
    evaluated = [record for record in records if record["question_id"] in selected_ids]

    if not evaluated:
        console.print("[bold red]Nothing was evaluated.[/]")
        return

    print_summary(evaluated)
    console.print()
    console.print(f"[bold green]Saved:[/] {RESULTS_PATH}")


if __name__ == "__main__":
    main()