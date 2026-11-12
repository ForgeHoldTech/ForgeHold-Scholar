from __future__ import annotations

import ast
import json
import operator
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from sentence_transformers import CrossEncoder, SentenceTransformer

from corpus_answer import (
    EMBEDDING_MODEL,
    INDEX_PATH,
    RERANKER_MODEL,
    Passage,
    call_gemini,
    call_ollama,
    extract_citations,
    load_index,
    neutralise_references,
    retrieve,
    tokenize,
)


CATALOG_PATH = Path("data/catalog.json")
REPORT_PATH = Path("output/ingest_report.json")
DATABASE_PATH = Path("output/catalog.db")
TRACE_PATH = Path("output/agent_traces.json")

MAX_STEPS = 6
SEARCH_RESULTS = 5
SQL_ROW_LIMIT = 40
MAX_OBSERVATION_CHARS = 6000
MAX_EXPONENT = 100
REQUEST_DELAY = 2.0
RETRY_DELAY = 30.0
MAX_RETRIES = 3

FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

SCHEMA = """
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT,
    year INTEGER,
    published TEXT,
    primary_category TEXT,
    first_author TEXT,
    author_count INTEGER,
    pages INTEGER,
    sections INTEGER,
    paragraphs INTEGER,
    chunks INTEGER
);
CREATE TABLE authors (
    paper_id TEXT,
    position INTEGER,
    name TEXT
);
CREATE TABLE categories (
    paper_id TEXT,
    category TEXT
);
"""

AGENT_PROMPT = """You are a research assistant working over a private collection \
of {paper_count} research papers. You solve the question step by step by calling tools.

Tools:
- search_papers: find passages inside the papers.
  Arguments: {{"query": "...", "paper": "optional title fragment or arXiv id", \
"year": "optional four digit year"}}
- query_catalog: run one read-only SQL SELECT over the paper catalog (SQLite).
  Arguments: {{"sql": "..."}}
  Tables:
    papers(paper_id, title, year, published, primary_category, first_author, \
author_count, pages, sections, paragraphs, chunks)
    authors(paper_id, position, name)
    categories(paper_id, category)
- calculator: evaluate an arithmetic expression.
  Arguments: {{"expression": "..."}}
- finish: give the final answer.
  Arguments: {{"answer": "..."}}

Rules:
- Reply with exactly one JSON object and nothing else:
  {{"thought": "...", "tool": "...", "arguments": {{}}}}
- Use query_catalog for questions about counts, years, authors, categories or \
paper sizes. Use search_papers for questions about what the papers say.
- Facts about paper content must come from search_papers results and be cited \
with the passage number in square brackets, like [3].
- Facts from query_catalog or calculator need no citation.
- Never invent passage numbers, paper titles or figures.
- If the tools cannot answer the question, call finish with an answer that \
starts with NOT_IN_CONTEXT.
- You have at most {max_steps} steps. Call finish as soon as you have enough.
- Write the final answer in plain text.

Question: {question}

Steps so far:
{history}

Next step as JSON:"""

SAMPLE_QUESTIONS = [
    "How many papers in the collection were published in 2025, and what are their titles?",
    "Which paper has the most pages, and what problem does it address?",
    "Compare what BERT and the original Retrieval-Augmented Generation paper say about pre-training.",
    "What is the population of Lahore?",
]


console = Console()


@dataclass
class Step:
    number: int
    thought: str
    tool: str
    arguments: dict
    observation: str
    seconds: float


@dataclass
class Run:
    question: str
    answer: str
    finished: bool
    seconds: float
    steps: list[Step] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)


def build_catalog(chunks: list[dict]) -> sqlite3.Connection:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    report: dict[str, dict] = {}
    if REPORT_PATH.exists():
        payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        report = {entry["arxiv_id"]: entry for entry in payload.get("papers", [])}

    chunk_counts = Counter(chunk["paper_id"] for chunk in chunks)

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    writer = sqlite3.connect(DATABASE_PATH)
    writer.executescript(SCHEMA)

    for paper in catalog.get("papers", []):
        paper_id = paper["arxiv_id"]
        if paper_id not in chunk_counts:
            continue

        stats = report.get(paper_id, {})
        published = paper.get("published", "")[:10]
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = paper.get("authors", [])
        categories = paper.get("categories", [])

        writer.execute(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                paper_id,
                paper.get("title", ""),
                year,
                published,
                categories[0] if categories else "",
                authors[0] if authors else "",
                len(authors),
                stats.get("pages"),
                stats.get("sections"),
                stats.get("paragraphs"),
                chunk_counts[paper_id],
            ),
        )
        writer.executemany(
            "INSERT INTO authors VALUES (?, ?, ?)",
            [(paper_id, position, name) for position, name in enumerate(authors, start=1)],
        )
        writer.executemany(
            "INSERT INTO categories VALUES (?, ?)",
            [(paper_id, category) for category in categories],
        )

    writer.commit()
    writer.close()

    return sqlite3.connect(f"file:{DATABASE_PATH.as_posix()}?mode=ro", uri=True)


def evaluate_node(node: ast.AST) -> float:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = evaluate_node(node.left)
        right = evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent is too large")
        return OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = evaluate_node(node.operand)
        return -value if isinstance(node.op, ast.USub) else value

    raise ValueError("only numbers and arithmetic operators are allowed")


class Toolbox:
    def __init__(
        self,
        model: SentenceTransformer,
        reranker: CrossEncoder,
        bm25: BM25Okapi,
        chunks: list[dict],
        vectors: np.ndarray,
        connection: sqlite3.Connection,
    ) -> None:
        self.model = model
        self.reranker = reranker
        self.bm25 = bm25
        self.chunks = chunks
        self.vectors = vectors
        self.connection = connection
        self.passages: list[Passage] = []
        self.numbers: dict[str, int] = {}
        self.handlers = {
            "search_papers": self.search_papers,
            "query_catalog": self.query_catalog,
            "calculator": self.calculator,
        }

    def reset(self) -> None:
        self.passages = []
        self.numbers = {}

    def search_papers(self, arguments: dict) -> str:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "Error: the query argument is required."

        paper = str(arguments.get("paper") or "").strip() or None
        year = str(arguments.get("year") or "").strip() or None

        found, _ = retrieve(
            self.model, self.reranker, self.bm25, self.chunks, self.vectors,
            query, paper, year,
        )
        if not found:
            return "No passages matched. Try a different query or remove the filters."

        blocks = []
        for passage in found[:SEARCH_RESULTS]:
            number = self.numbers.get(passage.chunk_id)
            if number is None:
                number = len(self.passages) + 1
                self.numbers[passage.chunk_id] = number
                self.passages.append(replace(passage, number=number))

            pages = ", ".join(str(page) for page in passage.pages)
            blocks.append(
                f"[{number}] Paper: {passage.paper_title} ({passage.published[:4]})\n"
                f"Section {passage.section}, page {pages}\n"
                f"{neutralise_references(passage.text)}"
            )

        return "\n\n".join(blocks)

    def query_catalog(self, arguments: dict) -> str:
        sql = str(arguments.get("sql") or "").strip().rstrip(";").strip()
        if not sql:
            return "Error: the sql argument is required."
        if ";" in sql:
            return "Error: only one statement is allowed."
        if not sql.lower().startswith(("select", "with")):
            return "Error: only SELECT statements are allowed."
        if FORBIDDEN_SQL.search(sql):
            return "Error: the statement contains a keyword that is not allowed."

        try:
            cursor = self.connection.execute(sql)
            rows = cursor.fetchmany(SQL_ROW_LIMIT + 1)
        except sqlite3.Error as error:
            return f"SQL error: {error}"

        if not rows:
            return "The query returned no rows."

        columns = [column[0] for column in cursor.description or []]
        lines = [" | ".join(columns)]
        for row in rows[:SQL_ROW_LIMIT]:
            lines.append(" | ".join(str(value) for value in row))
        if len(rows) > SQL_ROW_LIMIT:
            lines.append(f"(only the first {SQL_ROW_LIMIT} rows are shown)")

        return "\n".join(lines)

    def calculator(self, arguments: dict) -> str:
        expression = str(arguments.get("expression") or "").strip()
        if not expression:
            return "Error: the expression argument is required."

        try:
            tree = ast.parse(expression, mode="eval")
            value = evaluate_node(tree.body)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as error:
            return f"Calculation error: {error}"

        return str(value)


def call_model(prompt: str, provider: str, api_key: str | None) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == "ollama":
                return call_ollama(prompt)
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set in the environment")
            return call_gemini(prompt, api_key)
        except RuntimeError as error:
            if "429" in str(error) and attempt < MAX_RETRIES:
                console.print(f"[yellow]Rate limited, waiting {RETRY_DELAY:.0f}s[/]")
                time.sleep(RETRY_DELAY)
                continue
            raise
    return ""


def parse_action(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        payload = json.loads(text[start:end + 1], strict=False)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    tool = payload.get("tool")
    arguments = payload.get("arguments", {})
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        return None

    return {
        "thought": str(payload.get("thought", "")),
        "tool": tool.strip(),
        "arguments": arguments,
    }


def format_history(steps: list[Step]) -> str:
    if not steps:
        return "(none yet)"

    blocks = []
    for step in steps:
        observation = step.observation
        if len(observation) > MAX_OBSERVATION_CHARS:
            observation = observation[:MAX_OBSERVATION_CHARS] + "\n(truncated)"
        blocks.append(
            f"Step {step.number}\n"
            f"Thought: {step.thought}\n"
            f"Tool: {step.tool}\n"
            f"Arguments: {json.dumps(step.arguments, ensure_ascii=False)}\n"
            f"Observation:\n{observation}"
        )
    return "\n\n".join(blocks)


def run_agent(
    question: str,
    toolbox: Toolbox,
    provider: str,
    api_key: str | None,
    paper_count: int,
) -> Run:
    toolbox.reset()
    steps: list[Step] = []
    answer = ""
    finished = False
    started = time.perf_counter()

    for number in range(1, MAX_STEPS + 1):
        history = format_history(steps)
        if number == MAX_STEPS:
            history += "\n\nThis is your final step. You must call finish now."

        prompt = AGENT_PROMPT.format(
            paper_count=paper_count,
            max_steps=MAX_STEPS,
            question=question,
            history=history,
        )

        step_started = time.perf_counter()
        response = call_model(prompt, provider, api_key)
        action = parse_action(response)

        if action is None:
            steps.append(
                Step(
                    number=number,
                    thought="",
                    tool="invalid",
                    arguments={},
                    observation=(
                        "Your reply was not a valid JSON object with the keys "
                        "thought, tool and arguments. Reply with one JSON object only."
                    ),
                    seconds=time.perf_counter() - step_started,
                )
            )
            time.sleep(REQUEST_DELAY)
            continue

        tool = action["tool"]
        arguments = action["arguments"]

        if tool == "finish":
            answer = str(arguments.get("answer") or "").strip()
            steps.append(
                Step(
                    number=number,
                    thought=action["thought"],
                    tool="finish",
                    arguments={},
                    observation="",
                    seconds=time.perf_counter() - step_started,
                )
            )
            finished = bool(answer)
            break

        handler = toolbox.handlers.get(tool)
        if handler is None:
            available = ", ".join(sorted(toolbox.handlers)) + ", finish"
            observation = f"Error: unknown tool '{tool}'. Available tools: {available}."
        else:
            observation = handler(arguments)

        steps.append(
            Step(
                number=number,
                thought=action["thought"],
                tool=tool,
                arguments=arguments,
                observation=observation,
                seconds=time.perf_counter() - step_started,
            )
        )
        time.sleep(REQUEST_DELAY)

    if not finished:
        answer = "NOT_IN_CONTEXT The agent did not reach an answer within the step limit."

    cited = extract_citations(answer)
    valid = [number for number in cited if 1 <= number <= len(toolbox.passages)]
    invalid = [number for number in cited if number not in valid]

    return Run(
        question=question,
        answer=answer,
        finished=finished,
        seconds=time.perf_counter() - started,
        steps=steps,
        passages=list(toolbox.passages),
        cited=valid,
        invalid_citations=invalid,
    )


def shorten(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def print_run(run: Run) -> None:
    console.print()
    console.print(f"[bold cyan]Question:[/] {escape(run.question)}")

    table = Table(title=f"Agent trace ({run.seconds:.1f}s)")
    table.add_column("Step", justify="right")
    table.add_column("Tool")
    table.add_column("Input")
    table.add_column("Result")
    table.add_column("Sec", justify="right")

    for step in run.steps:
        arguments = json.dumps(step.arguments, ensure_ascii=False) if step.arguments else ""
        table.add_row(
            str(step.number),
            escape(step.tool),
            escape(shorten(arguments, 70)),
            escape(shorten(step.observation, 60)),
            f"{step.seconds:.1f}",
        )

    console.print(table)
    console.print()

    if run.answer.upper().startswith("NOT_IN_CONTEXT"):
        console.print(run.answer, style="bold yellow", markup=False)
    else:
        console.print(run.answer, markup=False)

    cited_passages = [passage for passage in run.passages if passage.number in run.cited]
    if cited_passages:
        sources = Table(title="Cited sources")
        sources.add_column("#", justify="right")
        sources.add_column("Paper")
        sources.add_column("Year", justify="right")
        sources.add_column("Section")
        sources.add_column("Pg", justify="right")

        for passage in cited_passages:
            sources.add_row(
                str(passage.number),
                escape(passage.paper_title[:48]),
                passage.published[:4],
                escape(passage.section[:28]),
                ",".join(str(page) for page in passage.pages),
            )

        console.print()
        console.print(sources)

    if run.invalid_citations:
        console.print(
            f"[bold red]Invalid citations:[/] {run.invalid_citations} "
            f"(only 1 to {len(run.passages)} exist)"
        )


def save_traces(runs: list[Run], provider: str) -> None:
    payload = {
        "provider": provider,
        "run_count": len(runs),
        "runs": [
            {
                "question": run.question,
                "answer": run.answer,
                "finished": run.finished,
                "seconds": round(run.seconds, 2),
                "cited": run.cited,
                "invalid_citations": run.invalid_citations,
                "steps": [
                    {
                        "number": step.number,
                        "thought": step.thought,
                        "tool": step.tool,
                        "arguments": step.arguments,
                        "observation": step.observation[:MAX_OBSERVATION_CHARS],
                        "seconds": round(step.seconds, 2),
                    }
                    for step in run.steps
                ],
                "passages": [
                    {
                        "number": passage.number,
                        "chunk_id": passage.chunk_id,
                        "paper_title": passage.paper_title,
                        "section": passage.section,
                        "pages": passage.pages,
                    }
                    for passage in run.passages
                ],
            }
            for run in runs
        ],
    }
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    load_dotenv()

    arguments = sys.argv[1:]
    provider = "gemini"
    remaining = []
    for argument in arguments:
        if argument == "--ollama":
            provider = "ollama"
        elif argument == "--gemini":
            provider = "gemini"
        else:
            remaining.append(argument)

    question = " ".join(remaining).strip()

    if not INDEX_PATH.exists():
        console.print(f"[bold red]Index not found:[/] {INDEX_PATH.resolve()}")
        console.print("Run build_index.py first.")
        return

    if not CATALOG_PATH.exists():
        console.print(f"[bold red]Catalog not found:[/] {CATALOG_PATH.resolve()}")
        console.print("Run fetch_papers.py first.")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    index, chunks, vectors = load_index()
    connection = build_catalog(chunks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold cyan]Device:[/] {device}")
    console.print(
        f"[bold cyan]Corpus:[/] {index['document_count']} papers, "
        f"{index['chunk_count']:,} chunks"
    )
    console.print(f"[bold cyan]Catalog database:[/] {DATABASE_PATH}")

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    reranker = CrossEncoder(RERANKER_MODEL, device=device)
    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    toolbox = Toolbox(model, reranker, bm25, chunks, vectors, connection)

    questions = [question] if question else SAMPLE_QUESTIONS
    runs: list[Run] = []

    for item in questions:
        try:
            run = run_agent(item, toolbox, provider, api_key, index["document_count"])
        except requests.exceptions.ConnectionError:
            console.print(f"[bold red]Cannot reach {provider}.[/]")
            continue
        except requests.exceptions.Timeout:
            console.print(f"[bold red]{provider} timed out.[/]")
            continue
        except RuntimeError as error:
            console.print(f"[bold red]{escape(str(error))}[/]")
            continue

        runs.append(run)
        print_run(run)

    connection.close()

    if runs:
        save_traces(runs, provider)
        console.print()
        console.print(f"[bold green]Saved:[/] {TRACE_PATH}")


if __name__ == "__main__":
    main()