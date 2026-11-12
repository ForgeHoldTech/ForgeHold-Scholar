from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table


PAPERS_DIR = Path("data/papers")
CATALOG_PATH = Path("data/catalog.json")

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"

REQUEST_DELAY = 3.0
REQUEST_TIMEOUT = 60
PAPERS_PER_QUERY = 6

NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

QUERIES = [
    "all:\"attention mechanism\" AND cat:cs.CL",
    "all:\"retrieval augmented generation\" AND cat:cs.CL",
    "all:\"language model\" AND cat:cs.CL",
    "all:\"question answering\" AND cat:cs.CL",
    "all:\"dense retrieval\" AND cat:cs.IR",
]

SEED_PAPERS = [
    "1706.03762",
    "1810.04805",
    "2005.11401",
]


console = Console()


@dataclass
class Paper:
    arxiv_id: str
    version: str
    title: str
    authors: list[str]
    published: str
    updated: str
    categories: list[str]
    summary: str
    filename: str


def clean(text: str) -> str:
    return " ".join(text.split())


def safe_filename(arxiv_id: str) -> str:
    stem = arxiv_id.replace("/", "_").replace("\\", "_")
    return f"{stem}.pdf"


def parse_entry(entry: ElementTree.Element) -> Paper | None:
    identifier = entry.findtext("atom:id", default="", namespaces=NAMESPACES)
    if not identifier:
        return None

    tail = identifier.rsplit("/abs/", 1)[-1]
    head, separator, suffix = tail.rpartition("v")
    if separator and suffix.isdigit():
        arxiv_id, version = head, f"v{suffix}"
    else:
        arxiv_id, version = tail, ""

    authors = [
        clean(node.findtext("atom:name", default="", namespaces=NAMESPACES))
        for node in entry.findall("atom:author", NAMESPACES)
    ]

    categories = [
        node.attrib.get("term", "")
        for node in entry.findall("atom:category", NAMESPACES)
    ]

    return Paper(
        arxiv_id=arxiv_id,
        version=version,
        title=clean(entry.findtext("atom:title", default="", namespaces=NAMESPACES)),
        authors=[name for name in authors if name],
        published=entry.findtext("atom:published", default="", namespaces=NAMESPACES),
        updated=entry.findtext("atom:updated", default="", namespaces=NAMESPACES),
        categories=[name for name in categories if name],
        summary=clean(entry.findtext("atom:summary", default="", namespaces=NAMESPACES)),
        filename=safe_filename(arxiv_id),
    )


def search_arxiv(query: str, limit: int) -> list[Paper]:
    response = requests.get(
        ARXIV_API,
        params={
            "search_query": query,
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    root = ElementTree.fromstring(response.content)
    papers = []
    for entry in root.findall("atom:entry", NAMESPACES):
        paper = parse_entry(entry)
        if paper:
            papers.append(paper)
    return papers


def fetch_by_id(arxiv_ids: list[str]) -> list[Paper]:
    response = requests.get(
        ARXIV_API,
        params={"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    root = ElementTree.fromstring(response.content)
    papers = []
    for entry in root.findall("atom:entry", NAMESPACES):
        paper = parse_entry(entry)
        if paper:
            papers.append(paper)
    return papers


def download_pdf(paper: Paper) -> tuple[bool, str]:
    target = PAPERS_DIR / paper.filename
    if target.exists():
        return True, "already present"

    try:
        response = requests.get(
            ARXIV_PDF.format(arxiv_id=paper.arxiv_id),
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "ForgeHoldScholar/1.0"},
        )
        response.raise_for_status()
    except requests.RequestException as error:
        return False, str(error)[:60]

    content = response.content
    if not content.startswith(b"%PDF"):
        return False, "response was not a PDF"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    except OSError as error:
        return False, str(error)[:60]

    return True, f"{len(content) / 1_000_000:.1f} MB"


def main() -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    collected: dict[str, Paper] = {}

    console.print(f"[bold cyan]Fetching seed papers:[/] {len(SEED_PAPERS)}")
    try:
        for paper in fetch_by_id(SEED_PAPERS):
            collected[paper.arxiv_id] = paper
    except requests.RequestException as error:
        console.print(f"[yellow]Seed fetch failed:[/] {error}")
    time.sleep(REQUEST_DELAY)

    for query in QUERIES:
        console.print(f"[bold cyan]Searching:[/] {query}")
        try:
            found = search_arxiv(query, PAPERS_PER_QUERY)
        except requests.RequestException as error:
            console.print(f"[yellow]Search failed:[/] {error}")
            time.sleep(REQUEST_DELAY)
            continue

        for paper in found:
            collected.setdefault(paper.arxiv_id, paper)
        console.print(f"  found {len(found)}, unique so far {len(collected)}")
        time.sleep(REQUEST_DELAY)

    if not collected:
        console.print("[bold red]No papers found.[/]")
        return

    console.print()
    console.print(f"[bold cyan]Downloading {len(collected)} PDFs[/]")

    results = []
    for position, paper in enumerate(collected.values(), start=1):
        success, detail = download_pdf(paper)
        marker = "ok" if success else "failed"
        console.print(
            f"  [{position}/{len(collected)}] {paper.arxiv_id} {marker} ({detail})"
        )
        results.append((paper, success, detail))
        if detail != "already present":
            time.sleep(REQUEST_DELAY)

    downloaded = [paper for paper, success, _ in results if success]

    catalog = {
        "paper_count": len(downloaded),
        "papers": [asdict(paper) for paper in downloaded],
    }
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = Table(title="Downloaded papers")
    summary.add_column("arXiv ID")
    summary.add_column("Title")
    summary.add_column("Category")
    summary.add_column("Year", justify="right")

    for paper in downloaded:
        summary.add_row(
            paper.arxiv_id,
            paper.title[:55],
            paper.categories[0] if paper.categories else "",
            paper.published[:4],
        )

    console.print()
    console.print(summary)

    failures = [(paper, detail) for paper, success, detail in results if not success]
    if failures:
        console.print()
        console.print("[bold yellow]Failed downloads:[/]")
        for paper, detail in failures:
            console.print(f"  {paper.arxiv_id}: {detail}")

    console.print()
    console.print(f"[bold green]Saved:[/] {CATALOG_PATH}")
    console.print(f"[bold green]PDFs in:[/] {PAPERS_DIR}")


if __name__ == "__main__":
    main()