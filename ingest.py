from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import pymupdf
from rich.console import Console
from rich.table import Table


PAPERS_DIR = Path("data/papers")
CATALOG_PATH = Path("data/catalog.json")
OUTPUT_DIR = Path("output")
CORPUS_PATH = OUTPUT_DIR / "corpus.json"
REPORT_PATH = OUTPUT_DIR / "ingest_report.json"

SHORT_LINE_WORD_LIMIT = 3
FIGURE_PAGE_SHORT_LINE_RATIO = 0.60
FIGURE_PAGE_SIZE_RATIO = 0.50
PARAGRAPH_BREAK_RATIO = 0.90
MAX_SECTION_JUMP = 3

MIN_PARAGRAPHS = 5
MIN_BODY_CHARS = 2000
MAX_EMPTY_PAGE_RATIO = 0.40

SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+)*$")
CAPTION_START = re.compile(r"^(Figure|Table)\s+\d+\s*[:.]")
ARXIV_STAMP = re.compile(r"arXiv:(\d{4}\.\d{4,5})(v\d+)?\s*\[([^\]]+)\]")
REFERENCE_START = re.compile(r"\[(\d+)\]\s*")
BULLET_START = re.compile(r"^[\u2022\u00b7\-\*]\s")
SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]?$")
HEADING_TITLE = re.compile(r"^[A-Z(\u00c0-\u00dd]")
WORD = re.compile(r"[A-Za-z]{2,}")
COMPOUND = re.compile(r"[A-Za-z]{2,}-[A-Za-z]{2,}")

UNNUMBERED_HEADINGS = {
    "abstract",
    "acknowledgement",
    "acknowledgements",
    "acknowledgments",
    "references",
    "bibliography",
    "appendix",
    "introduction",
    "conclusion",
    "conclusions",
    "related work",
    "discussion",
    "limitations",
}

BOILERPLATE_PATTERNS = [
    re.compile(r"hereby grants permission", re.IGNORECASE),
    re.compile(r"reproduce the tables and figures", re.IGNORECASE),
    re.compile(r"^scholarly works\.?$", re.IGNORECASE),
    re.compile(r"Conference on Neural Information Processing Systems", re.IGNORECASE),
    re.compile(r"^arXiv:\d{4}\.\d{4,5}", re.IGNORECASE),
    re.compile(r"Preprint\.?\s*Under review", re.IGNORECASE),
    re.compile(r"^Published as a conference paper", re.IGNORECASE),
]


console = Console()


@dataclass
class Item:
    kind: str
    text: str
    page: int
    number: str = ""


@dataclass
class Paragraph:
    text: str
    page: int


@dataclass
class Section:
    heading: str
    start_page: int
    number: str = ""
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class PaperStats:
    arxiv_id: str = ""
    title: str = ""
    pages: int = 0
    empty_pages: int = 0
    figure_pages: int = 0
    hyphen_joins: int = 0
    hyphens_kept: int = 0
    rejected_headings: int = 0
    sections: int = 0
    paragraphs: int = 0
    references: int = 0
    body_chars: int = 0
    quality: str = "ok"
    issues: list[str] = field(default_factory=list)


def extract_pages(pdf_path: Path) -> dict[int, list[str]]:
    document = pymupdf.open(pdf_path)
    pages: dict[int, list[str]] = {}

    for page_number, page in enumerate(document, start=1):
        lines = [line.strip() for line in page.get_text().splitlines()]
        pages[page_number] = [line for line in lines if line]

    document.close()
    return pages


def raw_text_of(pages: dict[int, list[str]]) -> str:
    parts = []
    for page_number in sorted(pages):
        parts.append(f"=== PAGE {page_number} ===")
        parts.extend(pages[page_number])
    return "\n".join(parts)


def build_vocabulary(pages: dict[int, list[str]]) -> tuple[set[str], set[str]]:
    vocabulary: set[str] = set()
    compounds: set[str] = set()

    for lines in pages.values():
        previous_line_hyphenated = False

        for line in lines:
            for compound in COMPOUND.findall(line):
                compounds.add(compound.lower())

            words = WORD.findall(line)
            start = 1 if previous_line_hyphenated else 0
            end = len(words) - 1 if line.endswith("-") else len(words)

            for word in words[start:end]:
                vocabulary.add(word.lower())

            previous_line_hyphenated = line.endswith("-")

    return vocabulary, compounds


def remove_page_number(lines: list[str], page_number: int) -> list[str]:
    if lines and lines[-1] == str(page_number):
        return lines[:-1]
    return lines


def remove_boilerplate(lines: list[str]) -> list[str]:
    return [
        line
        for line in lines
        if not any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS)
    ]


def is_figure_page(lines: list[str], typical_page_size: float) -> bool:
    if not lines:
        return False

    short_lines = sum(1 for line in lines if len(line.split()) <= SHORT_LINE_WORD_LIMIT)
    short_ratio = short_lines / len(lines)
    page_size = sum(len(line) for line in lines)

    return (
        short_ratio >= FIGURE_PAGE_SHORT_LINE_RATIO
        and page_size < typical_page_size * FIGURE_PAGE_SIZE_RATIO
    )


def keep_captions_only(lines: list[str]) -> list[str]:
    kept: list[str] = []
    in_caption = False

    for line in lines:
        if CAPTION_START.match(line):
            in_caption = True
            kept.append(line)
            continue
        if in_caption and len(line.split()) > SHORT_LINE_WORD_LIMIT:
            kept.append(line)
            continue
        in_caption = False

    return kept


def is_plausible_next_section(candidate: str, seen: list[str]) -> bool:
    parts = [int(piece) for piece in candidate.split(".")]

    if not seen:
        return parts == [1]

    previous = [int(piece) for piece in seen[-1].split(".")]

    if len(parts) == len(previous) + 1 and parts[:-1] == previous:
        return parts[-1] == 1

    if len(parts) == len(previous):
        same_prefix = parts[:-1] == previous[:-1]
        step = parts[-1] - previous[-1]
        return same_prefix and 1 <= step <= MAX_SECTION_JUMP

    if len(parts) < len(previous):
        same_prefix = parts[:-1] == previous[:len(parts) - 1]
        step = parts[-1] - previous[len(parts) - 1]
        return same_prefix and 1 <= step <= MAX_SECTION_JUMP

    return False


def looks_like_heading_title(text: str) -> bool:
    if not HEADING_TITLE.match(text):
        return False
    if not WORD.search(text):
        return False
    return len(text.split()) <= 10 and not SENTENCE_END.search(text)


def merge_headings(
    lines: list[str],
    page_number: int,
    seen_numbers: list[str],
    stats: PaperStats,
) -> list[Item]:
    items: list[Item] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if SECTION_NUMBER.match(line) and index + 1 < len(lines):
            title = lines[index + 1]
            if (
                not SECTION_NUMBER.match(title)
                and looks_like_heading_title(title)
                and is_plausible_next_section(line, seen_numbers)
            ):
                seen_numbers.append(line)
                items.append(Item("heading", f"{line} {title}", page_number, line))
                index += 2
                continue
            if not SECTION_NUMBER.match(title):
                stats.rejected_headings += 1

        if line.lower().rstrip(".").strip() in UNNUMBERED_HEADINGS:
            items.append(Item("heading", line, page_number))
            index += 1
            continue

        items.append(Item("line", line, page_number))
        index += 1

    return items


def join_lines(
    first: str,
    second: str,
    vocabulary: set[str],
    compounds: set[str],
    stats: PaperStats,
) -> str:
    if not (
        first.endswith("-")
        and len(first) > 1
        and first[-2].isalpha()
        and second[:1].islower()
    ):
        return f"{first} {second}"

    left_match = WORD.search(first[:-1][::-1])
    left = left_match.group()[::-1].lower() if left_match else ""
    right_match = WORD.match(second)
    right = right_match.group().lower() if right_match else ""

    if left and right:
        if f"{left}-{right}" in compounds:
            stats.hyphens_kept += 1
            return first + second

        if f"{left}{right}" in vocabulary:
            stats.hyphen_joins += 1
            return first[:-1] + second

        if left in vocabulary and right in vocabulary:
            stats.hyphens_kept += 1
            return first + second

    stats.hyphen_joins += 1
    return first[:-1] + second


def build_paragraphs(
    lines: list[str],
    page_number: int,
    vocabulary: set[str],
    compounds: set[str],
    stats: PaperStats,
) -> list[Paragraph]:
    if not lines:
        return []

    typical_width = median(len(line) for line in lines)
    break_width = typical_width * PARAGRAPH_BREAK_RATIO

    paragraphs: list[Paragraph] = []
    buffer = ""

    for line in lines:
        if buffer and BULLET_START.match(line):
            paragraphs.append(Paragraph(buffer, page_number))
            buffer = line
        elif buffer:
            buffer = join_lines(buffer, line, vocabulary, compounds, stats)
        else:
            buffer = line

        if len(line) < break_width and SENTENCE_END.search(line):
            paragraphs.append(Paragraph(buffer, page_number))
            buffer = ""

    if buffer:
        paragraphs.append(Paragraph(buffer, page_number))

    return paragraphs


def extract_arxiv_stamp(raw_text: str) -> dict[str, str] | None:
    match = ARXIV_STAMP.search(raw_text)
    if not match:
        return None
    return {
        "arxiv_id": match.group(1),
        "version": match.group(2) or "",
        "category": match.group(3),
    }


def parse_references(lines: list[str]) -> list[str]:
    joined = " ".join(lines)
    pieces = REFERENCE_START.split(joined)

    references: list[str] = []
    for position in range(1, len(pieces) - 1, 2):
        number = pieces[position]
        body = " ".join(pieces[position + 1].split())
        if body:
            references.append(f"[{number}] {body}")

    return references


def prune_empty_sections(sections: list[Section]) -> list[Section]:
    kept: list[Section] = []

    for position, section in enumerate(sections):
        if section.paragraphs:
            kept.append(section)
            continue

        if not section.number:
            continue

        prefix = section.number + "."
        has_child = any(
            later.number.startswith(prefix) for later in sections[position + 1:]
        )
        if has_child:
            kept.append(section)

    return kept


def assess_quality(stats: PaperStats) -> None:
    issues = []

    if stats.pages and stats.empty_pages / stats.pages > MAX_EMPTY_PAGE_RATIO:
        issues.append("many pages produced no text, likely scanned")
    if stats.paragraphs < MIN_PARAGRAPHS:
        issues.append("very few paragraphs recovered")
    if stats.body_chars < MIN_BODY_CHARS:
        issues.append("body text is unusually short")
    if stats.sections <= 1:
        issues.append("no section structure detected")
    if stats.references == 0:
        issues.append("no references parsed")

    stats.issues = issues
    if not issues:
        stats.quality = "ok"
    elif len(issues) >= 3:
        stats.quality = "poor"
    else:
        stats.quality = "partial"


def ingest_paper(pdf_path: Path, metadata: dict) -> tuple[dict | None, PaperStats]:
    stats = PaperStats(
        arxiv_id=metadata.get("arxiv_id", pdf_path.stem),
        title=metadata.get("title", ""),
    )

    try:
        pages = extract_pages(pdf_path)
    except Exception as error:
        stats.quality = "poor"
        stats.issues = [f"could not open PDF: {str(error)[:60]}"]
        return None, stats

    stats.pages = len(pages)
    stats.empty_pages = sum(1 for lines in pages.values() if not lines)

    if not pages:
        stats.quality = "poor"
        stats.issues = ["PDF contained no pages"]
        return None, stats

    raw_text = raw_text_of(pages)
    stamp = extract_arxiv_stamp(raw_text)
    vocabulary, compounds = build_vocabulary(pages)

    page_sizes = [sum(len(line) for line in lines) for lines in pages.values()]
    typical_page_size = median(page_sizes) if page_sizes else 0.0

    items: list[Item] = []
    seen_numbers: list[str] = []

    for page_number in sorted(pages):
        lines = remove_page_number(pages[page_number], page_number)

        if page_number == 1:
            lines = remove_boilerplate(lines)

        if is_figure_page(lines, typical_page_size):
            stats.figure_pages += 1
            lines = keep_captions_only(lines)

        items.extend(merge_headings(lines, page_number, seen_numbers, stats))

    reference_index = next(
        (
            position
            for position, item in enumerate(items)
            if item.kind == "heading"
            and item.text.lower().rstrip(".") in {"references", "bibliography"}
        ),
        len(items),
    )

    body_items = items[:reference_index]
    reference_items = items[reference_index + 1:]

    references = parse_references(
        [
            item.text
            for item in reference_items
            if item.kind == "line" and not CAPTION_START.match(item.text)
        ]
    )

    sections: list[Section] = []
    current = Section(heading="(front matter)", start_page=1)
    pending_lines: list[str] = []
    pending_page = 1

    def flush() -> None:
        nonlocal pending_lines
        if pending_lines:
            current.paragraphs.extend(
                build_paragraphs(
                    pending_lines, pending_page, vocabulary, compounds, stats
                )
            )
            pending_lines = []

    for item in body_items:
        if item.kind == "heading":
            flush()
            sections.append(current)
            current = Section(
                heading=item.text, start_page=item.page, number=item.number
            )
            pending_page = item.page
            continue

        if item.page != pending_page:
            flush()
            pending_page = item.page

        pending_lines.append(item.text)

    flush()
    sections.append(current)
    sections = prune_empty_sections(sections)

    stats.sections = len(sections)
    stats.paragraphs = sum(len(section.paragraphs) for section in sections)
    stats.references = len(references)
    stats.body_chars = sum(
        len(paragraph.text)
        for section in sections
        for paragraph in section.paragraphs
    )

    assess_quality(stats)

    document = {
        "paper_id": metadata.get("arxiv_id", pdf_path.stem),
        "title": metadata.get("title", ""),
        "authors": metadata.get("authors", []),
        "published": metadata.get("published", ""),
        "categories": metadata.get("categories", []),
        "abstract": metadata.get("summary", ""),
        "arxiv_stamp": stamp,
        "source_file": pdf_path.name,
        "page_count": stats.pages,
        "quality": stats.quality,
        "sections": [
            {
                "heading": section.heading,
                "number": section.number,
                "start_page": section.start_page,
                "paragraphs": [
                    {"text": paragraph.text, "page": paragraph.page}
                    for paragraph in section.paragraphs
                ],
            }
            for section in sections
        ],
        "references": references,
    }

    return document, stats


def main() -> None:
    if not CATALOG_PATH.exists():
        console.print(f"[bold red]Catalog not found:[/] {CATALOG_PATH.resolve()}")
        console.print("Run fetch_papers.py first.")
        return

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    papers = catalog.get("papers", [])
    console.print(f"[bold cyan]Papers in catalog:[/] {len(papers)}")

    documents = []
    all_stats: list[PaperStats] = []

    for position, metadata in enumerate(papers, start=1):
        pdf_path = PAPERS_DIR / metadata["filename"]
        if not pdf_path.exists():
            console.print(f"  [{position}/{len(papers)}] missing file: {pdf_path.name}")
            continue

        document, stats = ingest_paper(pdf_path, metadata)
        all_stats.append(stats)

        marker = {"ok": "ok", "partial": "partial", "poor": "poor"}[stats.quality]
        console.print(
            f"  [{position}/{len(papers)}] {stats.arxiv_id} {marker} "
            f"({stats.sections} sections, {stats.paragraphs} paragraphs)"
        )

        if document and stats.quality != "poor":
            documents.append(document)

    if not documents:
        console.print("[bold red]No documents were ingested successfully.[/]")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    corpus = {
        "document_count": len(documents),
        "documents": documents,
    }
    CORPUS_PATH.write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = {
        "papers_attempted": len(all_stats),
        "papers_ingested": len(documents),
        "papers": [
            {
                "arxiv_id": stats.arxiv_id,
                "title": stats.title,
                "quality": stats.quality,
                "pages": stats.pages,
                "empty_pages": stats.empty_pages,
                "figure_pages": stats.figure_pages,
                "sections": stats.sections,
                "paragraphs": stats.paragraphs,
                "references": stats.references,
                "body_chars": stats.body_chars,
                "hyphen_joins": stats.hyphen_joins,
                "hyphens_kept": stats.hyphens_kept,
                "rejected_headings": stats.rejected_headings,
                "issues": stats.issues,
            }
            for stats in all_stats
        ],
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = Table(title="Ingestion summary")
    summary.add_column("Quality")
    summary.add_column("Papers", justify="right")
    summary.add_column("Sections", justify="right")
    summary.add_column("Paragraphs", justify="right")

    for quality in ("ok", "partial", "poor"):
        rows = [stats for stats in all_stats if stats.quality == quality]
        if not rows:
            continue
        summary.add_row(
            quality,
            str(len(rows)),
            str(sum(stats.sections for stats in rows)),
            str(sum(stats.paragraphs for stats in rows)),
        )

    console.print()
    console.print(summary)

    problems = [stats for stats in all_stats if stats.issues]
    if problems:
        issue_table = Table(title="Papers with issues")
        issue_table.add_column("arXiv ID")
        issue_table.add_column("Quality")
        issue_table.add_column("Pages", justify="right")
        issue_table.add_column("Paras", justify="right")
        issue_table.add_column("Issues")

        for stats in problems:
            issue_table.add_row(
                stats.arxiv_id,
                stats.quality,
                str(stats.pages),
                str(stats.paragraphs),
                "; ".join(stats.issues),
            )

        console.print()
        console.print(issue_table)

    total_paragraphs = sum(stats.paragraphs for stats in all_stats)
    total_chars = sum(stats.body_chars for stats in all_stats)

    console.print()
    console.print(f"[bold cyan]Corpus:[/] {len(documents)} documents, "
                  f"{total_paragraphs:,} paragraphs, {total_chars:,} characters")
    console.print(f"[bold green]Saved:[/] {CORPUS_PATH}")
    console.print(f"[bold green]Saved:[/] {REPORT_PATH}")


if __name__ == "__main__":
    main()