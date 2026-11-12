from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from rich.console import Console
from rich.table import Table


INPUT_TEXT = Path("output/attention.txt")
OUTPUT_DIR = Path("output")

SHORT_LINE_WORD_LIMIT = 3
FIGURE_PAGE_SHORT_LINE_RATIO = 0.60
FIGURE_PAGE_SIZE_RATIO = 0.50
PARAGRAPH_BREAK_RATIO = 0.90
MAX_SECTION_JUMP = 3

PAGE_MARKER = re.compile(r"^=== PAGE (\d+) ===$")
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
    "appendix",
    "attention visualizations",
}

BOILERPLATE_PATTERNS = [
    re.compile(r"hereby grants permission", re.IGNORECASE),
    re.compile(r"reproduce the tables and figures", re.IGNORECASE),
    re.compile(r"^scholarly works\.?$", re.IGNORECASE),
    re.compile(r"Conference on Neural Information Processing Systems", re.IGNORECASE),
    re.compile(r"^arXiv:\d{4}\.\d{4,5}", re.IGNORECASE),
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
class Stats:
    raw_characters: int = 0
    clean_characters: int = 0
    hyphen_joins: int = 0
    hyphens_kept: int = 0
    page_numbers_removed: int = 0
    boilerplate_lines_removed: int = 0
    figure_pages: list[int] = field(default_factory=list)
    figure_lines_dropped: int = 0
    rejected_headings: int = 0


def split_pages(raw_text: str) -> dict[int, list[str]]:
    pages: dict[int, list[str]] = {}
    current_page = 0

    for line in raw_text.splitlines():
        stripped = line.strip()
        marker = PAGE_MARKER.match(stripped)
        if marker:
            current_page = int(marker.group(1))
            pages[current_page] = []
            continue
        if current_page and stripped:
            pages[current_page].append(stripped)

    return pages


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


def remove_page_number(lines: list[str], page_number: int, stats: Stats) -> list[str]:
    if lines and lines[-1] == str(page_number):
        stats.page_numbers_removed += 1
        return lines[:-1]
    return lines


def remove_boilerplate(lines: list[str], stats: Stats) -> list[str]:
    kept = []
    for line in lines:
        if any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS):
            stats.boilerplate_lines_removed += 1
            continue
        kept.append(line)
    return kept


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


def keep_captions_only(lines: list[str], stats: Stats) -> list[str]:
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

    stats.figure_lines_dropped += len(lines) - len(kept)
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
    stats: Stats,
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

        if line.lower().rstrip(".") in UNNUMBERED_HEADINGS:
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
    stats: Stats,
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
    stats: Stats,
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


def extract_arxiv_id(raw_text: str) -> dict[str, str] | None:
    match = ARXIV_STAMP.search(raw_text)
    if not match:
        return None
    return {
        "arxiv_id": match.group(1),
        "version": match.group(2) or "",
        "category": match.group(3),
    }


def extract_title(first_page_lines: list[str]) -> str:
    for line in first_page_lines:
        if "@" in line or SECTION_NUMBER.match(line):
            continue
        if len(line) < 8:
            continue
        return line
    return ""


def parse_references(lines: list[str]) -> list[str]:
    joined = " ".join(lines)
    pieces = REFERENCE_START.split(joined)

    references: list[str] = []
    for position in range(1, len(pieces) - 1, 2):
        number = pieces[position]
        body = " ".join(pieces[position + 1].split())
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


def main() -> None:
    if not INPUT_TEXT.exists():
        console.print(f"[bold red]File not found:[/] {INPUT_TEXT.resolve()}")
        console.print("Run extract_text.py first.")
        return

    raw_text = INPUT_TEXT.read_text(encoding="utf-8")
    stats = Stats(raw_characters=len(raw_text))

    arxiv_info = extract_arxiv_id(raw_text)
    pages = split_pages(raw_text)
    console.print(f"[bold cyan]Pages read:[/] {len(pages)}")

    vocabulary, compounds = build_vocabulary(pages)
    console.print(
        f"[bold cyan]Vocabulary:[/] {len(vocabulary):,} words, "
        f"{len(compounds):,} hyphenated compounds"
    )

    page_sizes = [sum(len(line) for line in lines) for lines in pages.values()]
    typical_page_size = median(page_sizes) if page_sizes else 0.0

    items: list[Item] = []
    first_page_lines: list[str] = []
    seen_numbers: list[str] = []

    for page_number in sorted(pages):
        lines = pages[page_number]
        lines = remove_page_number(lines, page_number, stats)

        if page_number == 1:
            lines = remove_boilerplate(lines, stats)
            first_page_lines = list(lines)

        if is_figure_page(lines, typical_page_size):
            stats.figure_pages.append(page_number)
            lines = keep_captions_only(lines, stats)

        items.extend(merge_headings(lines, page_number, seen_numbers, stats))

    title = extract_title(first_page_lines)

    reference_index = next(
        (
            position
            for position, item in enumerate(items)
            if item.kind == "heading" and item.text.lower().rstrip(".") == "references"
        ),
        len(items),
    )

    body_items = items[:reference_index]
    reference_items = items[reference_index + 1:]

    reference_lines = [
        item.text
        for item in reference_items
        if item.kind == "line" and not CAPTION_START.match(item.text)
    ]
    references = parse_references(reference_lines)

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

    document = {
        "source_file": INPUT_TEXT.name,
        "title": title,
        "arxiv": arxiv_info,
        "page_count": len(pages),
        "figure_pages": stats.figure_pages,
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

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = INPUT_TEXT.stem
    json_path = OUTPUT_DIR / f"{stem}.json"
    json_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    readable_parts: list[str] = [f"TITLE: {title}", ""]
    for section in sections:
        readable_parts.append(f"\n## {section.heading}\n")
        for paragraph in section.paragraphs:
            readable_parts.append(f"[p{paragraph.page}] {paragraph.text}\n")

    clean_text = "\n".join(readable_parts)
    stats.clean_characters = len(clean_text)

    clean_path = OUTPUT_DIR / f"{stem}.clean.txt"
    clean_path.write_text(clean_text, encoding="utf-8")

    paragraph_count = sum(len(section.paragraphs) for section in sections)

    report = Table(title="Cleaning report")
    report.add_column("Measure")
    report.add_column("Value", justify="right")
    report.add_row("Title detected", title or "(none)")
    report.add_row("arXiv ID", arxiv_info["arxiv_id"] if arxiv_info else "(none)")
    report.add_row("Raw characters", f"{stats.raw_characters:,}")
    report.add_row("Clean characters", f"{stats.clean_characters:,}")
    report.add_row("Broken words rejoined", str(stats.hyphen_joins))
    report.add_row("Real hyphens preserved", str(stats.hyphens_kept))
    report.add_row("False headings rejected", str(stats.rejected_headings))
    report.add_row("Page numbers removed", str(stats.page_numbers_removed))
    report.add_row("Boilerplate lines removed", str(stats.boilerplate_lines_removed))
    report.add_row("Figure pages detected", str(stats.figure_pages))
    report.add_row("Figure lines dropped", str(stats.figure_lines_dropped))
    report.add_row("Sections found", str(len(sections)))
    report.add_row("Paragraphs built", str(paragraph_count))
    report.add_row("References parsed", str(len(references)))

    console.print()
    console.print(report)
    console.print()
    console.print(f"[bold green]Saved:[/] {json_path}")
    console.print(f"[bold green]Saved:[/] {clean_path}")

    console.print()
    console.print("[bold cyan]Section headings found:[/]")
    for section in sections:
        count = len(section.paragraphs)
        console.print(f"  p{section.start_page:>3}  {section.heading}  ({count} paragraphs)")


if __name__ == "__main__":
    main()