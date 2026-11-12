from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median

from rich.console import Console
from rich.table import Table


INPUT_JSON = Path("output/attention.json")
OUTPUT_DIR = Path("output")

TARGET_CHUNK_SIZE = 1200
MIN_CHUNK_SIZE = 200
OVERLAP_SENTENCES = 1
FIXED_CHUNK_SIZE = 1000
FIXED_CHUNK_OVERLAP = 150

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\u00c0-\u00dd])")
WORD = re.compile(r"[A-Za-z0-9]+")

STRATEGIES = ["fixed", "paragraph", "section", "structured"]


console = Console()


@dataclass
class Chunk:
    chunk_id: str
    text: str
    strategy: str
    section: str
    section_number: str
    pages: list[int]
    char_count: int
    word_count: int


def load_document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def split_sentences(text: str) -> list[str]:
    return [piece.strip() for piece in SENTENCE_SPLIT.split(text) if piece.strip()]


def make_chunk(
    index: int,
    text: str,
    strategy: str,
    section: str,
    section_number: str,
    pages: list[int],
) -> Chunk:
    return Chunk(
        chunk_id=f"{strategy}-{index:04d}",
        text=text,
        strategy=strategy,
        section=section,
        section_number=section_number,
        pages=sorted(set(pages)),
        char_count=len(text),
        word_count=len(WORD.findall(text)),
    )


def chunk_fixed(document: dict) -> list[Chunk]:
    pieces: list[tuple[str, str, str, int]] = []
    for section in document["sections"]:
        for paragraph in section["paragraphs"]:
            pieces.append(
                (
                    paragraph["text"],
                    section["heading"],
                    section.get("number", ""),
                    paragraph["page"],
                )
            )

    full_text = " ".join(piece[0] for piece in pieces)

    offsets: list[tuple[int, int, str, str, int]] = []
    cursor = 0
    for text, heading, number, page in pieces:
        offsets.append((cursor, cursor + len(text), heading, number, page))
        cursor += len(text) + 1

    def locate(start: int, end: int) -> tuple[str, str, list[int]]:
        heading = ""
        number = ""
        pages: list[int] = []
        for piece_start, piece_end, piece_heading, piece_number, page in offsets:
            if piece_start < end and piece_end > start:
                pages.append(page)
                if not heading:
                    heading = piece_heading
                    number = piece_number
        return heading, number, pages

    chunks: list[Chunk] = []
    position = 0
    index = 0

    while position < len(full_text):
        end = position + FIXED_CHUNK_SIZE
        text = full_text[position:end].strip()
        if text:
            heading, number, pages = locate(position, end)
            chunks.append(make_chunk(index, text, "fixed", heading, number, pages))
            index += 1
        position += FIXED_CHUNK_SIZE - FIXED_CHUNK_OVERLAP

    return chunks


def chunk_paragraph(document: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0

    for section in document["sections"]:
        for paragraph in section["paragraphs"]:
            text = paragraph["text"].strip()
            if len(text) < MIN_CHUNK_SIZE:
                continue
            chunks.append(
                make_chunk(
                    index,
                    text,
                    "paragraph",
                    section["heading"],
                    section.get("number", ""),
                    [paragraph["page"]],
                )
            )
            index += 1

    return chunks


def chunk_section(document: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0

    for section in document["sections"]:
        if not section["paragraphs"]:
            continue
        text = " ".join(paragraph["text"] for paragraph in section["paragraphs"])
        pages = [paragraph["page"] for paragraph in section["paragraphs"]]
        chunks.append(
            make_chunk(
                index,
                f"{section['heading']}\n\n{text}",
                "section",
                section["heading"],
                section.get("number", ""),
                pages,
            )
        )
        index += 1

    return chunks


def chunk_structured(document: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0

    for section in document["sections"]:
        heading = section["heading"]
        number = section.get("number", "")
        buffer: list[str] = []
        buffer_pages: list[int] = []
        buffer_size = 0

        def flush() -> None:
            nonlocal index, buffer, buffer_pages, buffer_size
            if not buffer:
                return
            body = " ".join(buffer)
            chunks.append(
                make_chunk(
                    index,
                    f"{heading}\n\n{body}",
                    "structured",
                    heading,
                    number,
                    list(buffer_pages),
                )
            )
            index += 1
            tail = split_sentences(body)[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
            buffer = list(tail)
            buffer_pages = buffer_pages[-1:] if buffer_pages else []
            buffer_size = sum(len(piece) for piece in buffer)

        for paragraph in section["paragraphs"]:
            text = paragraph["text"].strip()
            if not text:
                continue

            if len(text) > TARGET_CHUNK_SIZE:
                flush()
                sentences = split_sentences(text)
                current: list[str] = []
                current_size = 0
                for sentence in sentences:
                    if current and current_size + len(sentence) > TARGET_CHUNK_SIZE:
                        chunks.append(
                            make_chunk(
                                index,
                                f"{heading}\n\n{' '.join(current)}",
                                "structured",
                                heading,
                                number,
                                [paragraph["page"]],
                            )
                        )
                        index += 1
                        current = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
                        current_size = sum(len(piece) for piece in current)
                    current.append(sentence)
                    current_size += len(sentence) + 1
                if current:
                    buffer = list(current)
                    buffer_pages = [paragraph["page"]]
                    buffer_size = current_size
                continue

            if buffer_size + len(text) > TARGET_CHUNK_SIZE:
                flush()

            buffer.append(text)
            buffer_pages.append(paragraph["page"])
            buffer_size += len(text) + 1

        if buffer_size >= MIN_CHUNK_SIZE:
            flush()
        elif buffer and chunks and chunks[-1].section == heading:
            merged = chunks[-1].text + " " + " ".join(buffer)
            chunks[-1] = make_chunk(
                int(chunks[-1].chunk_id.split("-")[1]),
                merged,
                "structured",
                heading,
                number,
                chunks[-1].pages + buffer_pages,
            )
            buffer = []
            buffer_pages = []
            buffer_size = 0
        elif buffer:
            flush()

    return chunks


def summarise(strategy: str, chunks: list[Chunk]) -> dict:
    if not chunks:
        return {
            "strategy": strategy,
            "count": 0,
            "mean_chars": 0,
            "median_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "undersized": 0,
            "oversized": 0,
        }

    sizes = [chunk.char_count for chunk in chunks]
    return {
        "strategy": strategy,
        "count": len(chunks),
        "mean_chars": round(mean(sizes)),
        "median_chars": round(median(sizes)),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "undersized": sum(1 for size in sizes if size < MIN_CHUNK_SIZE),
        "oversized": sum(1 for size in sizes if size > TARGET_CHUNK_SIZE * 1.5),
    }


def main() -> None:
    if not INPUT_JSON.exists():
        console.print(f"[bold red]File not found:[/] {INPUT_JSON.resolve()}")
        console.print("Run clean_text.py first.")
        return

    document = load_document(INPUT_JSON)
    console.print(f"[bold cyan]Document:[/] {document['title']}")
    console.print(f"[bold cyan]Sections:[/] {len(document['sections'])}")

    builders = {
        "fixed": chunk_fixed,
        "paragraph": chunk_paragraph,
        "section": chunk_section,
        "structured": chunk_structured,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = INPUT_JSON.stem
    summaries = []
    all_chunks: dict[str, list[Chunk]] = {}

    for strategy in STRATEGIES:
        chunks = builders[strategy](document)
        all_chunks[strategy] = chunks
        summaries.append(summarise(strategy, chunks))

        payload = {
            "source_document": document["title"],
            "arxiv": document.get("arxiv"),
            "strategy": strategy,
            "target_chunk_size": TARGET_CHUNK_SIZE,
            "chunk_count": len(chunks),
            "chunks": [asdict(chunk) for chunk in chunks],
        }
        path = OUTPUT_DIR / f"{stem}.chunks.{strategy}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        console.print(f"[bold green]Saved:[/] {path}")

    comparison = Table(title="Chunking strategy comparison")
    comparison.add_column("Strategy")
    comparison.add_column("Chunks", justify="right")
    comparison.add_column("Mean", justify="right")
    comparison.add_column("Median", justify="right")
    comparison.add_column("Min", justify="right")
    comparison.add_column("Max", justify="right")
    comparison.add_column("Undersized", justify="right")
    comparison.add_column("Oversized", justify="right")

    for summary in summaries:
        comparison.add_row(
            summary["strategy"],
            str(summary["count"]),
            str(summary["mean_chars"]),
            str(summary["median_chars"]),
            str(summary["min_chars"]),
            str(summary["max_chars"]),
            str(summary["undersized"]),
            str(summary["oversized"]),
        )

    console.print()
    console.print(comparison)

    report_path = OUTPUT_DIR / f"{stem}.chunking_report.json"
    report_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"[bold green]Saved:[/] {report_path}")

    sample = all_chunks["structured"]
    if sample:
        preview = sample[len(sample) // 2]
        console.print()
        console.print("[bold cyan]Sample structured chunk:[/]")
        console.print(f"  id: {preview.chunk_id}")
        console.print(f"  section: {preview.section}")
        console.print(f"  pages: {preview.pages}")
        console.print(f"  chars: {preview.char_count}, words: {preview.word_count}")
        console.print("-" * 60)
        console.print(preview.text[:700])
        console.print("-" * 60)


if __name__ == "__main__":
    main()