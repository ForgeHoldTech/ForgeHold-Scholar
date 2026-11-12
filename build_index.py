from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer


CORPUS_PATH = Path("output/corpus.json")
OUTPUT_DIR = Path("output")
INDEX_PATH = OUTPUT_DIR / "corpus.index.json"
VECTOR_PATH = OUTPUT_DIR / "corpus.vectors.npy"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32

TARGET_CHUNK_SIZE = 1200
MIN_CHUNK_SIZE = 150
OVERLAP_SENTENCES = 1

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\u00c0-\u00dd])")
WORD = re.compile(r"[A-Za-z0-9]+")


console = Console()


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    published: str
    section: str
    section_number: str
    pages: list[int]
    text: str
    char_count: int
    word_count: int


def safe_id(paper_id: str) -> str:
    return paper_id.replace("/", "_")


def split_sentences(text: str) -> list[str]:
    return [piece.strip() for piece in SENTENCE_SPLIT.split(text) if piece.strip()]


def make_chunk(
    document: dict,
    index: int,
    heading: str,
    number: str,
    pages: list[int],
    body: str,
) -> Chunk:
    title = document["title"]
    text = f"{title}\n{heading}\n\n{body}"
    return Chunk(
        chunk_id=f"{safe_id(document['paper_id'])}-{index:04d}",
        paper_id=document["paper_id"],
        paper_title=title,
        published=document.get("published", "")[:10],
        section=heading,
        section_number=number,
        pages=sorted(set(pages)),
        text=text,
        char_count=len(text),
        word_count=len(WORD.findall(text)),
    )


def chunk_document(document: dict) -> list[Chunk]:
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
                make_chunk(document, index, heading, number, buffer_pages, body)
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
                current: list[str] = []
                current_size = 0
                for sentence in split_sentences(text):
                    if current and current_size + len(sentence) > TARGET_CHUNK_SIZE:
                        chunks.append(
                            make_chunk(
                                document,
                                index,
                                heading,
                                number,
                                [paragraph["page"]],
                                " ".join(current),
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
            chunks[-1].text = merged
            chunks[-1].char_count = len(merged)
            chunks[-1].word_count = len(WORD.findall(merged))
            chunks[-1].pages = sorted(set(chunks[-1].pages + buffer_pages))
        elif buffer:
            flush()

    return [chunk for chunk in chunks if chunk.char_count >= MIN_CHUNK_SIZE]


def main() -> None:
    if not CORPUS_PATH.exists():
        console.print(f"[bold red]Corpus not found:[/] {CORPUS_PATH.resolve()}")
        console.print("Run ingest.py first.")
        return

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    documents = corpus["documents"]
    console.print(f"[bold cyan]Documents:[/] {len(documents)}")

    all_chunks: list[Chunk] = []
    per_document: list[tuple[str, int]] = []

    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)
        per_document.append((document["paper_id"], len(chunks)))

    if not all_chunks:
        console.print("[bold red]No chunks were produced.[/]")
        return

    sizes = [chunk.char_count for chunk in all_chunks]
    console.print(f"[bold cyan]Chunks:[/] {len(all_chunks):,}")
    console.print(
        f"[bold cyan]Size:[/] mean {round(mean(sizes))}, "
        f"median {round(median(sizes))}, min {min(sizes)}, max {max(sizes)}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold cyan]Device:[/] {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    started = time.perf_counter()
    vectors = model.encode(
        [chunk.text for chunk in all_chunks],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - started

    OUTPUT_DIR.mkdir(exist_ok=True)
    np.save(VECTOR_PATH, vectors)

    index = {
        "document_count": len(documents),
        "chunk_count": len(all_chunks),
        "model": MODEL_NAME,
        "dimensions": int(vectors.shape[1]),
        "target_chunk_size": TARGET_CHUNK_SIZE,
        "vector_file": VECTOR_PATH.name,
        "papers": [
            {
                "paper_id": document["paper_id"],
                "title": document["title"],
                "published": document.get("published", "")[:10],
                "categories": document.get("categories", []),
            }
            for document in documents
        ],
        "chunks": [asdict(chunk) for chunk in all_chunks],
    }
    INDEX_PATH.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    distribution = Table(title="Chunks per paper")
    distribution.add_column("Paper")
    distribution.add_column("Chunks", justify="right")

    for paper_id, count in sorted(per_document, key=lambda pair: -pair[1])[:10]:
        distribution.add_row(paper_id, str(count))

    counts = [count for _, count in per_document]
    distribution.add_row("---", "---")
    distribution.add_row("mean", str(round(mean(counts))))
    distribution.add_row("min", str(min(counts)))
    distribution.add_row("max", str(max(counts)))

    console.print()
    console.print(distribution)

    console.print()
    console.print(
        f"[bold green]Embedded[/] {len(all_chunks):,} chunks in {elapsed:.1f}s "
        f"({len(all_chunks) / elapsed:.0f} chunks/sec)"
    )
    console.print(f"[bold green]Saved:[/] {INDEX_PATH}")
    console.print(f"[bold green]Saved:[/] {VECTOR_PATH}")


if __name__ == "__main__":
    main()