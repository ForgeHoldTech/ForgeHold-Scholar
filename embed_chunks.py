from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer


INPUT_DIR = Path("output")
OUTPUT_DIR = Path("output")
DOCUMENT_STEM = "attention"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 16
MIN_CHUNK_CHARS = 150

STRATEGIES = ["fixed", "paragraph", "section", "structured"]


console = Console()


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_chunks(strategy: str) -> dict | None:
    path = INPUT_DIR / f"{DOCUMENT_STEM}.chunks.{strategy}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def filter_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    kept = [chunk for chunk in chunks if chunk["char_count"] >= MIN_CHUNK_CHARS]
    return kept, len(chunks) - len(kept)


def embed_strategy(
    model: SentenceTransformer,
    strategy: str,
    payload: dict,
) -> dict | None:
    chunks, dropped = filter_chunks(payload["chunks"])
    if not chunks:
        console.print(f"[yellow]No usable chunks for strategy:[/] {strategy}")
        return None

    texts = [chunk["text"] for chunk in chunks]

    started = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - started

    vectors = vectors.astype(np.float32)
    vector_path = OUTPUT_DIR / f"{DOCUMENT_STEM}.vectors.{strategy}.npy"
    np.save(vector_path, vectors)

    metadata = {
        "source_document": payload["source_document"],
        "arxiv": payload.get("arxiv"),
        "strategy": strategy,
        "model": MODEL_NAME,
        "dimensions": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "dropped_chunks": dropped,
        "vector_file": vector_path.name,
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "section": chunk["section"],
                "section_number": chunk["section_number"],
                "pages": chunk["pages"],
                "char_count": chunk["char_count"],
                "word_count": chunk["word_count"],
            }
            for chunk in chunks
        ],
    }
    metadata_path = OUTPUT_DIR / f"{DOCUMENT_STEM}.index.{strategy}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "strategy": strategy,
        "embedded": len(chunks),
        "dropped": dropped,
        "dimensions": int(vectors.shape[1]),
        "seconds": round(elapsed, 2),
        "per_second": round(len(chunks) / elapsed, 1) if elapsed else 0.0,
    }


def verify_similarity(strategy: str) -> None:
    vector_path = OUTPUT_DIR / f"{DOCUMENT_STEM}.vectors.{strategy}.npy"
    metadata_path = OUTPUT_DIR / f"{DOCUMENT_STEM}.index.{strategy}.json"
    if not vector_path.exists() or not metadata_path.exists():
        return

    vectors = np.load(vector_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = metadata["chunks"]

    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -1.0)

    best = int(np.argmax(similarity))
    row, column = divmod(best, similarity.shape[1])

    console.print()
    console.print("[bold cyan]Closest pair of chunks:[/]")
    console.print(f"  score: {similarity[row][column]:.4f}")
    console.print(f"  A ({chunks[row]['chunk_id']}, {chunks[row]['section']})")
    console.print(f"     {chunks[row]['text'][:160]}...")
    console.print(f"  B ({chunks[column]['chunk_id']}, {chunks[column]['section']})")
    console.print(f"     {chunks[column]['text'][:160]}...")


def main() -> None:
    device = select_device()
    console.print(f"[bold cyan]Device:[/] {device}")
    if device == "cuda":
        console.print(f"[bold cyan]GPU:[/] {torch.cuda.get_device_name(0)}")

    console.print(f"[bold cyan]Loading model:[/] {MODEL_NAME}")
    console.print("First run downloads the model, which may take a few minutes.")

    model = SentenceTransformer(MODEL_NAME, device=device)
    console.print("[bold green]Model ready[/]")

    OUTPUT_DIR.mkdir(exist_ok=True)
    results = []

    for strategy in STRATEGIES:
        payload = load_chunks(strategy)
        if payload is None:
            console.print(f"[yellow]Missing chunk file for strategy:[/] {strategy}")
            continue

        console.print(f"Embedding strategy: {strategy}")
        result = embed_strategy(model, strategy, payload)
        if result:
            results.append(result)

    if not results:
        console.print("[bold red]Nothing was embedded. Run chunk_text.py first.[/]")
        return

    report = Table(title="Embedding report")
    report.add_column("Strategy")
    report.add_column("Embedded", justify="right")
    report.add_column("Dropped", justify="right")
    report.add_column("Dimensions", justify="right")
    report.add_column("Seconds", justify="right")
    report.add_column("Chunks/sec", justify="right")

    for result in results:
        report.add_row(
            result["strategy"],
            str(result["embedded"]),
            str(result["dropped"]),
            str(result["dimensions"]),
            str(result["seconds"]),
            str(result["per_second"]),
        )

    console.print()
    console.print(report)

    report_path = OUTPUT_DIR / f"{DOCUMENT_STEM}.embedding_report.json"
    report_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"[bold green]Saved:[/] {report_path}")

    verify_similarity("structured")


if __name__ == "__main__":
    main()