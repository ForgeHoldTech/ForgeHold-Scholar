"""
ForgeHold Scholar - PDF Text Extraction

This script opens a PDF research paper, extracts the text from every page,
prints a per-page summary, and writes the full text to a .txt file.

Usage:
    python extract_text.py
"""

from pathlib import Path

import pymupdf
from rich.console import Console
from rich.table import Table


# --- Configuration -----------------------------------------------------

INPUT_PDF = Path("data/attention.pdf")
OUTPUT_DIR = Path("output")

# Number of characters to show in the on-screen preview.
PREVIEW_LENGTH = 600


console = Console()


# --- Main routine ------------------------------------------------------

def main() -> None:
    """Extract text from the configured PDF and report what was found."""

    # Step 1: make sure the input file actually exists before doing anything.
    if not INPUT_PDF.exists():
        console.print(f"[bold red]File not found:[/] {INPUT_PDF.resolve()}")
        console.print("Place the PDF at that path and run the script again.")
        return

    # Step 2: open the PDF.
    console.print(f"[bold cyan]Opening:[/] {INPUT_PDF}")
    document = pymupdf.open(INPUT_PDF)

    # Step 3: report basic document information.
    metadata = document.metadata or {}
    console.print(f"[bold]Pages:[/] {document.page_count}")
    console.print(f"[bold]Title:[/] {metadata.get('title') or '(not set)'}")
    console.print(f"[bold]Author:[/] {metadata.get('author') or '(not set)'}")
    console.print()

    # Step 4: walk through every page and collect its text.
    page_texts = []
    summary = Table(title="Per-page extraction summary")
    summary.add_column("Page", justify="right")
    summary.add_column("Characters", justify="right")
    summary.add_column("Words", justify="right")

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()
        page_texts.append(text)
        summary.add_row(
            str(page_number),
            str(len(text)),
            str(len(text.split())),
        )

    document.close()

    console.print(summary)
    console.print()

    # Step 5: join all pages into one string, separated by page markers.
    # The markers make it easy to trace any piece of text back to its page.
    full_text = ""
    for page_number, text in enumerate(page_texts, start=1):
        full_text += f"\n\n=== PAGE {page_number} ===\n\n"
        full_text += text

    # Step 6: write the result to disk.
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{INPUT_PDF.stem}.txt"
    output_path.write_text(full_text, encoding="utf-8")

    total_chars = len(full_text)
    console.print(f"[bold green]Saved:[/] {output_path}  ({total_chars:,} characters)")
    console.print()

    # Step 7: show a short preview so the quality of the extraction is visible.
    console.print("[bold cyan]Preview of extracted text:[/]")
    console.print("-" * 60)
    console.print(page_texts[0][:PREVIEW_LENGTH])
    console.print("-" * 60)


if __name__ == "__main__":
    main()