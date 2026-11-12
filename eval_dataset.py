from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table


OUTPUT_DIR = Path("output")
DATASET_PATH = Path("data/golden_questions.json")

QUESTIONS = [
    {
        "id": "q001",
        "question": "How many attention heads does the base model use?",
        "relevant_sections": ["3.2.2"],
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "q002",
        "question": "What is the dimension of the model embeddings?",
        "relevant_sections": ["3.1", "3.4"],
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "q003",
        "question": "How many identical layers are in the encoder stack?",
        "relevant_sections": ["3.1"],
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "q004",
        "question": "What hardware was used to train the models?",
        "relevant_sections": ["5.2"],
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "q005",
        "question": "How long did it take to train the big model?",
        "relevant_sections": ["5.2", "6.1"],
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "q006",
        "question": "Which optimizer was used and with what beta values?",
        "relevant_sections": ["5.3"],
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "q007",
        "question": "What is the inner layer dimensionality of the feed-forward network?",
        "relevant_sections": ["3.3"],
        "category": "factual",
        "difficulty": "medium",
    },
    {
        "id": "q008",
        "question": "What dropout rate is used for the base model?",
        "relevant_sections": ["5.4"],
        "category": "factual",
        "difficulty": "medium",
    },
    {
        "id": "q009",
        "question": "What BLEU score did the big model achieve on English to German?",
        "relevant_sections": ["6.1", "Abstract"],
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "q010",
        "question": "How many warmup steps were used for the learning rate schedule?",
        "relevant_sections": ["5.3"],
        "category": "factual",
        "difficulty": "medium",
    },
    {
        "id": "q011",
        "question": "What label smoothing value was applied during training?",
        "relevant_sections": ["5.4"],
        "category": "factual",
        "difficulty": "medium",
    },
    {
        "id": "q012",
        "question": "What beam size was used during inference for translation?",
        "relevant_sections": ["6.1"],
        "category": "factual",
        "difficulty": "medium",
    },
    {
        "id": "q013",
        "question": "How large was the English to German training dataset?",
        "relevant_sections": ["5.1"],
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "id": "q014",
        "question": "Why are the dot products scaled before the softmax?",
        "relevant_sections": ["3.2.1"],
        "category": "reasoning",
        "difficulty": "medium",
    },
    {
        "id": "q015",
        "question": "Why is multi-head attention better than a single attention head?",
        "relevant_sections": ["3.2.2"],
        "category": "reasoning",
        "difficulty": "medium",
    },
    {
        "id": "q016",
        "question": "Why did the authors choose sinusoidal functions over learned embeddings?",
        "relevant_sections": ["3.5"],
        "category": "reasoning",
        "difficulty": "medium",
    },
    {
        "id": "q017",
        "question": "Why is masking applied in the decoder self-attention?",
        "relevant_sections": ["3.1", "3.2.3"],
        "category": "reasoning",
        "difficulty": "medium",
    },
    {
        "id": "q018",
        "question": "Why does sequential computation limit parallelization in recurrent models?",
        "relevant_sections": ["1"],
        "category": "reasoning",
        "difficulty": "medium",
    },
    {
        "id": "q019",
        "question": "What are the three criteria used to justify self-attention?",
        "relevant_sections": ["4"],
        "category": "reasoning",
        "difficulty": "hard",
    },
    {
        "id": "q020",
        "question": "Why might self-attention produce more interpretable models?",
        "relevant_sections": ["4"],
        "category": "reasoning",
        "difficulty": "hard",
    },
    {
        "id": "q021",
        "question": "How does the model know the order of words in a sentence?",
        "relevant_sections": ["3.5"],
        "category": "paraphrase",
        "difficulty": "medium",
    },
    {
        "id": "q022",
        "question": "What makes this architecture faster to train than older sequence models?",
        "relevant_sections": ["1", "4"],
        "category": "paraphrase",
        "difficulty": "hard",
    },
    {
        "id": "q023",
        "question": "How does the network avoid forgetting information from earlier layers?",
        "relevant_sections": ["3.1"],
        "category": "paraphrase",
        "difficulty": "hard",
    },
    {
        "id": "q024",
        "question": "What stops the decoder from cheating by looking at future tokens?",
        "relevant_sections": ["3.1", "3.2.3"],
        "category": "paraphrase",
        "difficulty": "hard",
    },
    {
        "id": "q025",
        "question": "How do the authors prevent the model from memorising the training data?",
        "relevant_sections": ["5.4"],
        "category": "paraphrase",
        "difficulty": "hard",
    },
    {
        "id": "q026",
        "question": "What techniques let the model relate distant words in a sentence?",
        "relevant_sections": ["2", "4"],
        "category": "paraphrase",
        "difficulty": "hard",
    },
    {
        "id": "q027",
        "question": "How is attention computed as a weighted sum?",
        "relevant_sections": ["3.2", "3.2.1"],
        "category": "conceptual",
        "difficulty": "medium",
    },
    {
        "id": "q028",
        "question": "What are the three different ways attention is used in the model?",
        "relevant_sections": ["3.2.3"],
        "category": "conceptual",
        "difficulty": "medium",
    },
    {
        "id": "q029",
        "question": "How is the encoder different from the decoder?",
        "relevant_sections": ["3.1"],
        "category": "conceptual",
        "difficulty": "medium",
    },
    {
        "id": "q030",
        "question": "What is the overall structure of the Transformer?",
        "relevant_sections": ["3", "3.1"],
        "category": "conceptual",
        "difficulty": "medium",
    },
    {
        "id": "q031",
        "question": "How does the complexity of self-attention compare to a recurrent layer?",
        "relevant_sections": ["4"],
        "category": "comparison",
        "difficulty": "hard",
    },
    {
        "id": "q032",
        "question": "How does additive attention differ from dot-product attention?",
        "relevant_sections": ["3.2.1"],
        "category": "comparison",
        "difficulty": "medium",
    },
    {
        "id": "q033",
        "question": "How did the Transformer perform on constituency parsing?",
        "relevant_sections": ["6.3"],
        "category": "factual",
        "difficulty": "medium",
    },
    {
        "id": "q034",
        "question": "What happens to quality when the number of heads is too large?",
        "relevant_sections": ["6.2"],
        "category": "reasoning",
        "difficulty": "medium",
    },
    {
        "id": "q035",
        "question": "What future directions do the authors propose?",
        "relevant_sections": ["7"],
        "category": "conceptual",
        "difficulty": "easy",
    },
]


console = Console()


def validate(questions: list[dict]) -> list[str]:
    problems = []
    seen_ids = set()
    seen_questions = set()

    for entry in questions:
        if entry["id"] in seen_ids:
            problems.append(f"Duplicate id: {entry['id']}")
        seen_ids.add(entry["id"])

        normalised = entry["question"].lower().strip()
        if normalised in seen_questions:
            problems.append(f"Duplicate question: {entry['id']}")
        seen_questions.add(normalised)

        if not entry["relevant_sections"]:
            problems.append(f"No relevant sections: {entry['id']}")

    return problems


def check_against_index(questions: list[dict]) -> list[str]:
    index_path = OUTPUT_DIR / "attention.index.structured.json"
    if not index_path.exists():
        return ["Index not found, section coverage not verified"]

    metadata = json.loads(index_path.read_text(encoding="utf-8"))
    available = set()
    for chunk in metadata["chunks"]:
        number = chunk.get("section_number", "")
        if number:
            available.add(number)
        heading = chunk.get("section", "")
        if heading and not number:
            available.add(heading.split()[0])

    problems = []
    for entry in questions:
        for section in entry["relevant_sections"]:
            if section not in available:
                problems.append(
                    f"{entry['id']}: section '{section}' has no chunks in the index"
                )

    return problems


def main() -> None:
    problems = validate(QUESTIONS)
    problems.extend(check_against_index(QUESTIONS))

    DATASET_PATH.parent.mkdir(exist_ok=True)
    payload = {
        "document": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "question_count": len(QUESTIONS),
        "questions": QUESTIONS,
    }
    DATASET_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = Table(title="Golden dataset")
    summary.add_column("Category")
    summary.add_column("Easy", justify="right")
    summary.add_column("Medium", justify="right")
    summary.add_column("Hard", justify="right")
    summary.add_column("Total", justify="right")

    categories = sorted({entry["category"] for entry in QUESTIONS})
    for category in categories:
        rows = [entry for entry in QUESTIONS if entry["category"] == category]
        summary.add_row(
            category,
            str(sum(1 for entry in rows if entry["difficulty"] == "easy")),
            str(sum(1 for entry in rows if entry["difficulty"] == "medium")),
            str(sum(1 for entry in rows if entry["difficulty"] == "hard")),
            str(len(rows)),
        )

    summary.add_row(
        "TOTAL",
        str(sum(1 for entry in QUESTIONS if entry["difficulty"] == "easy")),
        str(sum(1 for entry in QUESTIONS if entry["difficulty"] == "medium")),
        str(sum(1 for entry in QUESTIONS if entry["difficulty"] == "hard")),
        str(len(QUESTIONS)),
    )

    console.print(summary)
    console.print()
    console.print(f"[bold green]Saved:[/] {DATASET_PATH}")

    if problems:
        console.print()
        console.print("[bold yellow]Validation warnings:[/]")
        for problem in problems:
            console.print(f"  {problem}")
    else:
        console.print("[bold green]Validation passed[/]")


if __name__ == "__main__":
    main()