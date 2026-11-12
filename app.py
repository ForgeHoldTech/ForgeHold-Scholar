import json
import os
import sqlite3
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import requests
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from agent import DATABASE_PATH, Toolbox, build_catalog, run_agent
from corpus_answer import (
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    INDEX_PATH,
    RERANKER_MODEL,
    answer_question,
    load_index,
    tokenize,
)


WEB_DIR = Path("web")
RETRIEVAL_REPORT = Path("output/corpus_evaluation_report.json")
ANSWER_REPORT = Path("output/answer_evaluation.json")

HOST = "127.0.0.1"
PORT = 8000
SOURCE_TEXT_LIMIT = 700
OBSERVATION_LIMIT = 400
GRADES = {"yes", "partial", "no"}

state: dict = {}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["quick", "agent"] = "quick"
    paper: str | None = Field(default=None, max_length=120)
    year: str | None = Field(default=None, max_length=4)


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_dotenv()

    if not INDEX_PATH.exists():
        raise RuntimeError("Index not found. Run build_index.py first.")

    index, chunks, vectors = load_index()

    build_catalog(chunks).close()
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH.as_posix()}?mode=ro",
        uri=True,
        check_same_thread=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    reranker = CrossEncoder(RERANKER_MODEL, device=device)
    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    state.update(
        index=index,
        chunks=chunks,
        vectors=vectors,
        model=model,
        reranker=reranker,
        bm25=bm25,
        device=device,
        toolbox=Toolbox(model, reranker, bm25, chunks, vectors, connection),
        api_key=os.getenv("GEMINI_API_KEY"),
        lock=threading.Lock(),
    )

    yield

    connection.close()
    state.clear()


app = FastAPI(title="ForgeHold Scholar", version="1.0.0", lifespan=lifespan)


def load_metrics() -> dict:
    metrics: dict = {}

    if RETRIEVAL_REPORT.exists():
        payload = json.loads(RETRIEVAL_REPORT.read_text(encoding="utf-8"))
        results = payload.get("results", [])
        if results:
            best = max(results, key=lambda result: result.get("chunk_mrr", 0.0))
            metrics["retrieval"] = {
                "questions": payload.get("question_count"),
                "configuration": best.get("configuration"),
                "chunk_recall_at_5": best.get("chunk_recall_at_5"),
                "chunk_mrr": best.get("chunk_mrr"),
                "paper_recall_at_1": best.get("paper_recall_at_1"),
            }

    if ANSWER_REPORT.exists():
        payload = json.loads(ANSWER_REPORT.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        answered = [record for record in records if not record.get("refused")]
        graded = [record for record in answered if record.get("correctness") in GRADES]
        if records:
            correct = sum(1 for record in graded if record["correctness"] == "yes")
            metrics["answers"] = {
                "questions": len(records),
                "answered_rate": len(answered) / len(records),
                "correct_rate": correct / len(graded) if graded else None,
                "judge": "same model as the generator, synthetic questions",
            }

    return metrics


def source_payload(passage, cited: list[int]) -> dict:
    return {
        "number": passage.number,
        "paper_id": passage.paper_id,
        "paper_title": passage.paper_title,
        "year": passage.published[:4],
        "section": passage.section,
        "pages": passage.pages,
        "cited": passage.number in cited,
        "text": passage.text[:SOURCE_TEXT_LIMIT],
    }


@app.get("/")
def home():
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="web/index.html is missing.")
    return FileResponse(page)


@app.get("/api/status")
def status() -> dict:
    index = state["index"]
    return {
        "papers": index["document_count"],
        "chunks": index["chunk_count"],
        "device": state["device"],
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "generator_model": GEMINI_MODEL,
        "generator_configured": bool(state["api_key"]),
        "metrics": load_metrics(),
    }


@app.get("/api/papers")
def papers() -> dict:
    counts = Counter(chunk["paper_id"] for chunk in state["chunks"])
    entries = [
        {
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "published": paper.get("published", ""),
            "categories": paper.get("categories", []),
            "chunks": counts.get(paper["paper_id"], 0),
        }
        for paper in state["index"]["papers"]
    ]
    entries.sort(key=lambda entry: entry["published"], reverse=True)
    return {"count": len(entries), "papers": entries}


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    question = request.question.strip()
    paper = (request.paper or "").strip() or None
    year = (request.year or "").strip() or None

    api_key = state["api_key"]
    if not api_key:
        raise HTTPException(
            status_code=503, detail="GEMINI_API_KEY is not configured on the server."
        )

    started = time.perf_counter()

    try:
        with state["lock"]:
            if request.mode == "agent":
                run = run_agent(
                    question,
                    state["toolbox"],
                    "gemini",
                    api_key,
                    state["index"]["document_count"],
                )
                return {
                    "mode": "agent",
                    "question": question,
                    "answer": run.answer,
                    "refused": run.answer.upper().startswith("NOT_IN_CONTEXT"),
                    "seconds": round(time.perf_counter() - started, 2),
                    "model": GEMINI_MODEL,
                    "invalid_citations": run.invalid_citations,
                    "sources": [
                        source_payload(passage, run.cited) for passage in run.passages
                    ],
                    "steps": [
                        {
                            "number": step.number,
                            "tool": step.tool,
                            "thought": step.thought,
                            "arguments": step.arguments,
                            "observation": step.observation[:OBSERVATION_LIMIT],
                            "seconds": round(step.seconds, 2),
                        }
                        for step in run.steps
                    ],
                }

            answer = answer_question(
                state["model"],
                state["reranker"],
                state["bm25"],
                state["chunks"],
                state["vectors"],
                question,
                "gemini",
                api_key,
                paper,
                year,
            )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="The language model timed out.")
    except requests.exceptions.RequestException:
        raise HTTPException(
            status_code=502, detail="The language model could not be reached."
        )
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)[:200])

    return {
        "mode": "quick",
        "question": question,
        "answer": answer.text,
        "refused": answer.refused,
        "seconds": round(time.perf_counter() - started, 2),
        "model": answer.model or GEMINI_MODEL,
        "invalid_citations": answer.invalid_citations,
        "sources": [
            source_payload(passage, answer.cited) for passage in answer.passages
        ],
        "steps": [],
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)