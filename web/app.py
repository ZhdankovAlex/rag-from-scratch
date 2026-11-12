"""
FastAPI app behind the local web UI.

Pages are Jinja2 templates; everything they show comes from the JSON API below,
which wraps the same pipeline as the CLI (rag_system). Heavy objects — the
vector store with its embedding model, the eval report, corpus statistics —
are built on first use and cached until the next re-index.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from rag_system import ollama_chat
from rag_system.chunker import Chunk, chunk_documents
from rag_system.evaluate import SEARCH_DEPTH, evaluate_retrieval, load_questions
from rag_system.loader import DOCS_DIR, PROJECT_ROOT, load_documents
from rag_system.vector_store import VectorStore

WEB_DIR = Path(__file__).parent
EXPERIMENTS_FILE = PROJECT_ROOT / "eval" / "experiments.json"
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="rag-from-scratch",
    description=(
        "Local UI and API for a RAG pipeline over Markdown docs: semantic search in ChromaDB "
        "(multilingual-e5-small), answers from a local Ollama model with [n] citations, "
        "retrieval metrics on a held-out question set. Listens on 127.0.0.1 only."
    ),
    version="1.0",
)
templates = Jinja2Templates(directory=WEB_DIR / "templates")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

# (path, label, icon); the icon stays visible when the side panel is collapsed
NAV = [
    ("ask", "Спросить", "💬"),
    ("search", "Поиск", "🔎"),
    ("quality", "Качество", "📈"),
    ("documents", "Документы", "📚"),
]
# Cache-busting suffix for static files: changes on every server start.
STATIC_VERSION = str(int(time.time()))

# Re-indexing and the cached eval report must not interleave.
_index_lock = threading.Lock()
_eval_cache: dict[str, dict] = {}


# --- shared state --------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    """Open the Chroma store once; the embedding model loads on the first search."""
    return VectorStore()


@lru_cache(maxsize=1)
def corpus_chunks() -> tuple[int, list[Chunk]]:
    """
    Load and chunk docs/ (no embeddings, takes well under a second).

    Returns:
        Number of documents and the list of chunks.
    """
    documents = load_documents()
    return len(documents), chunk_documents(documents)


def ollama_settings() -> tuple[str, str]:
    """Return the Ollama model and server URL from the environment (.env), with CLI defaults."""
    model = os.environ.get("OLLAMA_MODEL") or ollama_chat.DEFAULT_MODEL
    url = (os.environ.get("OLLAMA_URL") or ollama_chat.DEFAULT_URL).rstrip("/")
    return model, url


def format_results(results: list[dict]) -> list[ChunkHit]:
    """Turn VectorStore.search() results into API objects with similarity instead of distance."""
    return [
        ChunkHit(
            rank=rank, id=r["id"], file=r["metadata"]["source"], title=r["metadata"]["title"],
            text=r["text"], similarity=round(1 - r["distance"], 4),
        )
        for rank, r in enumerate(results, 1)
    ]


# --- API models ----------------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """A retrieval query."""

    query: str = Field(min_length=1, max_length=1000, description="Question or keywords to search for.")
    top_k: int = Field(5, ge=1, le=20, description="How many chunks to return.")


class AskRequest(BaseModel):
    """A question for the LLM."""

    question: str = Field(min_length=1, max_length=1000, description="Question about the docs.")
    top_k: int = Field(5, ge=1, le=10, description="How many retrieved chunks to pass to the model.")


class ChunkHit(BaseModel):
    """One retrieved chunk."""

    rank: int = Field(description="1-based position in the search results.")
    id: str = Field(description="Stable chunk ID: file::section::part.")
    file: str = Field(description="Markdown file the chunk comes from.")
    title: str = Field(description="Breadcrumb title: Document > Section.")
    text: str = Field(description="Chunk text as stored and cited.")
    similarity: float = Field(description="Cosine similarity to the query, 0..1.")


class SearchResponse(BaseModel):
    """Search results with timing."""

    results: list[ChunkHit]
    ms: int = Field(description="Search time in milliseconds (embedding the query + vector search).")


class SourceOut(BaseModel):
    """A chunk the answer cites."""

    number: int = Field(description="Marker number in the answer text, e.g. 1 for [1].")
    file: str
    title: str
    chunk_id: str = Field(description="ID of the cited chunk, matches ChunkHit.id.")


class AskResponse(BaseModel):
    """The model's answer, its sources and the chunks it was given."""

    answer: str = Field(description="Answer text with [n] citation markers.")
    sources: list[SourceOut]
    unsupported: bool = Field(description="True when the answer cites nothing: it may come from general knowledge.")
    results: list[ChunkHit] = Field(description="Chunks passed to the model as numbered fragments.")
    model: str = Field(description="Ollama model that answered.")
    seconds: float = Field(description="Total time to answer, including retrieval.")


class OllamaStatus(BaseModel):
    """Whether the local LLM is ready."""

    url: str
    model: str = Field(description="Configured model (OLLAMA_MODEL).")
    running: bool = Field(description="The Ollama server answered.")
    model_pulled: bool = Field(description="The configured model is available locally.")


class StatusResponse(BaseModel):
    """Index and LLM readiness, shown in the side panel."""

    chunks: int = Field(description="Chunks in the vector store; 0 means `index` has not been run.")
    ollama: OllamaStatus


class IndexResponse(BaseModel):
    """Result of syncing the vector store with docs/."""

    documents: int
    chunks: int
    added: int
    updated: int
    unchanged: int
    removed: int
    seconds: float


# --- pages ---------------------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    """Open the main page."""
    return RedirectResponse("/ask")


@app.get("/{page}", response_class=HTMLResponse, include_in_schema=False)
def page(request: Request, page: str) -> HTMLResponse:
    """Render one of the UI pages listed in NAV."""
    if page not in {key for key, _, _ in NAV}:
        raise HTTPException(404, "No such page")
    return templates.TemplateResponse(
        request, f"{page}.html", {"nav": NAV, "active": page, "v": STATIC_VERSION},
    )


# --- API -----------------------------------------------------------------------------------------

@app.get("/api/status", response_model=StatusResponse, tags=["status"])
def status() -> StatusResponse:
    """Report how many chunks are indexed and whether Ollama and the configured model are ready."""
    model, url = ollama_settings()
    running, pulled = False, False
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=1.5) as response:
            names = {m["name"] for m in json.loads(response.read().decode("utf-8")).get("models", [])}
        running = True
        pulled = model in names or f"{model}:latest" in names
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return StatusResponse(
        chunks=get_store().count(),
        ollama=OllamaStatus(url=url, model=model, running=running, model_pulled=pulled),
    )


@app.post("/api/search", response_model=SearchResponse, tags=["retrieval"])
def search(body: SearchRequest) -> SearchResponse:
    """Retrieve the chunks closest to the query. No LLM call: fast and works without Ollama."""
    started = time.perf_counter()
    results = get_store().search(body.query, top_k=body.top_k)
    return SearchResponse(results=format_results(results), ms=round((time.perf_counter() - started) * 1000))


@app.post(
    "/api/ask", response_model=AskResponse, tags=["answers"],
    responses={503: {"description": "Ollama is not running or the model is not pulled."}},
)
def ask(body: AskRequest) -> AskResponse:
    """
    Answer a question with the local Ollama model, citing retrieved chunks as [n].

    On CPU this takes about a minute for a 7B model.
    """
    model, url = ollama_settings()
    started = time.perf_counter()
    store = get_store()
    results = store.search(body.question, top_k=body.top_k)
    try:
        answer = ollama_chat.answer_question(
            body.question, store, top_k=body.top_k, model=model, base_url=url, results=results,
        )
    except ollama_chat.OllamaError as error:
        raise HTTPException(503, str(error)) from error
    return AskResponse(
        answer=answer.text,
        sources=[SourceOut(number=s.number, file=s.file, title=s.title, chunk_id=s.chunk_id) for s in answer.sources],
        unsupported=answer.unsupported,
        results=format_results(results),
        model=model,
        seconds=round(time.perf_counter() - started, 1),
    )


@app.get("/api/eval", tags=["quality"])
def eval_report() -> dict:
    """
    Retrieval metrics (hit@k, MRR) per split and the rank of every question.

    Computed on first request (a few seconds) and cached until the next re-index.
    Negative questions (no answer in the docs) are listed but not scored.
    """
    with _index_lock:
        if not _eval_cache:
            store = get_store()
            for split in ("dev", "test"):
                questions = load_questions(split=split)
                report = evaluate_retrieval(store, questions)
                _eval_cache[split] = {
                    "hit_at": {str(k): v for k, v in report.hit_at.items()},
                    "mrr": report.mrr,
                    "questions": [
                        {"question": r.question.question, "expected": r.question.expected,
                         "rank": r.rank, "top_source": r.top_source}
                        for r in report.results
                    ],
                    "negatives": [q.question for q in questions if q.negative],
                }
        return {"depth": SEARCH_DEPTH, "splits": _eval_cache}


@app.get("/api/experiments", tags=["quality"], responses={404: {"description": "Comparison not run yet."}})
def experiments() -> dict:
    """Embedding setups compared by scripts/compare_embeddings.py (eval/experiments.json)."""
    if not EXPERIMENTS_FILE.is_file():
        raise HTTPException(404, "Run `python scripts/compare_embeddings.py` first.")
    return json.loads(EXPERIMENTS_FILE.read_text(encoding="utf-8"))


@app.get("/api/documents", tags=["documents"])
def documents_overview() -> dict:
    """Documents in docs/ with chunk counts and sizes, plus the chunk length distribution."""
    documents, chunks = corpus_chunks()
    by_file: dict[str, dict] = {}
    for chunk in chunks:
        entry = by_file.setdefault(
            chunk.source, {"file": chunk.source, "title": chunk.title.split(" > ")[0], "chunks": 0, "chars": 0},
        )
        entry["chunks"] += 1
        entry["chars"] += len(chunk.text)
    lengths = [len(chunk.text) for chunk in chunks]
    edges = [0, 200, 400, 600, 800, 1000, 1200]
    histogram = [
        {"label": f"{low}–{high}" if high < edges[-1] else f"{low}+",
         "count": sum(1 for n in lengths if low <= n < high or (high == edges[-1] and n >= low))}
        for low, high in zip(edges, edges[1:], strict=False)
    ]
    return {
        "documents": documents,
        "chunks": len(chunks),
        "chars": sum(lengths),
        "avg_chunk": round(sum(lengths) / len(lengths)) if lengths else 0,
        "files": sorted(by_file.values(), key=lambda e: -e["chunks"]),
        "histogram": histogram,
        "docs_dir": str(DOCS_DIR.relative_to(PROJECT_ROOT)),
    }


@app.post("/api/index", response_model=IndexResponse, tags=["documents"])
def reindex() -> IndexResponse:
    """Re-read docs/ and sync the vector store: only new or changed chunks are embedded."""
    with _index_lock:
        started = time.perf_counter()
        corpus_chunks.cache_clear()
        documents, chunks = corpus_chunks()
        store = get_store()
        stats = store.sync_chunks(chunks)
        _eval_cache.clear()
        return IndexResponse(
            documents=documents, chunks=store.count(), added=stats.added, updated=stats.updated,
            unchanged=stats.unchanged, removed=stats.removed, seconds=round(time.perf_counter() - started, 1),
        )
