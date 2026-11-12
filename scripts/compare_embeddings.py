"""
Compare embedding setups on the eval question set and save the results.

Each setup embeds the same chunks in memory (no Chroma) and is scored with
the same hit@k / MRR as `python main.py eval`. Results go to
eval/experiments.json, which the web UI shows on the "Quality" page.

Setups:
    * BAAI/bge-small-en-v1.5 — an English-only model of the same size;
    * multilingual-e5-small on the chunk text alone;
    * + the "Document > Section" title in the embedded text, without E5 prefixes;
    * + the "query: " / "passage: " prefixes (the pipeline's setup).

Usage::

    python scripts/compare_embeddings.py   # ~4 minutes on CPU, downloads bge-small-en (~130 MB) once
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.chunker import Chunk, chunk_documents  # noqa: E402
from rag_system.evaluate import evaluate_retrieval, load_questions  # noqa: E402
from rag_system.loader import load_documents  # noqa: E402

OUTPUT = PROJECT_ROOT / "eval" / "experiments.json"


@dataclass(frozen=True)
class Setup:
    """
    One embedding configuration to evaluate.

    Attributes:
        label: Human-readable name shown in the report.
        model: Hugging Face model id.
        query_prefix: Text prepended to questions.
        passage_prefix: Text prepended to chunks.
        with_title: Embed "Document > Section" + text instead of the text alone.
        current: True for the setup the pipeline uses.
    """

    label: str
    model: str
    query_prefix: str = ""
    passage_prefix: str = ""
    with_title: bool = True
    current: bool = False


E5 = "intfloat/multilingual-e5-small"
SETUPS = [
    Setup("bge-small-en-v1.5 (English-only)", "BAAI/bge-small-en-v1.5"),
    Setup("multilingual-e5-small, chunk text only", E5, "query: ", "passage: ", with_title=False),
    Setup("+ \"Document > Section\" title, no prefixes", E5),
    Setup("+ query: / passage: prefixes (current)", E5, "query: ", "passage: ", current=True),
]


class InMemoryStore:
    """Exact cosine search over chunk embeddings, with VectorStore's search() result shape."""

    def __init__(self, model: SentenceTransformer, setup: Setup, chunks: list[Chunk]):
        """
        Embed all chunks for one setup.

        Args:
            model: Loaded embedding model.
            setup: Prefixes and title option to embed with.
            chunks: Chunks to index.
        """
        self.model = model
        self.setup = setup
        self.chunks = chunks
        texts = [setup.passage_prefix + (c.embedding_text if setup.with_title else c.text) for c in chunks]
        self.matrix = model.encode(texts, batch_size=32, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Return the top_k chunks most similar to the query.

        Args:
            query: Question text.
            top_k: Number of results.

        Returns:
            Results with the "metadata" -> "source" key that evaluate_retrieval() reads.
        """
        vector = self.model.encode(self.setup.query_prefix + query, normalize_embeddings=True)
        best = np.argsort(-(self.matrix @ vector))[:top_k]
        return [{"metadata": {"source": self.chunks[i].source, "title": self.chunks[i].title}} for i in best]


def main() -> int:
    """Evaluate every setup on dev and test and write eval/experiments.json."""
    sys.stdout.reconfigure(encoding="utf-8")  # Cyrillic labels in a Windows console
    chunks = chunk_documents(load_documents())
    splits = {split: load_questions(split=split) for split in ("dev", "test")}
    models: dict[str, SentenceTransformer] = {}
    rows = []
    for setup in SETUPS:
        started = time.perf_counter()
        model = models.setdefault(setup.model, SentenceTransformer(setup.model))
        store = InMemoryStore(model, setup, chunks)
        row = {"label": setup.label, "model": setup.model, "current": setup.current}
        for split, questions in splits.items():
            report = evaluate_retrieval(store, questions)
            row[split] = {"hit_at": {str(k): v for k, v in report.hit_at.items()}, "mrr": report.mrr}
        rows.append(row)
        print(f"{setup.label}: dev MRR {row['dev']['mrr']:.3f}, test MRR {row['test']['mrr']:.3f} "
              f"({time.perf_counter() - started:.0f}s)")

    payload = {"chunks": len(chunks), "setups": rows}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
