"""
Shared fixtures. Tests run offline: no embedding model, no LLM server.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from rag_system import embeddings, vector_store

FAKE_DIM = 1024


def fake_vector(text: str) -> list[float]:
    """
    Deterministic bag-of-words embedding: texts sharing words are close.

    Args:
        text: Text to embed.

    Returns:
        L2-normalized vector of FAKE_DIM floats.
    """
    vector = np.zeros(FAKE_DIM)
    for word in text.lower().split():
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % FAKE_DIM] += 1
    norm = np.linalg.norm(vector)
    return (vector / norm if norm else vector).tolist()


@pytest.fixture(autouse=True)
def forbid_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test would load the real embedding model."""
    def refuse() -> None:
        raise AssertionError("tests must not load the embedding model; use fake_embeddings")

    monkeypatch.setattr(embeddings, "get_model", refuse)


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the embedding functions used by the vector store with fake_vector."""
    monkeypatch.setattr(vector_store, "embed_passages", lambda texts: [fake_vector(t) for t in texts])
    monkeypatch.setattr(vector_store, "embed_query", fake_vector)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_embeddings: None) -> vector_store.VectorStore:
    """A VectorStore persisted in a temporary directory, with fake embeddings."""
    monkeypatch.setattr(vector_store, "CHROMA_DIR", tmp_path / "chroma_db")
    return vector_store.VectorStore("test_collection")
