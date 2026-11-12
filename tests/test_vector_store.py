import pytest

from rag_system import embeddings, vector_store
from rag_system.chunker import Chunk
from rag_system.vector_store import SyncStats


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, source=f"{chunk_id}.md", title=f"Doc > {chunk_id}", text=text)


@pytest.fixture
def embedded(store, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every text the store sends for embedding."""
    texts: list[str] = []
    fake = vector_store.embed_passages

    def recording(batch: list[str]) -> list[list[float]]:
        texts.extend(batch)
        return fake(batch)

    monkeypatch.setattr(vector_store, "embed_passages", recording)
    return texts


def test_resync_of_unchanged_chunks_embeds_nothing(store, embedded):
    chunks = [make_chunk("a", "backup"), make_chunk("b", "replication")]
    assert store.sync_chunks(chunks) == SyncStats(added=2)
    embedded.clear()

    stats = store.sync_chunks(chunks)

    assert stats == SyncStats(unchanged=2)
    assert embedded == []


def test_only_changed_chunks_are_reembedded(store, embedded):
    store.sync_chunks([make_chunk("a", "backup"), make_chunk("b", "replication"), make_chunk("c", "alerts")])
    embedded.clear()

    retitled = Chunk(id="b", source="b.md", title="Doc > renamed", text="replication")
    stats = store.sync_chunks([make_chunk("a", "backup v2"), retitled, make_chunk("d", "trace")])

    assert stats == SyncStats(added=1, updated=2, unchanged=0, removed=1)
    assert sorted(embedded) == sorted(["Doc > a\n\nbackup v2", "Doc > renamed\n\nreplication", "Doc > d\n\ntrace"])


def test_changing_the_embedding_model_reembeds_everything(store, embedded, monkeypatch):
    chunks = [make_chunk("a", "backup"), make_chunk("b", "replication")]
    store.sync_chunks(chunks)
    embedded.clear()

    monkeypatch.setattr(embeddings, "MODEL_NAME", "another/model")
    stats = store.sync_chunks(chunks)

    assert stats == SyncStats(updated=2)
    assert len(embedded) == 2


def test_sync_is_idempotent(store):
    chunks = [make_chunk("a", "backup restore"), make_chunk("b", "replication slots")]

    store.sync_chunks(chunks)
    store.sync_chunks(chunks)

    assert store.count() == 2


def test_sync_removes_chunks_no_longer_present(store):
    store.sync_chunks([make_chunk("a", "backup"), make_chunk("b", "replication")])

    store.sync_chunks([make_chunk("a", "backup")])

    assert store.count() == 1
    assert [r["id"] for r in store.search("replication", top_k=5)] == ["a"]


def test_sync_updates_changed_text(store):
    # collection.add() would silently keep "old text" for an existing ID.
    store.sync_chunks([make_chunk("a", "old text")])

    store.sync_chunks([make_chunk("a", "new text")])

    assert store.search("new text", top_k=1)[0]["text"] == "new text"


def test_sync_writes_in_batches(store, monkeypatch):
    monkeypatch.setattr(store.client, "get_max_batch_size", lambda: 2)
    chunks = [make_chunk(str(i), f"text {i}") for i in range(5)]

    store.sync_chunks(chunks)

    assert store.count() == 5


def test_chunk_is_found_by_words_only_in_its_title(store):
    # "email" is only in the title "Doc > email", not in the chunk text.
    store.sync_chunks([make_chunk("email", "smtp host and port"), make_chunk("backup", "restore database")])

    best = store.search("email", top_k=1)[0]

    assert best["id"] == "email"
    assert best["text"] == "smtp host and port"  # stored text has no title prepended


def test_search_returns_closest_chunk_with_metadata(store):
    store.sync_chunks([
        make_chunk("backup", "restore database from backup"),
        make_chunk("alerts", "email alert when metric exceeds threshold"),
    ])

    best = store.search("restore from backup", top_k=1)[0]

    assert best["id"] == "backup"
    assert best["metadata"] == {"source": "backup.md", "title": "Doc > backup"}
    assert 0 <= best["distance"] < 1
