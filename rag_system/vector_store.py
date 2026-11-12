from dataclasses import dataclass

import chromadb

from .chunker import Chunk
from .embeddings import embed_passages, embed_query, passage_fingerprint
from .loader import PROJECT_ROOT

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Metadata key holding passage_fingerprint() of the embedded text.
FINGERPRINT_KEY = "fingerprint"


@dataclass
class SyncStats:
    """
    What VectorStore.sync_chunks() changed.

    Attributes:
        added: Chunks with new IDs (embedded and written).
        updated: Chunks whose text, title or embedding model changed
            (re-embedded and overwritten).
        unchanged: Chunks reused as stored, without computing embeddings.
        removed: Stored chunks no longer produced by the chunker (deleted).
    """

    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0


class VectorStore:
    def __init__(self, collection_name: str = "rag_documents"):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the Chroma collection.
        """
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def sync_chunks(self, chunks: list[Chunk]) -> SyncStats:
        """
        Make the collection mirror the given chunks, embedding only what changed.

        Each record stores a fingerprint of its embedded text (model, prefix,
        title and text). Chunks whose fingerprint matches the stored one are
        skipped, so re-indexing unchanged docs computes no embeddings.
        Changed and new chunks are upserted by their stable ID (upsert, not
        add: collection.add() silently keeps the old version for existing
        IDs). Records whose IDs are no longer produced by the chunker (a
        section was renamed or removed) are deleted, so stale text never
        shows up in search results.

        Args:
            chunks: The full current set of chunks to store.

        Returns:
            Counts of added, updated, unchanged and removed chunks.
        """
        stored = self.collection.get(include=["metadatas"])
        stored_fingerprints = {
            record_id: (metadata or {}).get(FINGERPRINT_KEY)
            for record_id, metadata in zip(stored["ids"], stored["metadatas"], strict=True)
        }

        stats = SyncStats()
        to_write: list[tuple[Chunk, str]] = []
        for chunk in chunks:
            fingerprint = passage_fingerprint(chunk.embedding_text)
            if chunk.id not in stored_fingerprints:
                stats.added += 1
            elif stored_fingerprints[chunk.id] != fingerprint:
                stats.updated += 1
            else:
                stats.unchanged += 1
                continue
            to_write.append((chunk, fingerprint))

        if to_write:
            embeddings = embed_passages([chunk.embedding_text for chunk, _ in to_write])
            # Chroma rejects writes larger than the client's max batch size.
            batch_size = self.client.get_max_batch_size()
            for start in range(0, len(to_write), batch_size):
                batch = to_write[start:start + batch_size]
                self.collection.upsert(
                    ids=[chunk.id for chunk, _ in batch],
                    embeddings=embeddings[start:start + batch_size],
                    documents=[chunk.text for chunk, _ in batch],
                    metadatas=[
                        {"source": chunk.source, "title": chunk.title, FINGERPRINT_KEY: fingerprint}
                        for chunk, fingerprint in batch
                    ],
                )

        stale_ids = set(stored_fingerprints) - {chunk.id for chunk in chunks}
        if stale_ids:
            self.collection.delete(ids=list(stale_ids))
        stats.removed = len(stale_ids)
        return stats

    def count(self) -> int:
        """
        Return the number of chunks stored in the collection.
        """
        return self.collection.count()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search for similar chunks using Chroma's built-in search.

        Args:
            query: Search query text.
            top_k: Number of top results to return.

        Returns:
            List of dictionaries containing chunk data, similarity/distance, and metadata.
        """
        query_embedding = embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        formatted_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                # Internal fields such as the fingerprint stay out of results
                'metadata': {'source': metadata['source'], 'title': metadata['title']},
                'distance': results['distances'][0][i]
            })

        return formatted_results


if __name__ == "__main__":
    # Example usage
    from .chunker import chunk_documents
    from .loader import load_documents

    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Initializing vector store...")
    store = VectorStore()

    print("Syncing chunks to vector store...")
    stats = store.sync_chunks(chunks)
    print(f"Collection now holds {store.count()} chunks ({stats})")

    print("\nSearch test:")
    query = "как разрешить запросы с другого домена"
    results = store.search(query, top_k=3)

    print(f"Query: '{query}'")
    print(f"Top {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Distance: {result['distance']:.4f}")
        print(f"     Source: {result['metadata']['source']}")
        print(f"     Title: {result['metadata']['title']}")
        print(f"     Text preview: {result['text'][:100]}...")
        print()
