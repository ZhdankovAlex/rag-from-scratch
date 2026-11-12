from dataclasses import dataclass

from .chunker import Chunk
from .embeddings import embed_passages


@dataclass
class IndexedChunk:
    chunk: Chunk
    embedding: list[float]


def index_chunks(chunks: list[Chunk]) -> list[IndexedChunk]:
    """
    Create embeddings for each chunk and return indexed chunks.

    Args:
        chunks: List of Chunk objects to index.

    Returns:
        List of IndexedChunk objects with embeddings.
    """
    embeddings = embed_passages([chunk.embedding_text for chunk in chunks])
    return [
        IndexedChunk(chunk=chunk, embedding=embedding)
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


if __name__ == "__main__":
    # Example usage
    from .chunker import chunk_documents
    from .loader import load_documents

    documents = load_documents()
    chunks = chunk_documents(documents)
    indexed = index_chunks(chunks)

    print(f"Indexed {len(indexed)} chunks:")
    for idx_chunk in indexed[:3]:  # Show first 3
        print(f"  [{idx_chunk.chunk.id}] {idx_chunk.chunk.title}")
        print(f"    Embedding dimension: {len(idx_chunk.embedding)}")
