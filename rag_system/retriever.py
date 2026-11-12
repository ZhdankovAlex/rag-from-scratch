import numpy as np

from .embeddings import embed_query
from .index import IndexedChunk


def retrieve(
    query: str,
    indexed_chunks: list[IndexedChunk],
    top_k: int = 3,
) -> list[tuple[IndexedChunk, float]]:
    """
    Retrieve top-k most relevant chunks for a given query using semantic search.

    Args:
        query: The search query text.
        indexed_chunks: List of IndexedChunk objects to search through.
        top_k: Number of top results to return. Defaults to 3.

    Returns:
        List of tuples (IndexedChunk, similarity_score) sorted by relevance.
    """
    query_embedding = np.array(embed_query(query))

    # Calculate cosine similarity with all chunks
    similarities = []
    for idx_chunk in indexed_chunks:
        chunk_embedding = np.array(idx_chunk.embedding)
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        similarities.append((idx_chunk, similarity))

    # Sort by similarity (descending) and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


if __name__ == "__main__":
    # Full workflow demo
    from .chunker import chunk_documents
    from .index import index_chunks
    from .loader import load_documents

    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Indexing chunks...")
    indexed = index_chunks(chunks)
    print(f"Indexed {len(indexed)} chunks")

    print("\nRetrieval test:")
    query = "как разрешить запросы с другого домена"
    results = retrieve(query, indexed, top_k=3)

    print(f"Query: '{query}'")
    print(f"Top {len(results)} results:")
    for i, (chunk, similarity) in enumerate(results, 1):
        print(f"  {i}. Similarity: {similarity:.4f}")
        print(f"     Source: {chunk.chunk.source}")
        print(f"     Title: {chunk.chunk.title}")
        print()
