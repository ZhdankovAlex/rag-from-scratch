import hashlib
from functools import cache

import numpy as np
from sentence_transformers import SentenceTransformer

# Multilingual model: the indexed docs are in Russian, while the previous
# bge-small-en-v1.5 was trained on English only.
MODEL_NAME = "intfloat/multilingual-e5-small"

# E5 models are trained with role prefixes and lose quality without them:
# questions get "query: ", indexed texts get "passage: ".
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


@cache
def get_model() -> SentenceTransformer:
    """
    Load the embedding model on first use and reuse it afterwards.

    Deferred so that importing this module (and everything that imports it)
    does not load ~470 MB of weights or contact the Hugging Face Hub, e.g.
    in tests that replace the embedding functions.

    Returns:
        The shared SentenceTransformer instance.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_query(text: str) -> list[float]:
    """
    Create an embedding for a search query.

    Args:
        text: The user's question or search query.

    Returns:
        L2-normalized embedding as a list of floats.
    """
    embedding = get_model().encode(QUERY_PREFIX + text, normalize_embeddings=True)
    return embedding.tolist()


def embed_passages(texts: list[str]) -> list[list[float]]:
    """
    Create embeddings for documents (chunks) that will be searched over.

    Texts are encoded in batches, which is much faster than one call per text.

    Args:
        texts: Chunk texts to embed.

    Returns:
        L2-normalized embeddings, one per input text, in the same order.
    """
    embeddings = get_model().encode(
        [PASSAGE_PREFIX + text for text in texts],
        normalize_embeddings=True,
        batch_size=32,
    )
    return embeddings.tolist()


def passage_fingerprint(text: str) -> str:
    """
    Identify the embedding embed_passages() would produce for a text.

    Covers everything the vector depends on: the model, the passage prefix
    and the text. Equal fingerprints mean a stored embedding can be reused.

    Args:
        text: Passage text as it would be passed to embed_passages().

    Returns:
        Hex SHA-256 digest.
    """
    return hashlib.sha256(f"{MODEL_NAME}\n{PASSAGE_PREFIX}{text}".encode()).hexdigest()


def cosine_similarity(query: str, passage: str) -> float:
    """
    Calculate cosine similarity between a query and a passage.

    Embeddings are L2-normalized, so the dot product equals the cosine.

    Args:
        query: Query text.
        passage: Passage text.

    Returns:
        A float value in range [-1.0, 1.0] representing cosine similarity.
    """
    query_embedding = np.array(embed_query(query))
    passage_embedding = np.array(embed_passages([passage])[0])
    return float(np.dot(query_embedding, passage_embedding))


if __name__ == "__main__":
    # Example usage for embed_query
    sample_text = "Как разрешить запросы с другого домена?"
    embedding = embed_query(sample_text)
    print(f"Text: {sample_text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print()

    # Example usage for cosine_similarity
    query = "запросы с другого источника"
    related = "Чтобы разрешить запросы с другого источника, добавьте CORSMiddleware."
    unrelated = "Пицца с грибами и сыром."

    sim_related = cosine_similarity(query, related)
    sim_unrelated = cosine_similarity(query, unrelated)

    print(f"Cosine similarity between '{query}' and '{related}': {sim_related:.4f}")
    print(f"Cosine similarity between '{query}' and '{unrelated}': {sim_unrelated:.4f}")
