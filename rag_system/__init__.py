"""
RAG pipeline over a folder of Markdown docs: load -> chunk -> embed -> store/retrieve -> answer.

Modules:
    loader        read Markdown files from docs/
    chunker       split documents into titled chunks
    embeddings    multilingual-e5 query/passage embeddings
    vector_store  Chroma collection with incremental sync and search
    index, retriever   the same retrieval in memory with NumPy (for comparison)
    answer        answer and source types
    ollama_chat   answers from a local Ollama model with [n] markers
    evaluate      retrieval metrics on the eval question set
    cli           command-line interface (`python -m rag_system`)

Each pipeline module has a demo under `__main__`: `python -m rag_system.<module>`.
"""
