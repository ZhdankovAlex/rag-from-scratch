"""
Command-line interface for the RAG system over the Markdown docs in docs/.

Run as `python main.py <command>` or `python -m rag_system <command>`.

Commands:
    index           Load docs/, chunk them and sync the Chroma vector store.
    search QUERY    Show the chunks retrieval finds (no LLM call).
    ask QUESTION    Answer one question, citing doc sections.
    chat            Ask questions in a loop until an empty line or Ctrl+C.
    eval            Measure retrieval quality on eval/retrieval_questions.json.

Answers come from a local Ollama model (OLLAMA_MODEL, OLLAMA_URL in .env).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from dotenv import load_dotenv

# Hugging Face warns about symlinks on Windows without Developer Mode;
# the cache works without them, so the warning is just noise here.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Pipeline modules pull in torch, sentence-transformers and chromadb, which
# take seconds to import; commands import them lazily so `--help` and
# argument errors stay instant.
if TYPE_CHECKING:
    from .answer import Answer
    from .vector_store import VectorStore

DEFAULT_TOP_K = 5


def cmd_index(args: argparse.Namespace) -> int:
    """
    Rebuild the vector store from the docs directory.

    Args:
        args: Parsed CLI arguments (unused).

    Returns:
        Process exit code.
    """
    from .chunker import chunk_documents
    from .loader import DOCS_DIR, load_documents
    from .vector_store import VectorStore

    started = time.perf_counter()
    documents = load_documents()
    if not documents:
        print(f"No .md files found in {DOCS_DIR}", file=sys.stderr)
        return 1

    chunks = chunk_documents(documents)
    print(f"Loaded {len(documents)} documents, created {len(chunks)} chunks")

    store = VectorStore()
    stats = store.sync_chunks(chunks)
    elapsed = time.perf_counter() - started
    print(
        f"Vector store holds {store.count()} chunks: {stats.added} added, {stats.updated} updated, "
        f"{stats.unchanged} unchanged, {stats.removed} removed ({elapsed:.1f}s)"
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """
    Print the chunks retrieved for a query, without calling the LLM.

    Args:
        args: Parsed CLI arguments: query, top_k.

    Returns:
        Process exit code.
    """
    store = open_store()
    if store is None:
        return 1

    for rank, result in enumerate(store.search(args.query, top_k=args.top_k), 1):
        similarity = 1 - result["distance"]  # Chroma returns cosine distance
        preview = " ".join(result["text"].split())[:160]
        print(f"{rank}. [{similarity:.3f}] {result['metadata']['source']} — {result['metadata']['title']}")
        print(f"   {preview}...")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """
    Answer a single question with the local Ollama model.

    Args:
        args: Parsed CLI arguments: question, top_k.

    Returns:
        Process exit code.
    """
    store = open_store()
    if store is None:
        return 1
    return 0 if ask_and_print(args.question, store, args.top_k) else 1


def cmd_chat(args: argparse.Namespace) -> int:
    """
    Answer questions in a loop. Each question is answered independently.

    Args:
        args: Parsed CLI arguments: top_k.

    Returns:
        Process exit code.
    """
    store = open_store()
    if store is None:
        return 1

    print("Ask a question about the docs. Empty line or Ctrl+C to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0
        ask_and_print(question, store, args.top_k)


def cmd_eval(args: argparse.Namespace) -> int:
    """
    Report retrieval quality (hit@k, MRR) on the evaluation question set.

    Dev and test are reported separately: dev questions were used to choose
    retrieval settings, so only test gives an unbiased estimate.

    Args:
        args: Parsed CLI arguments: verbose, split.

    Returns:
        Process exit code.
    """
    from .evaluate import SEARCH_DEPTH, evaluate_retrieval, load_questions

    store = open_store()
    if store is None:
        return 1

    splits = [args.split] if args.split else ["dev", "test"]
    for split in splits:
        questions = load_questions(split=split)
        report = evaluate_retrieval(store, questions)
        if args.verbose:
            print(f"[{split}]")
            for result in report.results:
                rank = str(result.rank) if result.rank else f">{SEARCH_DEPTH}"
                mark = "+" if result.rank == 1 else ("~" if result.rank else "-")
                print(f"{mark} rank {rank:>3}  {result.question.question}")
                if result.rank != 1:
                    print(f"            top-1: {result.top_source}, expected: {', '.join(result.question.expected)}")
            print()

        hits = "  ".join(f"hit@{k} = {share:.0%}" for k, share in report.hit_at.items())
        negatives = sum(1 for question in questions if question.negative)
        skipped = f" (+{negatives} negative, not scored)" if negatives else ""
        print(f"{split:>4}: {len(report.results)} questions{skipped}: {hits}  MRR = {report.mrr:.3f}")
    return 0


def open_store() -> VectorStore | None:
    """
    Open the vector store and check that it has been indexed.

    Returns:
        The store, or None (after telling the user how to fix it) when it is empty.
    """
    from .vector_store import VectorStore

    store = VectorStore()
    if store.count() == 0:
        print("The index is empty. Run `python main.py index` first.", file=sys.stderr)
        return None
    return store


@contextmanager
def waiting(message: str, stream: TextIO | None = None) -> Iterator[None]:
    """
    Show "<message> <seconds>s" on one line while the block runs, then erase it.

    A local model on CPU can take a minute to answer with no output; the
    ticking counter shows the CLI is not stuck. Nothing is written when the
    stream is not a terminal (output piped to a file).

    Args:
        message: Text shown before the elapsed seconds.
        stream: Where to draw the status line; stderr by default, so the
            answer on stdout stays clean.
    """
    stream = stream or sys.stderr
    if not stream.isatty():
        yield
        return

    done = threading.Event()
    started = time.monotonic()

    def tick() -> None:
        while not done.wait(0.5):
            stream.write(f"\r{message} {time.monotonic() - started:.0f}s")
            stream.flush()

    stream.write(f"\r{message}")
    stream.flush()
    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    try:
        yield
    finally:
        done.set()
        ticker.join()
        # Erase the status line before the answer or an error is printed
        stream.write("\r" + " " * (len(message) + 8) + "\r")
        stream.flush()


def ask_and_print(
    question: str,
    store: VectorStore,
    top_k: int,
) -> bool:
    """
    Ask the local model one question and print the answer, reporting failures readably.

    Args:
        question: The user's question.
        store: Indexed vector store.
        top_k: Number of chunks to retrieve.

    Returns:
        True when an answer was printed.
    """
    answer = answer_with_ollama(question, store, top_k)
    if answer is None:
        return False

    print_answer(answer)
    return True


def answer_with_ollama(question: str, store: VectorStore, top_k: int) -> Answer | None:
    """
    Get an answer from a local Ollama model, printing a readable message on failure.

    The model and server come from OLLAMA_MODEL and OLLAMA_URL in .env.

    Args:
        question: The user's question.
        store: Indexed vector store.
        top_k: Number of chunks to retrieve.

    Returns:
        The answer, or None if the request failed.
    """
    from .ollama_chat import DEFAULT_MODEL, DEFAULT_URL, OllamaError, answer_question

    model = os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
    base_url = (os.environ.get("OLLAMA_URL") or DEFAULT_URL).rstrip("/")
    try:
        with waiting(f"Searching the docs and asking {model} (local, may take a minute)..."):
            return answer_question(question, store, top_k=top_k, model=model, base_url=base_url)
    except OllamaError as error:
        print(error, file=sys.stderr)
    return None


def print_answer(answer: Answer) -> None:
    """
    Print an answer followed by its numbered sources.

    Args:
        answer: The answer to print.
    """
    # Flush so the warning on stderr lands after the answer when output is piped
    print(answer.text, flush=True)
    if answer.unsupported:
        print(
            "\n⚠ The answer cites no sources: it may come from the model's general "
            "knowledge rather than the docs. Check it with `python main.py search`.",
            file=sys.stderr,
        )
    if not answer.sources:
        return
    print(f"\nSources ({len(answer.sources)} of {answer.retrieved} retrieved chunks cited):")
    for source in answer.sources:
        print(f"  [{source.number}] {source.file} — {source.title}")


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Returns:
        Parser with index, search, ask, chat and eval subcommands.
    """
    parser = argparse.ArgumentParser(
        description="RAG over Markdown documentation: local retrieval + answers from a local LLM (Ollama).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("index", help="load docs/, chunk and sync the vector store").set_defaults(func=cmd_index)

    search = commands.add_parser("search", help="show retrieved chunks without calling the LLM")
    search.add_argument("query", help="search query")
    search.set_defaults(func=cmd_search)

    ask = commands.add_parser("ask", help="answer one question with cited sources")
    ask.add_argument("question", help="question about the docs")
    ask.set_defaults(func=cmd_ask)

    chat = commands.add_parser("chat", help="interactive question loop")
    chat.set_defaults(func=cmd_chat)

    evaluate = commands.add_parser("eval", help="measure retrieval quality on the eval question set")
    evaluate.add_argument("-v", "--verbose", action="store_true", help="print the rank for every question")
    evaluate.add_argument(
        "-s", "--split", choices=["dev", "test"],
        help="evaluate one part of the question set (default: report dev and test separately)",
    )
    evaluate.set_defaults(func=cmd_eval)

    for command in (search, ask, chat):
        command.add_argument(
            "-k", "--top-k", type=int, default=DEFAULT_TOP_K,
            help=f"number of chunks to retrieve (default: {DEFAULT_TOP_K})",
        )

    return parser


def main() -> int:
    """
    Parse arguments and run the selected command.

    Returns:
        Process exit code.
    """
    # Cyrillic output must survive redirection to a file or pipe on Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_ROOT / ".env")

    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
