import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterator

from .answer import NO_ANSWER_TEXT, Answer, Source
from .vector_store import VectorStore

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_URL = "http://localhost:11434"

# Ollama's default context window is small and longer prompts are cut
# silently; five chunks in Russian plus instructions need more room.
CONTEXT_TOKENS = 8192

# Low temperature: answers should follow the sources, not improvise.
TEMPERATURE = 0.1

# Answers on CPU can take minutes; fail only when the server is clearly stuck.
TIMEOUT_SECONDS = 600

# The prompt stays in Russian on purpose: the docs and the questions are Russian,
# and the model answers in the language of the question.
SYSTEM_PROMPT = f"""\
Ты — ассистент по технической документации.

Тебе дают пронумерованные фрагменты документации [1], [2], ... и вопрос.
Правила:
1. Отвечай только по этим фрагментам. Не добавляй общие знания.
2. После каждого утверждения ставь номер фрагмента, на который оно опирается, \
в квадратных скобках, например: «Тело запроса описывается моделью Pydantic [2].»
3. Используй только номера переданных фрагментов.
4. Если во фрагментах нет ответа, ответь ровно: «{NO_ANSWER_TEXT}»
5. Отвечай на языке вопроса, кратко. Команды и пути приводи как в документации."""

# "[2]", "[1, 3]" or "[1,3]" as written by the model, with the spaces before
# it, so a dropped marker takes its leading space along.
MARKER = re.compile(r"([ \t]*)\[(\d+(?:\s*,\s*\d+)*)\]")


class OllamaError(Exception):
    """Raised when the Ollama server is unreachable or rejects the request."""


class ModelDerailedError(OllamaError):
    """Raised when the model's reply slides into another script (e.g. Qwen switching to Chinese)."""


# Chinese, Japanese and Korean characters. The docs and questions are Russian or
# English, so such characters in a reply mean the model has derailed: small Qwen
# models sometimes switch to Chinese mid-answer and then write unrelated text.
DERAILED_SCRIPT = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
DERAILED_MESSAGE = (
    "The model went off track and started writing in another language. Generation stopped: ask the question again."
)


def build_messages(results: list[dict], question: str) -> list[dict]:
    """
    Build the chat messages: numbered source fragments followed by the question.

    The labels (file, section, question) are in Russian to match the prompt.

    Ollama models have no native citations, so each chunk gets a number the
    model is asked to put after the statements it supports.

    Args:
        results: Items returned by VectorStore.search().
        question: The user's question.

    Returns:
        Messages for Ollama's /api/chat.
    """
    fragments = "\n\n".join(
        f"[{number}] Файл: {result['metadata']['source']} | Раздел: {result['metadata']['title']}\n"
        f"{result['text']}"
        for number, result in enumerate(results, 1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Фрагменты документации:\n\n{fragments}\n\nВопрос: {question}"},
    ]


def parse_answer(text: str, results: list[dict]) -> Answer:
    """
    Turn the model's "[n]" markers into numbered sources.

    Sources are renumbered in the order they are first cited, so the answer
    reads [1], [2], ... regardless of the retrieval order. Markers pointing
    outside the given fragments (a small model may invent them) are dropped.

    Args:
        text: Answer text produced by the model.
        results: The search results that were sent as numbered fragments.

    Returns:
        The answer with renumbered markers and the cited sources.
    """
    sources: dict[int, Source] = {}  # keyed by 0-based result index

    def renumber(match: re.Match) -> str:
        numbers: list[int] = []
        for raw in match.group(2).split(","):
            index = int(raw) - 1
            if not 0 <= index < len(results):
                continue
            if index not in sources:
                metadata = results[index]["metadata"]
                sources[index] = Source(
                    number=len(sources) + 1, file=metadata["source"], title=metadata["title"],
                    chunk_id=results[index].get("id", ""),
                )
            if sources[index].number not in numbers:
                numbers.append(sources[index].number)
        if not numbers:
            return ""
        return match.group(1) + "".join(f"[{number}]" for number in numbers)

    return Answer(
        text=MARKER.sub(renumber, text).strip(),
        sources=sorted(sources.values(), key=lambda source: source.number),
        retrieved=len(results),
    )


def stream_reply(
    results: list[dict], question: str, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_URL,
) -> Iterator[str]:
    """
    Stream the model's raw reply (with "[n]" markers) piece by piece.

    The HTTP connection stays open while the generator is consumed; closing
    the generator early (e.g. the user pressed "stop") closes the connection,
    and Ollama stops generating.

    Args:
        results: Retrieved chunks to pass as numbered fragments.
        question: The user's question.
        model: Ollama model name, e.g. "qwen2.5:7b".
        base_url: Ollama server address.

    Yields:
        Pieces of the reply text in order.

    Raises:
        OllamaError: The server is not running, the model is not pulled,
            or the request failed.
        ModelDerailedError: The reply switched to Chinese/Japanese/Korean;
            generation is stopped at that point.
    """
    payload = {
        "model": model,
        "messages": build_messages(results, question),
        "stream": True,
        "options": {"num_ctx": CONTEXT_TOKENS, "temperature": TEMPERATURE},
    }
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        if error.code == 404 and "not found" in detail:
            raise OllamaError(f"Model {model} is not pulled. Run `ollama pull {model}`.") from error
        raise OllamaError(f"Ollama returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise OllamaError(f"Ollama is not running at {base_url}. Start the Ollama app or `ollama serve`.") from error

    with response:
        for line in response:  # one JSON object per line
            if not line.strip():
                continue
            event = json.loads(line.decode("utf-8"))
            if "error" in event:
                raise OllamaError(f"Ollama error: {event['error']}")
            piece = event.get("message", {}).get("content", "")
            if DERAILED_SCRIPT.search(piece):
                # Leaving the with-block closes the connection, so Ollama stops generating.
                raise ModelDerailedError(DERAILED_MESSAGE)
            if piece:
                yield piece
            if event.get("done"):
                return


def answer_question(
    question: str,
    store: VectorStore,
    top_k: int = 5,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_URL,
    results: list[dict] | None = None,
) -> Answer:
    """
    Answer a question about the docs with a local Ollama model.

    Args:
        question: The user's question.
        store: Vector store with indexed chunks.
        top_k: How many chunks to retrieve and pass to the model.
        model: Ollama model name, e.g. "qwen2.5:7b".
        base_url: Ollama server address.
        results: Chunks already retrieved for this question; searched with
            top_k when not given.

    Returns:
        The model's answer with cited sources.

    Raises:
        OllamaError: The server is not running, the model is not pulled,
            or the request failed.
        ModelDerailedError: The reply switched to Chinese/Japanese/Korean;
            generation is stopped at that point.
    """
    if results is None:
        results = store.search(question, top_k=top_k)
    if not results:
        return Answer(text="The index is empty. Index the docs first.", sources=[], retrieved=0)
    reply = "".join(stream_reply(results, question, model=model, base_url=base_url))
    return parse_answer(reply, results)


if __name__ == "__main__":
    # Example usage: needs a running Ollama with the model pulled and an indexed store
    question = "Как разрешить запросы с другого домена?"
    answer = answer_question(question, VectorStore())

    print(f"Q: {question}\n")
    print(answer.text)
    print(f"\nSources ({len(answer.sources)} of {answer.retrieved} retrieved chunks cited):")
    for source in answer.sources:
        print(f"  [{source.number}] {source.file} — {source.title}")
