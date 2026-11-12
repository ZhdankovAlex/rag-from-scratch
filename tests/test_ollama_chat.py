import io
import json
import urllib.error

import pytest

from rag_system import ollama_chat
from rag_system.ollama_chat import CONTEXT_TOKENS, OllamaError, answer_question, build_messages, parse_answer


def result(source: str, title: str, text: str = "chunk text") -> dict:
    return {"id": source, "text": text, "metadata": {"source": source, "title": title}, "distance": 0.1}


RESULTS = [
    result("backup.md", "Backup > Restore", "uvicorn restore"),
    result("api.md", "API > Backups", "GET /backups"),
    result("alerting.md", "Alerting > Email", "smtp"),
]


class FakeStore:
    def __init__(self, results: list[dict]):
        self.results = results

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        return self.results[:top_k]


def test_build_messages_numbers_fragments_with_file_and_section():
    messages = build_messages(RESULTS[:2], "Как восстановить?")

    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "[1] Файл: backup.md | Раздел: Backup > Restore\nuvicorn restore" in user
    assert "[2] Файл: api.md" in user
    assert user.endswith("Вопрос: Как восстановить?")


def test_parse_answer_renumbers_sources_in_order_of_first_citation():
    answer = parse_answer("Выберите бэкап [2]. Запустите restore [1][2].", RESULTS)

    assert answer.text == "Выберите бэкап [1]. Запустите restore [2][1]."
    assert [(s.number, s.file) for s in answer.sources] == [(1, "api.md"), (2, "backup.md")]
    assert answer.retrieved == 3


def test_parse_answer_handles_grouped_and_repeated_markers():
    answer = parse_answer("Нужны Uvicorn и SMTP [3, 1]. Снова [1,1].", RESULTS)

    assert answer.text == "Нужны Uvicorn и SMTP [1][2]. Снова [2]."
    assert [s.file for s in answer.sources] == ["alerting.md", "backup.md"]


def test_parse_answer_drops_markers_outside_the_fragments():
    answer = parse_answer("Это выдумано [7]. А это нет [0] [2].", RESULTS)

    assert answer.text == "Это выдумано. А это нет [1]."
    assert [s.file for s in answer.sources] == ["api.md"]


def test_parse_answer_without_markers_has_no_sources():
    answer = parse_answer("В найденной документации нет ответа на этот вопрос.", RESULTS)

    assert answer.sources == []
    assert answer.text == "В найденной документации нет ответа на этот вопрос."


def fake_urlopen(response: list[dict] | Exception, captured: dict):
    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        if isinstance(response, Exception):
            raise response
        # Ollama streams one JSON object per line
        lines = [json.dumps(event) for event in response]
        return io.BytesIO(("\n".join(lines) + "\n").encode("utf-8"))

    return urlopen


def test_answer_question_sends_chat_request_and_parses_reply(monkeypatch):
    captured: dict = {}
    reply = [
        {"message": {"role": "assistant", "content": "Используйте "}, "done": False},
        {"message": {"role": "assistant", "content": "Uvicorn [1]."}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    monkeypatch.setattr(ollama_chat.urllib.request, "urlopen", fake_urlopen(reply, captured))

    answer = answer_question("Как восстановить?", FakeStore(RESULTS), top_k=2, model="qwen2.5:3b")

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["model"] == "qwen2.5:3b"
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["options"]["num_ctx"] == CONTEXT_TOKENS
    assert answer.text == "Используйте Uvicorn [1]."
    assert answer.retrieved == 2


def test_answer_question_reports_server_not_running(monkeypatch):
    refused = urllib.error.URLError(ConnectionRefusedError())
    monkeypatch.setattr(ollama_chat.urllib.request, "urlopen", fake_urlopen(refused, {}))

    with pytest.raises(OllamaError, match="not running"):
        answer_question("q", FakeStore(RESULTS))


def test_answer_question_reports_model_not_pulled(monkeypatch):
    missing = urllib.error.HTTPError(
        "http://localhost:11434/api/chat", 404, "Not Found", {},
        io.BytesIO(b'{"error":"model \\"qwen2.5:7b\\" not found, try pulling it first"}'),
    )
    monkeypatch.setattr(ollama_chat.urllib.request, "urlopen", fake_urlopen(missing, {}))

    with pytest.raises(OllamaError, match="ollama pull qwen2.5:7b"):
        answer_question("q", FakeStore(RESULTS))


def test_stream_reply_reports_errors_sent_mid_stream(monkeypatch):
    reply = [{"message": {"content": "Начало"}, "done": False}, {"error": "model runner crashed"}]
    monkeypatch.setattr(ollama_chat.urllib.request, "urlopen", fake_urlopen(reply, {}))

    pieces = ollama_chat.stream_reply(RESULTS, "q")

    assert next(pieces) == "Начало"
    with pytest.raises(OllamaError, match="runner crashed"):
        next(pieces)


def test_stream_reply_stops_when_the_model_switches_to_chinese(monkeypatch):
    reply = [
        {"message": {"content": "Чтобы разрешить запросы,"}, "done": False},
        {"message": {"content": "工作的人员正在加班"}, "done": False},
        {"message": {"content": " и так далее"}, "done": False},
    ]
    monkeypatch.setattr(ollama_chat.urllib.request, "urlopen", fake_urlopen(reply, {}))

    pieces = ollama_chat.stream_reply(RESULTS, "q")

    assert next(pieces) == "Чтобы разрешить запросы,"
    with pytest.raises(ollama_chat.ModelDerailedError, match="другом языке"):
        next(pieces)
