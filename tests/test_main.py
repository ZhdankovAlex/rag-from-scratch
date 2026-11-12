import io
import time

import pytest

from rag_system import ollama_chat
from rag_system.answer import Answer, Source
from rag_system.cli import ask_and_print, waiting

CITED = Answer(text="Use Uvicorn [1].", sources=[Source(1, "backup.md", "Backup > Restore")], retrieved=5)
UNCITED = Answer(text="Run CREATE PUBLICATION.", sources=[], retrieved=5)


@pytest.fixture
def ollama(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Record the Ollama call instead of hitting a server."""
    calls: dict = {}

    def fake_answer(question, store, top_k, model, base_url):
        calls.update(question=question, top_k=top_k, model=model, base_url=base_url)
        return calls.get("answer", CITED)

    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("OLLAMA_URL", "http://gpu-box:11434/")
    monkeypatch.setattr(ollama_chat, "answer_question", fake_answer)
    return calls


def test_ollama_is_called_with_model_and_url_from_env(ollama, capsys):
    assert ask_and_print("Как восстановить?", store=None, top_k=3)

    assert ollama == {
        "question": "Как восстановить?", "top_k": 3,
        "model": "qwen2.5:3b", "base_url": "http://gpu-box:11434",
    }
    out = capsys.readouterr()
    assert "Use Uvicorn [1]." in out.out
    assert "[1] backup.md — Backup > Restore" in out.out
    assert "cites no sources" not in out.err


def test_uncited_answer_gets_a_warning(ollama, capsys):
    ollama["answer"] = UNCITED

    assert ask_and_print("q", store=None, top_k=5)

    assert "cites no sources" in capsys.readouterr().err


def test_ollama_error_is_reported_without_traceback(monkeypatch, capsys):
    def failing(*args, **kwargs):
        raise ollama_chat.OllamaError("Ollama is not running at http://localhost:11434.")

    monkeypatch.setattr(ollama_chat, "answer_question", failing)

    assert not ask_and_print("q", store=None, top_k=5)
    assert "not running" in capsys.readouterr().err


class FakeStream(io.StringIO):
    def __init__(self, tty: bool):
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


def test_waiting_writes_nothing_when_not_a_terminal():
    stream = FakeStream(tty=False)

    with waiting("Asking...", stream):
        pass

    assert stream.getvalue() == ""


def test_waiting_shows_elapsed_seconds_and_erases_the_line():
    stream = FakeStream(tty=True)

    with waiting("Asking...", stream):
        time.sleep(0.7)  # long enough for one tick

    output = stream.getvalue()
    assert "\rAsking... 1s" in output or "\rAsking... 0s" in output
    assert output.endswith("\r" + " " * (len("Asking...") + 8) + "\r")


def test_waiting_erases_the_line_when_the_block_raises():
    stream = FakeStream(tty=True)

    with pytest.raises(RuntimeError), waiting("Asking...", stream):
        raise RuntimeError("API error")

    assert stream.getvalue().endswith("\r")
