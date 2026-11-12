import pytest
from fastapi.testclient import TestClient

from rag_system import ollama_chat
from rag_system.answer import Answer, Source
from rag_system.chunker import Chunk
from web import app as web_app


@pytest.fixture
def client(store, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The web app over a temporary store with fake embeddings and three chunks."""
    store.sync_chunks([
        Chunk(id="cors.md::CORS::0", source="cors.md", title="CORS > Шаги", text="cors origins frontend backend"),
        Chunk(id="body.md::Body::0", source="body.md", title="Тело запроса", text="pydantic model request body"),
        Chunk(id="docker.md::Image::0", source="docker.md", title="Docker > Образ", text="docker image build"),
    ])
    monkeypatch.setattr(web_app, "get_store", lambda: store)
    return TestClient(web_app.app)


def test_pages_render_and_unknown_page_is_404(client):
    for page in ("ask", "search", "quality", "documents"):
        response = client.get(f"/{page}")
        assert response.status_code == 200
        assert 'class="on"' in response.text  # active item in the side panel

    assert client.get("/nope").status_code == 404


def test_search_returns_ranked_chunks_with_similarity(client):
    response = client.post("/api/search", json={"query": "docker image", "top_k": 2})

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["rank"] for r in results] == [1, 2]
    assert results[0]["file"] == "docker.md"
    assert results[0]["similarity"] > results[1]["similarity"]


def test_search_rejects_an_empty_query(client):
    assert client.post("/api/search", json={"query": ""}).status_code == 422


def test_ask_returns_answer_with_cited_chunk_ids(client, monkeypatch):
    def fake_answer(question, store, top_k, model, base_url, results):
        assert [r["metadata"]["source"] for r in results][0] == "cors.md"  # retrieved once, passed through
        return Answer(text="Добавьте CORSMiddleware [1].",
                      sources=[Source(1, "cors.md", "CORS > Шаги", chunk_id=results[0]["id"])], retrieved=len(results))

    monkeypatch.setattr(ollama_chat, "answer_question", fake_answer)

    body = client.post("/api/ask", json={"question": "cors frontend", "top_k": 2}).json()

    assert body["answer"] == "Добавьте CORSMiddleware [1]."
    assert body["sources"][0]["chunk_id"] == "cors.md::CORS::0"
    assert body["results"][0]["id"] == "cors.md::CORS::0"
    assert body["unsupported"] is False


def test_ask_reports_ollama_errors_as_503(client, monkeypatch):
    def failing(*args, **kwargs):
        raise ollama_chat.OllamaError("Ollama is not running at http://localhost:11434.")

    monkeypatch.setattr(ollama_chat, "answer_question", failing)

    response = client.post("/api/ask", json={"question": "cors"})

    assert response.status_code == 503
    assert "not running" in response.json()["detail"]


def test_status_reports_index_size_and_unreachable_ollama(client, monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")  # nothing listens on the discard port

    body = client.get("/api/status").json()

    assert body["chunks"] == 3
    assert body["ollama"]["running"] is False
