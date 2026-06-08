from fastapi.testclient import TestClient
from faraday import server
from faraday.rag import RagEngine
from faraday.retriever import Retriever
from faraday.index_store import SqliteVecStore
from faraday.models import Chunk


def _engine(tmp_path, fake_embedder, make_llm, reply):
    store = SqliteVecStore(str(tmp_path / "srv.sqlite"), dim=fake_embedder.dim)
    chunks = [Chunk(doc_id="d", ord=0, text="The Pi 4 has 4GB RAM.", source="pi.txt"),
              Chunk(doc_id="d", ord=1, text="It uses an ARM CPU.", source="pi.txt")]
    store.add_chunks(chunks, fake_embedder.embed([c.text for c in chunks]))
    return RagEngine(Retriever(fake_embedder, store), make_llm(reply), top_k=2), store


def test_healthz_ok():
    client = TestClient(server.app)
    assert client.get("/healthz").status_code == 200


def test_chat_streams_sse_events(tmp_path, monkeypatch, fake_embedder, make_llm):
    engine, store = _engine(tmp_path, fake_embedder, make_llm, "Answer [1].")
    monkeypatch.setattr(server, "make_engine", lambda settings: (engine, store))
    monkeypatch.setattr(server, "_preflight_ok", lambda settings: True)
    client = TestClient(server.app)

    body = client.get("/chat", params={"q": "how much ram?"}).text
    assert "event: sources" in body
    assert "event: token" in body
    assert "event: done" in body
    assert '"cited": [1]' in body          # the done event carries the verified citation


def test_chat_503_when_servers_down(monkeypatch):
    monkeypatch.setattr(server, "_preflight_ok", lambda settings: False)
    client = TestClient(server.app)
    assert client.get("/chat", params={"q": "x"}).status_code == 503


def test_index_page_served():
    client = TestClient(server.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "EventSource" in r.text and "Faraday" in r.text
