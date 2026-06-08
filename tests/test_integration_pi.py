"""Runs only on the Pi with both llama-servers up: pytest -m integration"""
import json as _json
import pytest
from faraday.config import Settings
from faraday.embedder import HttpEmbedder
from faraday.index_store import SqliteVecStore
from faraday.ingest import ingest
from faraday.llm_client import HttpLLMClient
from faraday.rag import RagEngine
from faraday.retriever import Retriever


@pytest.mark.integration
def test_end_to_end_offline_answer(tmp_path):
    s = Settings()
    store = SqliteVecStore(str(tmp_path / "e2e.sqlite"), dim=s.embed_dim)
    stats = ingest("examples/corpus", store=store, embedder=HttpEmbedder(s))
    assert stats.documents >= 1

    engine = RagEngine(Retriever(HttpEmbedder(s), store), HttpLLMClient(s), top_k=s.top_k)
    ans = engine.answer("How much RAM can a Raspberry Pi 4 have?")
    print("\nANSWER:", ans.text)
    assert "8gb" in ans.text.lower() or "8 gb" in ans.text.lower()
    assert ans.sources                      # retrieved something
    assert ans.invalid_citations == []      # no hallucinated sources
    store.close()


def _collect_tokens(sse_body: str) -> str:
    """Join the text of all `event: token` frames in an SSE body."""
    out = []
    lines = sse_body.splitlines()
    for i, line in enumerate(lines):
        if line == "event: token" and i + 1 < len(lines) and lines[i + 1].startswith("data:"):
            out.append(_json.loads(lines[i + 1][len("data:"):].strip())["text"])
    return "".join(out)


@pytest.mark.integration
def test_chat_endpoint_streams_grounded_answer(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from faraday import server

    db = str(tmp_path / "srv.sqlite")
    monkeypatch.setenv("FARADAY_DB", db)
    s = Settings.from_env()
    store = SqliteVecStore(db, dim=s.embed_dim)
    ingest("examples/corpus", store=store, embedder=HttpEmbedder(s))
    store.close()

    client = TestClient(server.app)
    body = client.get("/chat", params={"q": "How much RAM can a Raspberry Pi 4 have?"}).text
    answer = _collect_tokens(body)
    print("\nSTREAMED ANSWER:", answer)
    assert "event: done" in body
    assert "8gb" in answer.lower() or "8 gb" in answer.lower()


@pytest.mark.integration
def test_metrics_endpoint_after_chat(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from faraday import server

    db = str(tmp_path / "m.sqlite")
    monkeypatch.setenv("FARADAY_DB", db)
    s = Settings.from_env()
    store = SqliteVecStore(db, dim=s.embed_dim)
    ingest("examples/corpus", store=store, embedder=HttpEmbedder(s))
    store.close()

    client = TestClient(server.app)
    client.get("/chat", params={"q": "How much RAM can a Raspberry Pi 4 have?"}).text
    body = client.get("/metrics").text
    print("\nMETRICS SAMPLE:\n" + "\n".join(
        l for l in body.splitlines() if l.startswith("faraday_") and "pi_" in l))
    assert "faraday_requests_total" in body
    assert "faraday_pi_temp_celsius" in body          # host collector live on the Pi
    assert "faraday_ttft_seconds_count" in body
