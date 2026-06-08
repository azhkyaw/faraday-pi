"""Runs only on the Pi with both llama-servers up: pytest -m integration"""
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
