from faraday.rag import RagEngine
from faraday.retriever import Retriever
from faraday.index_store import SqliteVecStore
from faraday.models import Chunk


def _store(tmp_path, embedder):
    store = SqliteVecStore(str(tmp_path / "rag.sqlite"), dim=embedder.dim)
    chunks = [Chunk(doc_id="d", ord=0, text="The Pi 4 has 4GB RAM.", source="pi.txt"),
              Chunk(doc_id="d", ord=1, text="It uses an ARM CPU.", source="pi.txt")]
    store.add_chunks(chunks, embedder.embed([c.text for c in chunks]))
    return store


def test_answer_assembles_sources_and_valid_citations(tmp_path, fake_embedder, fake_llm):
    store = _store(tmp_path, fake_embedder)
    engine = RagEngine(Retriever(fake_embedder, store), fake_llm, top_k=2)
    ans = engine.answer("How much RAM?")
    assert ans.text == "Answer [1]."
    assert len(ans.sources) == 2
    assert ans.cited_indices == [1]
    assert ans.invalid_citations == []
    store.close()


def test_answer_flags_hallucinated_citation(tmp_path, fake_embedder, make_llm):
    store = _store(tmp_path, fake_embedder)
    engine = RagEngine(Retriever(fake_embedder, store), make_llm("Nope [9]."), top_k=2)
    ans = engine.answer("q")
    assert ans.invalid_citations == [9]
    store.close()
