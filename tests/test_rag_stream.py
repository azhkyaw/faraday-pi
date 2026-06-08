from faraday.rag import RagEngine
from faraday.retriever import Retriever
from faraday.index_store import SqliteVecStore
from faraday.models import Chunk
from faraday.events import SourcesEvent, TokenEvent, DoneEvent


def _store(tmp_path, embedder):
    store = SqliteVecStore(str(tmp_path / "s.sqlite"), dim=embedder.dim)
    chunks = [Chunk(doc_id="d", ord=0, text="The Pi 4 has 4GB RAM.", source="pi.txt"),
              Chunk(doc_id="d", ord=1, text="It uses an ARM CPU.", source="pi.txt")]
    store.add_chunks(chunks, embedder.embed([c.text for c in chunks]))
    return store


def test_answer_stream_emits_sources_tokens_then_done(tmp_path, fake_embedder, make_llm):
    store = _store(tmp_path, fake_embedder)
    engine = RagEngine(Retriever(fake_embedder, store), make_llm("Answer [1]."), top_k=2)
    events = list(engine.answer_stream("how much ram?"))
    store.close()

    assert isinstance(events[0], SourcesEvent)
    assert len(events[0].sources) == 2
    tokens = [e.text for e in events if isinstance(e, TokenEvent)]
    assert "".join(tokens) == "Answer [1]."          # exact reconstruction
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].cited_indices == [1]
    assert events[-1].invalid_citations == []


def test_answer_stream_done_flags_hallucinated_citation(tmp_path, fake_embedder, make_llm):
    store = _store(tmp_path, fake_embedder)
    engine = RagEngine(Retriever(fake_embedder, store), make_llm("Nope [9]."), top_k=2)
    events = list(engine.answer_stream("q"))
    store.close()
    assert events[-1].invalid_citations == [9]
