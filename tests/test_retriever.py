from faraday.index_store import SqliteVecStore
from faraday.retriever import Retriever
from faraday.models import Chunk


def test_retriever_embeds_query_then_searches(tmp_path, fake_embedder):
    store = SqliteVecStore(str(tmp_path / "r.sqlite"), dim=fake_embedder.dim)
    chunks = [Chunk(doc_id="d", ord=0, text="alpha beta", source="s.txt"),
              Chunk(doc_id="d", ord=1, text="gamma delta", source="s.txt")]
    store.add_chunks(chunks, fake_embedder.embed([c.text for c in chunks]))
    retriever = Retriever(embedder=fake_embedder, store=store)
    results = retriever.search("alpha", k=1)
    assert results[0].chunk.text == "alpha beta"
    store.close()
