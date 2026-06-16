from faraday.grammar import build_citation_grammar
from faraday.models import Chunk, RetrievedChunk
from faraday.rag import RagEngine


class FakeRetriever:
    def search(self, query, k=4):
        mk = lambda i: RetrievedChunk(  # noqa: E731
            chunk=Chunk(doc_id="d", ord=i, text=f"t{i}", source="s.txt"), score=0.9)
        return [mk(0), mk(1)]


def test_engine_builds_grammar_from_n_sources(fake_llm):
    eng = RagEngine(FakeRetriever(), fake_llm, top_k=2,
                    grammar_builder=build_citation_grammar)
    eng.answer("q")
    assert fake_llm.last_grammar is not None
    assert '"[2]"' in fake_llm.last_grammar and '"[3]"' not in fake_llm.last_grammar


def test_engine_passes_none_without_builder(fake_llm):
    eng = RagEngine(FakeRetriever(), fake_llm, top_k=2)
    eng.answer("q")
    assert fake_llm.last_grammar is None


def test_stream_also_carries_grammar(fake_llm):
    eng = RagEngine(FakeRetriever(), fake_llm, top_k=2,
                    grammar_builder=build_citation_grammar)
    list(eng.answer_stream("q"))
    assert fake_llm.last_grammar is not None
