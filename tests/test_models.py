from faraday.models import Chunk, RetrievedChunk


def test_retrieved_chunk_exposes_source():
    c = Chunk(doc_id="doc1", ord=0, text="hello", source="a.txt")
    rc = RetrievedChunk(chunk=c, score=0.9)
    assert rc.chunk.source == "a.txt"
    assert rc.score == 0.9
