from faraday.eval.config import AblationConfig
from faraday.eval.runner import append_record, done_keys, record_from_answer
from faraday.models import Answer, Chunk, RetrievedChunk


def _answer():
    rc = RetrievedChunk(chunk=Chunk(doc_id="d", ord=3, text="...", source="moon.txt"), score=0.9)
    return Answer(text="July 1969 [1].", sources=[rc], cited_indices=[1], invalid_citations=[])


def test_record_from_answer_shape():
    cfg = AblationConfig(top_k=4, chunk_size=1200, chunk_overlap=200)
    rec = record_from_answer(cfg, "q1", _answer())
    assert rec["qid"] == "q1"
    assert rec["config"] == {"top_k": 4, "chunk_size": 1200, "chunk_overlap": 200}
    assert rec["retrieved"] == [{"source": "moon.txt", "ord": 3}]
    assert rec["cited"] == [1] and rec["invalid"] == [] and rec["abstained"] is False


def test_append_and_done_keys_roundtrip(tmp_path):
    cfg = AblationConfig(top_k=2, chunk_size=600, chunk_overlap=100)
    path = tmp_path / "raw.jsonl"
    append_record(path, record_from_answer(cfg, "q1", _answer()))
    append_record(path, record_from_answer(cfg, "q2", _answer()))
    assert done_keys(path) == {(cfg.slug, "q1"), (cfg.slug, "q2")}
    assert done_keys(tmp_path / "missing.jsonl") == set()
