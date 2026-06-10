from faraday.eval.config import AblationConfig
from faraday.eval.dataset import EvalItem
from faraday.eval.runner import append_record, done_keys, record_from_answer, run_config
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


class _FakeEngine:
    """Stands in for RagEngine: returns a canned Answer, records the question."""
    def __init__(self):
        self.asked = []

    def answer(self, query):
        self.asked.append(query)
        rc = RetrievedChunk(chunk=Chunk(doc_id="d", ord=0, text="x", source="moon.txt"),
                            score=0.9)
        return Answer(text="ans [1].", sources=[rc], cited_indices=[1], invalid_citations=[])


def _items():
    return [EvalItem("q1", "Q1?", True, "moon.txt", (0, 10), "a"),
            EvalItem("q2", "Q2?", True, "moon.txt", (0, 10), "b")]


def test_run_config_records_each_question(tmp_path):
    cfg = AblationConfig(top_k=4, chunk_size=1200, chunk_overlap=200)
    raw = tmp_path / "raw.jsonl"
    eng = _FakeEngine()
    n = run_config(cfg, eng, _items(), raw)
    assert n == 2 and eng.asked == ["Q1?", "Q2?"]
    assert done_keys(raw) == {(cfg.slug, "q1"), (cfg.slug, "q2")}


def test_run_config_is_resumable(tmp_path):
    cfg = AblationConfig(top_k=4, chunk_size=1200, chunk_overlap=200)
    raw = tmp_path / "raw.jsonl"
    append_record(raw, record_from_answer(cfg, "q1", _FakeEngine().answer("Q1?")))
    eng = _FakeEngine()
    n = run_config(cfg, eng, _items(), raw)
    assert n == 1 and eng.asked == ["Q2?"]   # q1 skipped (already done)
