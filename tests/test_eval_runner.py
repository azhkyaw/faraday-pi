import json

from faraday.eval.config import AblationConfig
from faraday.eval.dataset import EvalItem
from faraday.eval.runner import append_record, done_keys, record_from_answer, run_config
from faraday.models import Answer, Chunk, RetrievedChunk


def _answer():
    rc = RetrievedChunk(chunk=Chunk(doc_id="d", ord=3, text="On 20 July 1969 the LM landed.",
                                    source="moon.txt"), score=0.9)
    return Answer(text="July 1969 [1].", sources=[rc], cited_indices=[1], invalid_citations=[])


def test_record_from_answer_shape():
    cfg = AblationConfig(top_k=4, chunk_size=1200, chunk_overlap=200)
    rec = record_from_answer(cfg, "q1", _answer())
    assert rec["qid"] == "q1"
    assert rec["config"] == {"top_k": 4, "chunk_size": 1200, "chunk_overlap": 200}
    assert rec["retrieved"] == [{"source": "moon.txt", "ord": 3,
                                 "text": "On 20 July 1969 the LM landed."}]
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


class _FakeRetriever:
    def search(self, query, k):
        rc = RetrievedChunk(chunk=Chunk(doc_id="d", ord=0, text="ctx", source="moon.txt"),
                            score=0.9)
        return [rc]


def test_run_default_llm_uses_batch_timeout(tmp_path, monkeypatch):
    """The eval is a batch job: deep-context prefill measured 6.75 tok/s on the
    Pi, so a k8_c2400 question takes ~12-14 min before llama-server sends a
    byte. The runner's default LLM client needs a read timeout sized for that,
    not the interactive app's 120 s."""
    import faraday.llm_client as llm_client
    from faraday.eval import config, runner

    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps({
        "id": "q1", "question": "Q1?", "answerable": True,
        "relevant_doc": "moon.txt", "relevant_span": [0, 10], "reference_answer": "a",
    }) + "\n")
    monkeypatch.setattr(config, "GOLDEN_PATH", golden)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")

    captured = {}

    class FakeHttpLLM:
        def __init__(self, settings=None, timeout=120.0):
            captured["timeout"] = timeout

        def complete(self, messages, max_tokens=512):
            return "ans [1]."

    monkeypatch.setattr(llm_client, "HttpLLMClient", FakeHttpLLM)
    runner.run(retriever_factory=lambda size, overlap, settings: _FakeRetriever())
    assert captured["timeout"] == 1800.0


def test_run_ingests_once_per_chunk_size(tmp_path, monkeypatch, fake_llm):
    """The grid shares one store per chunk-size: 3 ingests for 9 configs, not 9."""
    from faraday.eval import config, runner

    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps({
        "id": "q1", "question": "Q1?", "answerable": True,
        "relevant_doc": "moon.txt", "relevant_span": [0, 10], "reference_answer": "a",
    }) + "\n")
    monkeypatch.setattr(config, "GOLDEN_PATH", golden)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")

    ingests: list[int] = []

    def factory(chunk_size, overlap, settings):
        ingests.append(chunk_size)
        return _FakeRetriever()

    runner.run(retriever_factory=factory, llm=fake_llm)

    assert ingests == sorted(config.CHUNK_SIZES)                       # one per size
    assert len(list((tmp_path / "raw").glob("*.jsonl"))) == len(config.configs())
