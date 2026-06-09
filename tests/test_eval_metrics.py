from faraday.eval.dataset import EvalItem
from faraday.eval.metrics import (
    aggregate,
    chunk_is_relevant,
    is_abstention,
)


def _item(qid, answerable, doc="moon.txt", span=(100, 260)):
    return EvalItem(qid, "?", answerable, doc if answerable else "",
                    span if answerable else (0, 0), "ref")


def test_chunk_span_overlap_uses_ord_and_config():
    item = _item("q1", True, "moon.txt", (100, 260))
    # size=200, overlap=0 -> step=200. ord=0 covers [0,200): overlaps [100,260). ord=2 -> [400,600): no.
    assert chunk_is_relevant("moon.txt", 0, item, size=200, overlap=0) is True
    assert chunk_is_relevant("moon.txt", 2, item, size=200, overlap=0) is False
    assert chunk_is_relevant("other.txt", 0, item, size=200, overlap=0) is False  # wrong doc


def test_is_abstention_detects_dont_know():
    assert is_abstention("I don't know based on the sources.") is True
    assert is_abstention("Apollo 11 landed in July 1969 [1].") is False


def test_aggregate_computes_recall_mrr_citation_abstention():
    items = {"q1": _item("q1", True), "q2": _item("q2", False)}
    # q1 (answerable): top-2 retrieved; ord=0 is relevant (rank 1). Cites [1] valid.
    # q2 (unanswerable): correctly abstains.
    rows = [
        {"qid": "q1", "retrieved": [{"source": "moon.txt", "ord": 0},
                                    {"source": "moon.txt", "ord": 5}],
         "answer": "July 1969 [1].", "cited": [1], "invalid": [2], "abstained": False},
        {"qid": "q2", "retrieved": [{"source": "x.txt", "ord": 0}],
         "answer": "I don't know.", "cited": [], "invalid": [], "abstained": True},
    ]
    m = aggregate(rows, items, size=200, overlap=0)
    assert m["recall_at_k"] == 1.0          # q1 found a relevant chunk
    assert m["mrr"] == 1.0                   # relevant at rank 1
    assert m["citation_validity"] == 0.5     # 1 valid / (1 valid + 1 invalid)
    assert m["abstention_accuracy"] == 1.0   # q1 answered (correct), q2 abstained (correct)
    assert m["n_answerable"] == 1
