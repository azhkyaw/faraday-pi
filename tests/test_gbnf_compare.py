from faraday.eval.dataset import EvalItem
from faraday.eval.gbnf_compare import compare


def _row(qid, cited, invalid):
    return {"qid": qid, "retrieved": [{"source": "s.txt", "ord": 0}], "answer": "a [1].",
            "cited": cited, "invalid": invalid, "abstained": False}


def test_compare_reports_citation_validity_both_ways():
    items = {"q1": EvalItem("q1", "?", True, "s.txt", (0, 10), "r")}
    before = [_row("q1", [1], [3])]   # 50% valid
    after = [_row("q1", [1], [])]     # 100% valid
    out = compare(before, after, items, size=1200, overlap=200)
    assert out["before"]["citation_validity"] == 0.5
    assert out["after"]["citation_validity"] == 1.0
