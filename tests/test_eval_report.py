from faraday.eval.dataset import EvalItem
from faraday.eval.judge import JudgeVerdict
from faraday.eval.report import (
    _ablation_order,
    abstention_cross_check,
    judge_rows,
    load_or_classify_abstentions,
    load_or_score,
    make_scorecard,
    render_ablation,
)


class FakeJudge:
    def __init__(self):
        self.seen = []

    def score(self, **kwargs):
        self.seen.append(kwargs)
        return JudgeVerdict(faithfulness=5, correctness=4, rationale="ok")

    def classify_abstention(self, *, question, answer):
        return "no idea" in answer.lower()


def _item(qid, answerable):
    return EvalItem(qid, "?", answerable, "moon.txt" if answerable else "",
                    (100, 260) if answerable else (0, 0), "ref")


def _rows(slug):
    return [
        {"slug": slug, "qid": "q1",
         "retrieved": [{"source": "moon.txt", "ord": 0,
                        "text": "Apollo 11 landed on 20 July 1969."}],
         "answer": "July 1969 [1].", "cited": [1], "invalid": [], "abstained": False},
    ]


def test_judge_rows_scores_answered_questions():
    items = {"q1": _item("q1", True)}
    scores = judge_rows(_rows("k2_c200_o0"), items, FakeJudge())
    assert scores["q1"].faithfulness == 5 and scores["q1"].correctness == 4


def test_judge_sees_chunk_text_in_context():
    items = {"q1": _item("q1", True)}
    judge = FakeJudge()
    judge_rows(_rows("k2_c200_o0"), items, judge)
    assert "Apollo 11 landed on 20 July 1969." in judge.seen[0]["context"]


def test_make_scorecard_has_a_row_per_config():
    per_config = {
        "k2_c200_o0": {"recall_at_k": 1.0, "mrr": 1.0, "citation_validity": 1.0,
                       "abstention_accuracy": 1.0, "faithfulness": 5.0, "correctness": 4.0},
    }
    md = make_scorecard(per_config)
    assert "k2_c200_o0" in md and "recall@k" in md.lower()


def test_render_ablation_writes_png(tmp_path):
    per_config = {
        "k2_c600_o100": {"recall_at_k": 0.8, "faithfulness": 4.0},
        "k4_c600_o100": {"recall_at_k": 0.9, "faithfulness": 4.5},
    }
    out = tmp_path / "ablations.png"
    render_ablation(per_config, out)
    assert out.exists() and out.stat().st_size > 0


def test_ablation_order_sorts_by_topk_then_chunksize():
    # Alphabetical sort gives c1200, c2400, c600 within each top_k; we want
    # ascending chunk size (c600, c1200, c2400) so the plot/scorecard read cleanly.
    per_config = {s: {} for s in [
        "k8_c2400_o400", "k2_c600_o100", "k4_c1200_o200", "k2_c2400_o400",
        "k8_c600_o100", "k4_c600_o100", "k2_c1200_o200", "k8_c1200_o200",
        "k4_c2400_o400",
    ]}
    assert _ablation_order(per_config) == [
        "k2_c600_o100", "k2_c1200_o200", "k2_c2400_o400",
        "k4_c600_o100", "k4_c1200_o200", "k4_c2400_o400",
        "k8_c600_o100", "k8_c1200_o200", "k8_c2400_o400",
    ]


class BoomJudge:
    def score(self, **kwargs):
        raise AssertionError("should not be called when cache exists")

    def classify_abstention(self, **kwargs):
        raise AssertionError("should not be called when cache exists")


def test_load_or_score_writes_then_reads_cache(tmp_path):
    items = {"q1": _item("q1", True)}
    rows = _rows("k4_c1200_o200")
    cache = tmp_path / "judge_k4.jsonl"
    first = load_or_score(rows, items, FakeJudge(), cache)   # scores + writes cache
    assert first["q1"].faithfulness == 5
    assert cache.exists()
    again = load_or_score(rows, items, BoomJudge(), cache)   # loads cache, no judge call
    assert again["q1"].correctness == 4


def _rows_with_missed_abstention(slug):
    """q2's answer abstains with phrasing the heuristic doesn't know ('no idea')."""
    return _rows(slug) + [
        {"slug": slug, "qid": "q2", "retrieved": [],
         "answer": "I have no idea, sorry.", "cited": [], "invalid": [],
         "abstained": False},
    ]


def test_load_or_classify_abstentions_writes_then_reads_cache(tmp_path):
    items = {"q1": _item("q1", True), "q2": _item("q2", False)}
    rows = _rows_with_missed_abstention("k4_c1200_o200")
    cache = tmp_path / "abstention_k4.jsonl"
    first = load_or_classify_abstentions(rows, items, FakeJudge(), cache)
    assert first == {"q1": False, "q2": True}    # judge catches the missed phrasing
    assert cache.exists()
    again = load_or_classify_abstentions(rows, items, BoomJudge(), cache)
    assert again == {"q1": False, "q2": True}    # cache hit, no judge call


def test_abstention_cross_check_scores_judge_and_flags_disagreements():
    items = {"q1": _item("q1", True), "q2": _item("q2", False)}
    rows = [
        {"qid": "q1", "abstained": False},
        {"qid": "q2", "abstained": False},        # heuristic wrong on q2
    ]
    judged = {"q1": False, "q2": True}            # judge right on both
    out = abstention_cross_check(rows, items, judged)
    assert out["abstention_judged"] == 1.0
    assert out["disagreements"] == ["q2"]
