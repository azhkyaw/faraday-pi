from faraday.eval.dataset import EvalItem
from faraday.eval.judge import JudgeVerdict
from faraday.eval.report import judge_rows, make_scorecard, render_ablation


class FakeJudge:
    def score(self, **kwargs):
        return JudgeVerdict(faithfulness=5, correctness=4, rationale="ok")


def _item(qid, answerable):
    return EvalItem(qid, "?", answerable, "moon.txt" if answerable else "",
                    (100, 260) if answerable else (0, 0), "ref")


def _rows(slug):
    return [
        {"slug": slug, "qid": "q1", "retrieved": [{"source": "moon.txt", "ord": 0}],
         "answer": "July 1969 [1].", "cited": [1], "invalid": [], "abstained": False},
    ]


def test_judge_rows_scores_answered_questions():
    items = {"q1": _item("q1", True)}
    scores = judge_rows(_rows("k2_c200_o0"), items, FakeJudge())
    assert scores["q1"].faithfulness == 5 and scores["q1"].correctness == 4


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
