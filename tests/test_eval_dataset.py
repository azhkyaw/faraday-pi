from faraday.eval.dataset import EvalItem, load_golden

from eval_samples import GOLDEN_JSONL


def test_load_golden_parses_items(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(GOLDEN_JSONL)
    items = load_golden(path)
    assert [i.id for i in items] == ["q1", "q2"]
    a = items[0]
    assert isinstance(a, EvalItem)
    assert a.answerable is True
    assert a.relevant_doc == "moon.txt"
    assert a.relevant_span == (100, 260)
    assert items[1].answerable is False


def test_load_golden_skips_blank_lines(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text("\n" + GOLDEN_JSONL + "\n")
    assert len(load_golden(path)) == 2
