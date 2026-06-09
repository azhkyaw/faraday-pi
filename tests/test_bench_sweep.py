from faraday.bench.config import Cell
from faraday.bench.sweep import append_row, pending, read_completed


def _row(size, quant, status="ok"):
    return {
        "size": size, "quant": quant, "status": status, "disk_bytes": "1",
        "peak_rss_bytes": "2", "prefill_tps": "3", "decode_tps": "4",
        "perplexity": "5", "notes": "",
    }


def test_read_completed_empty_when_no_file(tmp_path):
    assert read_completed(tmp_path / "nope.csv") == set()


def test_append_then_read_completed_roundtrip(tmp_path):
    csv_path = tmp_path / "sweep.csv"
    append_row(csv_path, _row("0.5B", "Q8_0"))
    append_row(csv_path, _row("1.5B", "Q4_K_M"))
    assert read_completed(csv_path) == {("0.5B", "Q8_0"), ("1.5B", "Q4_K_M")}


def test_append_writes_header_once(tmp_path):
    csv_path = tmp_path / "sweep.csv"
    append_row(csv_path, _row("0.5B", "Q8_0"))
    append_row(csv_path, _row("0.5B", "Q6_K"))
    header_lines = [ln for ln in csv_path.read_text().splitlines() if ln.startswith("size,")]
    assert len(header_lines) == 1


def test_pending_excludes_completed_cells():
    universe = [Cell("0.5B", "Q8_0"), Cell("0.5B", "Q6_K"), Cell("1.5B", "Q8_0")]
    todo = pending(universe, {("0.5B", "Q8_0")})
    assert Cell("0.5B", "Q8_0") not in todo
    assert todo == [Cell("0.5B", "Q6_K"), Cell("1.5B", "Q8_0")]
