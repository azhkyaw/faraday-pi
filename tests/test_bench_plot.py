import csv

from faraday.bench.config import CSV_COLUMNS
from faraday.bench.plot import (
    load_rows,
    make_leaderboard,
    pareto_front,
    render_frontier,
)


def test_pareto_front_marks_non_dominated_minimizing_both():
    # (rss, ppl); minimize both. (150,12) is dominated by (100,10).
    pts = [(100, 10), (200, 8), (150, 12), (300, 5)]
    assert pareto_front(pts) == [True, True, False, True]


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(size, quant, status, rss="", ppl="", notes=""):
    return {
        "size": size, "quant": quant, "status": status, "disk_bytes": "1",
        "peak_rss_bytes": rss, "prefill_tps": "1", "decode_tps": "1",
        "perplexity": ppl, "notes": notes,
    }


def test_make_leaderboard_ranks_ok_cells_and_lists_failures(tmp_path):
    rows = [
        _row("0.5B", "Q4_K_M", "ok", rss=str(600 * 1024 * 1024), ppl="9.5"),
        _row("1.5B", "Q4_K_M", "ok", rss=str(1100 * 1024 * 1024), ppl="7.2"),
        _row("3B", "Q8_0", "oom", notes="won't fit"),
    ]
    md = make_leaderboard(rows)
    assert "1.5B-Q4_K_M" in md and "0.5B-Q4_K_M" in md
    assert "7.2000" in md                         # ppl formatted
    assert "★" in md                              # at least one frontier marker
    assert "Did not complete" in md and "3B-Q8_0" in md


def test_render_and_load_roundtrip_writes_a_png(tmp_path):
    csv_path = tmp_path / "sweep.csv"
    _write_csv(csv_path, [
        _row("0.5B", "Q4_K_M", "ok", rss=str(600 * 1024 * 1024), ppl="9.5"),
        _row("1.5B", "Q4_K_M", "ok", rss=str(1100 * 1024 * 1024), ppl="7.2"),
    ])
    rows = load_rows(csv_path)
    assert len(rows) == 2
    out = tmp_path / "frontier.png"
    render_frontier(rows, out)
    assert out.exists() and out.stat().st_size > 0
