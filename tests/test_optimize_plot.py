from faraday.bench.optimize_plot import (
    lever_gains,
    make_leaderboard,
    render_context_curve,
    render_waterfall,
)


def _row(component, label, decode, prefill="5"):
    return {"component": component, "label": label, "prefill_tps": prefill,
            "decode_tps": str(decode), "peak_rss_bytes": "1", "accept_rate": "",
            "throttled": "throttled=0x0", "notes": "ok"}


def test_lever_gains_are_percent_over_baseline():
    rows = [_row("baseline", "baseline", 4.0),
            _row("governor", "governor=performance", 5.0),
            _row("threads", "threads=3", 4.4)]
    gains = lever_gains(rows)
    assert gains["governor=performance"] == 25.0   # +25%
    assert gains["threads=3"] == 10.0              # +10%


def test_make_leaderboard_sorts_by_decode_desc():
    rows = [_row("baseline", "baseline", 4.0),
            _row("stacked_best", "stacked_best", 6.0),
            _row("ollama", "ollama-default", 3.5)]
    md = make_leaderboard(rows)
    assert md.index("stacked_best") < md.index("baseline") < md.index("ollama-default")
    assert "decode" in md.lower()


def test_render_waterfall_and_context_write_pngs(tmp_path):
    rows = [_row("baseline", "baseline", 4.0),
            _row("governor", "governor=performance", 5.0),
            _row("stacked_best", "stacked_best", 6.0),
            _row("context", "ctx=512", 4.5, prefill="9.0"),
            _row("context", "ctx=2048", 4.0, prefill="7.0")]
    w = tmp_path / "waterfall.png"
    render_waterfall(rows, w)
    assert w.exists() and w.stat().st_size > 0
    c = tmp_path / "context.png"
    render_context_curve(rows, c)
    assert c.exists() and c.stat().st_size > 0
