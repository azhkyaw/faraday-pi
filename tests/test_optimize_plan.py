from faraday.bench.optimize_config import cells
from faraday.bench.optimize import stack_winners


def _row(component, label, decode, throttled="throttled=0x0"):
    return {"component": component, "label": label, "prefill_tps": "5",
            "decode_tps": str(decode), "peak_rss_bytes": "1", "accept_rate": "",
            "throttled": throttled, "notes": "ok"}


def test_stack_winners_picks_best_per_lever_over_baseline():
    cs = cells()
    rows = [
        _row("baseline", "baseline", 4.50),
        _row("governor", "governor=performance", 5.20),   # win
        _row("threads", "threads=2", 4.10),               # lose
        _row("threads", "threads=3", 4.90),               # win (best thread)
        _row("batch", "ubatch=1024", 4.40),               # lose
        _row("kvquant", "kv=q8_0", 4.80),                 # win
        _row("flashattn", "flash_attn", 4.55),            # win
    ]
    best = stack_winners(cs, rows)
    assert best.component == "stacked_best"
    assert best.governor == "performance"          # governor winner
    assert "-t" in best.flags and "3" in best.flags  # best thread setting only
    assert "-ctk" in best.flags                    # kvquant winner
    assert "-fa" in best.flags                      # flashattn (and kvquant) winner
    assert "-ub" not in best.flags                 # batch lost


def test_stack_winners_ignores_throttled_cells():
    cs = cells()
    rows = [
        _row("baseline", "baseline", 4.50),
        _row("governor", "governor=performance", 9.99, throttled="throttled=0x50000"),  # fast but DIRTY
    ]
    best = stack_winners(cs, rows)
    assert best.governor == "ondemand"  # throttled winner rejected
