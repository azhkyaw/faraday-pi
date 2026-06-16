from faraday.bench.optimize_config import CSV_COLUMNS, LeverCell, cells


def test_csv_schema_is_stable():
    assert CSV_COLUMNS == (
        "component", "label", "prefill_tps", "decode_tps", "peak_rss_bytes",
        "accept_rate", "throttled", "notes",
    )


def test_cells_has_baseline_first_and_expected_components():
    cs = cells()
    assert cs[0] == LeverCell("baseline", "baseline", "ondemand",
                              ("-p", "512", "-n", "128"), "llama_bench")
    comps = {c.component for c in cs}
    # every designed component present
    assert comps == {"baseline", "governor", "threads", "batch",
                     "kvquant", "flashattn", "context", "speculative", "ollama"}


def test_lever_cells_carry_the_right_flags():
    by_label = {c.label: c for c in cells()}
    assert by_label["governor=performance"].governor == "performance"
    assert by_label["threads=3"].flags == ("-p", "512", "-n", "128", "-t", "3")
    assert by_label["flash_attn"].flags == ("-p", "512", "-n", "128", "-fa", "on")
    # V-cache quant requires flash-attn in llama.cpp, so kvquant implies -fa;
    # this build's llama-bench needs an explicit -fa on|off|auto (bare -fa is rejected):
    assert by_label["kv=q8_0"].flags == (
        "-p", "512", "-n", "128", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on")
    assert by_label["ctx=2048"].flags == ("-p", "2048", "-n", "128")
    assert by_label["speculative"].kind == "speculative"
    assert by_label["ollama-default"].kind == "ollama"
