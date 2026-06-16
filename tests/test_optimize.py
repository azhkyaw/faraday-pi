from faraday.bench.optimize_config import CSV_COLUMNS, LeverCell
from faraday.bench.optimize import append_row, build_argv, read_done, run_cell
from faraday.bench.sweep import Completed

BENCH_OUT = (
    "| qwen2 1.5B Q4_K - Medium | 1.04 GiB | 1.54 B | CPU | 4 | pp512 | 9.80 ± 0.01 |\n"
    "| qwen2 1.5B Q4_K - Medium | 1.04 GiB | 1.54 B | CPU | 4 | tg128 | 4.50 ± 0.00 |\n"
)
TIME_V = "\tMaximum resident set size (kbytes): 1100000\n\tExit status: 0\n"
OLLAMA = "prompt eval rate: 21.49 tokens/s\neval rate: 24.23 tokens/s\n"
SPEC = "accept = 65.97%\ndecoded 128 tokens in 30.5 s, speed: 4.20 t/s\n"


def _fake_run(*, bench=(0, BENCH_OUT, TIME_V), ollama=(0, OLLAMA, ""),
              spec=(0, SPEC, ""), throttle="throttled=0x0"):
    calls = []

    def run(argv):
        calls.append(argv)
        if argv[0] == "sudo":
            return Completed(0, "", "")
        if "get_throttled" in argv:
            return Completed(0, throttle, "")
        if "llama-bench" in argv:
            return Completed(*bench)
        if "ollama" in argv:
            return Completed(*ollama)
        if "llama-speculative" in argv:
            return Completed(*spec)
        raise AssertionError(argv)

    run.calls = calls
    return run


def test_build_argv_llama_bench_wraps_in_time():
    cell = LeverCell("threads", "threads=3", "ondemand",
                     ("-p", "512", "-n", "128", "-t", "3"), "llama_bench")
    argv = build_argv(cell, model="m.gguf", draft="d.gguf", ollama_model="q", prompt="hi")
    assert argv[:4] == ["/usr/bin/time", "-v", "llama-bench", "-m"]
    assert "-t" in argv and "3" in argv and "-o" in argv


def test_build_argv_ollama_and_speculative():
    base = dict(model="m.gguf", draft="d.gguf", ollama_model="qwen2.5:1.5b", prompt="hi")
    oll = build_argv(LeverCell("ollama", "ollama-default", None, (), "ollama"), **base)
    assert oll == ["ollama", "run", "--verbose", "qwen2.5:1.5b", "hi"]
    spec = build_argv(LeverCell("speculative", "speculative", "ondemand", (), "speculative"), **base)
    assert spec[:3] == ["/usr/bin/time", "-v", "llama-speculative"]
    assert "-md" in spec and "d.gguf" in spec


def test_run_cell_llama_bench_records_row(tmp_path):
    cell = LeverCell("baseline", "baseline", "ondemand", ("-p", "512", "-n", "128"), "llama_bench")
    run = _fake_run()
    row = run_cell(cell, run, model="m.gguf", draft="d.gguf",
                   ollama_model="q", prompt="hi", raw_dir=tmp_path)
    assert row["component"] == "baseline"
    assert row["prefill_tps"] == 9.80 and row["decode_tps"] == 4.50
    assert row["peak_rss_bytes"] == 1100000 * 1024
    assert row["throttled"] == "throttled=0x0"
    assert row["notes"] == "ok"


def test_run_cell_sets_governor_then_benches():
    cell = LeverCell("governor", "governor=performance", "performance",
                     ("-p", "512", "-n", "128"), "llama_bench")
    run = _fake_run()
    run_cell(cell, run, model="m.gguf", draft="d", ollama_model="q", prompt="hi", raw_dir=None)
    assert any(a[0] == "sudo" and "performance" in " ".join(a) for a in run.calls)


def test_run_cell_speculative_records_accept(tmp_path):
    cell = LeverCell("speculative", "speculative", "ondemand", (), "speculative")
    row = run_cell(cell, _fake_run(), model="m", draft="d", ollama_model="q",
                   prompt="hi", raw_dir=tmp_path)
    assert row["decode_tps"] == 4.20 and row["accept_rate"] == 65.97


def test_append_and_read_done_roundtrip(tmp_path):
    csv_path = tmp_path / "optimize.csv"
    row = dict.fromkeys(CSV_COLUMNS, "")
    row["component"], row["label"] = "baseline", "baseline"
    append_row(csv_path, row)
    assert read_done(csv_path) == {("baseline", "baseline")}
    assert read_done(tmp_path / "nope.csv") == set()
