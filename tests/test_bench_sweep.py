from bench_samples import LLAMA_BENCH_MD, PERPLEXITY, TIME_V
from faraday.bench import config as bench_config
from faraday.bench.config import Cell
from faraday.bench.sweep import (
    Completed,
    append_row,
    main,
    pending,
    read_completed,
    run_cell,
)


def _row(size, quant, status="ok"):
    return {
        "size": size, "quant": quant, "status": status, "disk_bytes": "1",
        "peak_rss_bytes": "2", "prefill_tps": "3", "decode_tps": "4",
        "perplexity": "5", "notes": "",
    }


# --- pure resumable core -----------------------------------------------------

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


# --- cell runner + main (injected subprocess) --------------------------------

def _fake_run_factory(models_dir, *, bench=(0, LLAMA_BENCH_MD, TIME_V),
                      ppl=(0, PERPLEXITY, ""), download_ok=True):
    """A Runner stub that dispatches on argv and (on download) creates the GGUF."""
    def fake_run(argv):
        if "download" in argv:
            if download_ok:
                # argv: hf download <repo> <filename> --local-dir <dir>
                fname = argv[argv.index("download") + 2]
                (models_dir / fname).write_bytes(b"x" * 1024)
                return Completed(0, "", "")
            return Completed(1, "", "404 not found")
        if "llama-bench" in argv:
            return Completed(*bench)
        if "llama-perplexity" in argv:
            return Completed(*ppl)
        raise AssertionError(f"unexpected argv: {argv}")
    return fake_run


def test_run_cell_happy_path(tmp_path):
    models, raw = tmp_path / "models", tmp_path / "raw"
    models.mkdir()
    cell = Cell("0.5B", "Q4_K_M")
    row = run_cell(cell, run=_fake_run_factory(models), threads=4,
                   models_dir=models, raw_dir=raw)
    assert row["status"] == "ok"
    assert row["disk_bytes"] == 1024
    assert row["peak_rss_bytes"] == 1093284 * 1024
    assert row["prefill_tps"] == 7.71
    assert row["decode_tps"] == 3.87
    assert row["perplexity"] == 6.9543
    assert not (models / cell.filename).exists()       # GGUF deleted


def test_run_cell_download_failure_is_recorded_not_raised(tmp_path):
    models, raw = tmp_path / "models", tmp_path / "raw"
    models.mkdir()
    cell = Cell("3B", "Q8_0")
    run = _fake_run_factory(models, download_ok=False)
    row = run_cell(cell, run=run, threads=4, models_dir=models, raw_dir=raw)
    assert row["status"] == "download_failed"


def test_run_cell_bench_nonzero_exit_is_oom(tmp_path):
    models, raw = tmp_path / "models", tmp_path / "raw"
    models.mkdir()
    cell = Cell("3B", "Q8_0")
    run = _fake_run_factory(models, bench=(137, "", TIME_V))  # 137 = OOM-killed
    row = run_cell(cell, run=run, threads=4, models_dir=models, raw_dir=raw)
    assert row["status"] == "oom"
    assert not (models / cell.filename).exists()


def test_main_skips_completed_and_runs_only_pending(tmp_path, monkeypatch):
    csv_path = tmp_path / "sweep.csv"
    models = tmp_path / "models"
    models.mkdir()
    # Universe of 2 cells; pre-seed the CSV with one already done.
    monkeypatch.setattr(bench_config, "CSV_PATH", csv_path)
    monkeypatch.setattr(bench_config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(bench_config, "BENCH_MODELS_DIR", models)
    monkeypatch.setattr(bench_config, "cells",
                        lambda: [Cell("0.5B", "Q8_0"), Cell("0.5B", "Q6_K")])
    append_row(csv_path, _row("0.5B", "Q8_0"))

    calls = []
    base = _fake_run_factory(models)

    def counting_run(argv):
        if "download" in argv:
            calls.append(argv[argv.index("download") + 1])  # the repo
        return base(argv)

    main(run=counting_run)

    # Only the pending cell was downloaded; CSV now has both rows.
    assert calls == ["bartowski/Qwen2.5-0.5B-Instruct-GGUF"]
    assert read_completed(csv_path) == {("0.5B", "Q8_0"), ("0.5B", "Q6_K")}
