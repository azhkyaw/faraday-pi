"""Resumable, serial orchestrator for the M4a sweep. Runs ON the Pi:
for each pending cell -> download GGUF -> time -v llama-bench -> llama-perplexity
-> append a CSV row -> delete the GGUF. Resumable: cells already in the CSV are
skipped, so a reboot/power-off loses at most the in-flight cell.

Subprocess access is injected (`run`) so the orchestration unit-tests with a
fake runner — no Pi, no network, no GGUFs. (Mirrors the project's Protocol-DI
convention: real impls injected at the edge, fakes in tests.)
"""
from __future__ import annotations

import csv
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from faraday.bench import config, parsers
from faraday.bench.config import Cell


def read_completed(csv_path: Path) -> set[tuple[str, str]]:
    """(size, quant) keys already recorded, so we can skip them on a re-run."""
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as f:
        return {(r["size"], r["quant"]) for r in csv.DictReader(f)}


def pending(all_cells: list[Cell], completed: set[tuple[str, str]]) -> list[Cell]:
    """Cells not yet in the CSV, preserving smallest-first order."""
    return [c for c in all_cells if c.key not in completed]


def append_row(csv_path: Path, row: dict) -> None:
    """Append one cell's row, writing the header if the file is new."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=config.CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


@dataclass
class Completed:
    """Minimal subprocess result (so tests can fake it without subprocess)."""
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[list[str]], Completed]


def subprocess_runner(argv: list[str]) -> Completed:
    """Default Runner: actually shell out. (Binaries found via PATH, which the
    on-Pi runner script extends with ~/llama.cpp/build/bin.)"""
    p = subprocess.run(argv, capture_output=True, text=True)
    return Completed(p.returncode, p.stdout, p.stderr)


def _download_cmd(cell: Cell, dest: Path) -> list[str]:
    return ["hf", "download", cell.repo, cell.filename, "--local-dir", str(dest)]


def _bench_cmd(gguf: Path, threads: int) -> list[str]:
    # /usr/bin/time -v -> peak RSS on stderr; llama-bench md table on stdout.
    return ["/usr/bin/time", "-v", "llama-bench", "-m", str(gguf),
            "-p", "512", "-n", "128", "-t", str(threads), "-o", "md"]


def _perplexity_cmd(gguf: Path, threads: int) -> list[str]:
    return ["llama-perplexity", "-m", str(gguf), "-f", str(config.PERPLEXITY_CORPUS),
            "-t", str(threads), "--chunks", str(config.PERPLEXITY_CHUNKS)]


def _save_raw(raw_dir: Path, cell: Cell, kind: str, c: Completed) -> None:
    """Persist raw tool output for debugging. `.log` -> auto-ignored by git."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{cell.size}-{cell.quant}-{kind}.log").write_text(
        f"returncode={c.returncode}\n--- stdout ---\n{c.stdout}\n"
        f"--- stderr ---\n{c.stderr}\n"
    )


def _best_effort_rss(stderr: str) -> int | str:
    try:
        return parsers.parse_time_v(stderr)
    except ValueError:
        return ""


def run_cell(cell: Cell, *, run: Runner, threads: int,
             models_dir: Path | None = None, raw_dir: Path | None = None) -> dict:
    """Download -> bench -> perplexity -> build a CSV row, then delete the GGUF.
    Never raises for an expected failure (download/oom/parse): records `status`
    and returns so the sweep continues."""
    models_dir = models_dir or config.BENCH_MODELS_DIR
    raw_dir = raw_dir or config.RAW_DIR
    row = dict.fromkeys(config.CSV_COLUMNS, "")
    row["size"], row["quant"] = cell.size, cell.quant
    gguf = models_dir / cell.filename
    try:
        dl = run(_download_cmd(cell, models_dir))
        if dl.returncode != 0 or not gguf.exists():
            row["status"] = "download_failed"
            tail = dl.stderr.strip().splitlines()
            row["notes"] = tail[-1] if tail else ""
            return row
        row["disk_bytes"] = gguf.stat().st_size

        bench = run(_bench_cmd(gguf, threads))
        _save_raw(raw_dir, cell, "bench", bench)
        if bench.returncode != 0:
            row["status"] = "oom"  # bench failing to load = doesn't fit
            row["peak_rss_bytes"] = _best_effort_rss(bench.stderr)
            row["notes"] = "llama-bench non-zero exit (likely OOM / won't fit)"
            return row
        row["peak_rss_bytes"] = parsers.parse_time_v(bench.stderr)
        row["prefill_tps"], row["decode_tps"] = parsers.parse_llama_bench(bench.stdout)

        ppl = run(_perplexity_cmd(gguf, threads))
        _save_raw(raw_dir, cell, "ppl", ppl)
        row["perplexity"] = parsers.parse_perplexity(ppl.stdout + ppl.stderr)
        row["status"] = "ok"
        return row
    except Exception as e:  # parse error etc. — record and keep the sweep going
        row["status"] = "error"
        row["notes"] = f"{type(e).__name__}: {e}"
        return row
    finally:
        if gguf.exists():
            gguf.unlink()


def main(run: Runner = subprocess_runner) -> None:
    threads = os.cpu_count() or 4
    config.BENCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    done = read_completed(config.CSV_PATH)
    todo = pending(config.cells(), done)
    print(f"sweep: {len(done)} done, {len(todo)} pending", flush=True)
    for cell in todo:
        print(f"--- {cell.size} {cell.quant} ---", flush=True)
        row = run_cell(cell, run=run, threads=threads)
        append_row(config.CSV_PATH, row)
        print(f"    status={row['status']}", flush=True)


if __name__ == "__main__":
    main()
