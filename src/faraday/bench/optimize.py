"""M4c optimization runner: apply a lever cell, time the tool, record a CSV row.
Reuses the M4a injected-Runner + parsers. Resumable. The pure helpers unit-test
off the Pi; main() (path resolution + the full sweep) runs on the Pi.
"""
from __future__ import annotations

import csv
from pathlib import Path

from faraday.bench import parsers
from faraday.bench.optimize_config import CSV_COLUMNS, LeverCell
from faraday.bench.sweep import Completed, Runner


def _governor_cmd(gov: str) -> list[str]:
    # Passwordless sudo on the Pi; sh -c for the cpu* glob.
    return ["sudo", "sh", "-c",
            f"echo {gov} | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"]


def build_argv(cell: LeverCell, *, model: str, draft: str,
               ollama_model: str, prompt: str) -> list[str]:
    if cell.kind == "llama_bench":
        return ["/usr/bin/time", "-v", "llama-bench", "-m", model, *cell.flags, "-o", "md"]
    if cell.kind == "speculative":
        return ["/usr/bin/time", "-v", "llama-speculative", "-m", model, "-md", draft,
                "-p", prompt, "-n", "128", "--draft-max", "16"]
    if cell.kind == "ollama":
        return ["ollama", "run", "--verbose", ollama_model, prompt]
    raise ValueError(f"unknown cell kind: {cell.kind!r}")


def _save_raw(raw_dir: Path | None, cell: LeverCell, c: Completed) -> None:
    if raw_dir is None:
        return
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    (Path(raw_dir) / f"{cell.component}-{cell.label}.log".replace("/", "_")).write_text(
        f"returncode={c.returncode}\n--- stdout ---\n{c.stdout}\n--- stderr ---\n{c.stderr}\n")


def run_cell(cell: LeverCell, run: Runner, *, model: str, draft: str,
             ollama_model: str, prompt: str, raw_dir: Path | None) -> dict:
    """Apply governor (if any) -> run the tool -> parse -> build a CSV row.
    Never raises for a measurement failure: records `notes` and returns."""
    row = dict.fromkeys(CSV_COLUMNS, "")
    row["component"], row["label"] = cell.component, cell.label
    if cell.governor:
        run(_governor_cmd(cell.governor))
    c = run(build_argv(cell, model=model, draft=draft,
                       ollama_model=ollama_model, prompt=prompt))
    _save_raw(raw_dir, cell, c)
    row["throttled"] = run(["vcgencmd", "get_throttled"]).stdout.strip()
    try:
        if cell.kind == "llama_bench":
            row["peak_rss_bytes"] = parsers.parse_time_v(c.stderr)
            row["prefill_tps"], row["decode_tps"] = parsers.parse_llama_bench(c.stdout)
        elif cell.kind == "ollama":
            row["prefill_tps"], row["decode_tps"] = parsers.parse_ollama_bench(c.stdout + c.stderr)
        elif cell.kind == "speculative":
            row["decode_tps"], row["accept_rate"] = parsers.parse_speculative(c.stdout + c.stderr)
        row["notes"] = "ok" if c.returncode == 0 else f"exit {c.returncode}"
    except Exception as e:  # parse/oom/unsupported — record and keep going
        row["notes"] = f"{type(e).__name__}: {e}"
    return row


def append_row(csv_path: Path, row: dict) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    is_new = not Path(csv_path).exists()
    with Path(csv_path).open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def read_done(csv_path: Path) -> set[tuple[str, str]]:
    p = Path(csv_path)
    if not p.exists():
        return set()
    with p.open(newline="") as f:
        return {(r["component"], r["label"]) for r in csv.DictReader(f)}


def load_rows(csv_path: Path) -> list[dict]:
    with Path(csv_path).open(newline="") as f:
        return list(csv.DictReader(f))
