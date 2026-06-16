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


import os  # noqa: E402  (kept with the other imports conceptually; grouped at top on final lint)
from collections import defaultdict  # noqa: E402

from faraday.bench import optimize_config  # noqa: E402
from faraday.bench.optimize_config import CSV_PATH, RAW_DIR  # noqa: E402
from faraday.bench.sweep import subprocess_runner  # noqa: E402


def _is_clean(row: dict) -> bool:
    return bool(row["decode_tps"]) and "0x0" in str(row["throttled"])


def stack_winners(all_cells: list[LeverCell], rows: list[dict]) -> LeverCell:
    """Per lever-group, pick the single setting whose CLEAN decode beats baseline;
    union the winners' flags + best governor into one 'stacked_best' cell."""
    by_label = {r["label"]: r for r in rows}
    base_decode = float(by_label["baseline"]["decode_tps"])
    groups: dict[str, list[LeverCell]] = defaultdict(list)
    for c in all_cells:
        if c.component in ("governor", "threads", "batch", "kvquant", "flashattn"):
            groups[c.component].append(c)
    governor = "ondemand"
    extra: list[str] = []
    base_set = set(("-p", "512", "-n", "128"))
    for comp, comp_cells in groups.items():
        best_cell, best_dec = None, base_decode
        for cell in comp_cells:
            r = by_label.get(cell.label)
            if not r or not _is_clean(r):
                continue
            d = float(r["decode_tps"])
            if d > best_dec:
                best_dec, best_cell = d, cell
        if best_cell is None:
            continue
        if comp == "governor":
            governor = best_cell.governor
        else:
            for f in best_cell.flags:
                if f not in base_set and f not in extra:
                    extra.append(f)
    return LeverCell("stacked_best", "stacked_best", governor,
                     ("-p", "512", "-n", "128", *extra), "llama_bench")


def main(run: Runner = subprocess_runner) -> None:
    model = os.environ["FARADAY_OPT_MODEL"]      # set by scripts/90_optimize.sh
    draft = os.environ.get("FARADAY_OPT_DRAFT", "")
    ollama_model = os.environ.get("FARADAY_OPT_OLLAMA", "qwen2.5:1.5b")
    prompt = "Summarize the history of crewed spaceflight in three sentences."
    all_cells = optimize_config.cells()

    done = read_done(CSV_PATH)
    for cell in all_cells:
        if cell.key in done:
            continue
        print(f"--- {cell.component}/{cell.label} ---", flush=True)
        row = run_cell(cell, run, model=model, draft=draft,
                       ollama_model=ollama_model, prompt=prompt, raw_dir=RAW_DIR)
        append_row(CSV_PATH, row)
        print(f"    decode={row['decode_tps']} throttled={row['throttled']}", flush=True)

    best = stack_winners(all_cells, load_rows(CSV_PATH))
    if best.key not in read_done(CSV_PATH):
        print(f"--- {best.component} (gov={best.governor} {' '.join(best.flags)}) ---", flush=True)
        append_row(CSV_PATH, run_cell(best, run, model=model, draft=draft,
                                      ollama_model=ollama_model, prompt=prompt, raw_dir=RAW_DIR))
    # restore the default governor
    run(_governor_cmd("ondemand"))


if __name__ == "__main__":
    main()
