"""Resumable, serial orchestrator for the M4a sweep. Runs ON the Pi:
for each pending cell -> download GGUF -> time -v llama-bench -> llama-perplexity
-> append a CSV row -> delete the GGUF. Resumable: cells already in the CSV are
skipped, so a reboot/power-off loses at most the in-flight cell.

This file's pure half (read_completed/pending/append_row) is unit-tested here;
the imperative half (run_cell/main) is added in Task 5 with an injected runner.
"""
from __future__ import annotations

import csv
from pathlib import Path

from faraday.bench import config
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
