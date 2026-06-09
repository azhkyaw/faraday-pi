"""Real end-to-end smoke for ONE small cell. Deselected by default
(`integration` marker); run on the Pi with binaries on PATH + corpus present:
  export PATH="$HOME/llama.cpp/build/bin:$PATH"
  pytest -m integration tests/test_bench_integration.py -q
"""
import pytest

from faraday.bench.config import Cell
from faraday.bench.sweep import run_cell, subprocess_runner


@pytest.mark.integration
def test_one_cell_end_to_end(tmp_path):
    cell = Cell("0.5B", "Q4_K_M")  # smallest, certain to fit the 4 GB board
    row = run_cell(cell, run=subprocess_runner, threads=4,
                   models_dir=tmp_path, raw_dir=tmp_path / "raw")
    assert row["status"] == "ok", row["notes"]
    assert int(row["peak_rss_bytes"]) > 0
    assert float(row["decode_tps"]) > 0
    assert float(row["perplexity"]) > 0
