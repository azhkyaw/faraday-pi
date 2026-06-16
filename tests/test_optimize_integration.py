"""Real 1-cell optimization smoke. Deselected by default (`integration`); run on a
QUIET Pi with binaries on PATH + the model present:
  export PATH="$HOME/llama.cpp/build/bin:$PATH"
  pytest -m integration tests/test_optimize_integration.py -q
"""
import os
from pathlib import Path

import pytest

from faraday.bench.optimize import run_cell
from faraday.bench.optimize_config import LeverCell
from faraday.bench.sweep import subprocess_runner


@pytest.mark.integration
def test_baseline_cell_end_to_end(tmp_path):
    model = next(Path(os.path.expanduser("~/faraday/models")).glob("*q4_k_m.gguf"))
    cell = LeverCell("baseline", "baseline", "ondemand", ("-p", "128", "-n", "32"), "llama_bench")
    row = run_cell(cell, subprocess_runner, model=str(model), draft="", ollama_model="",
                   prompt="hi", raw_dir=tmp_path)
    assert row["notes"] == "ok", row["notes"]
    assert float(row["decode_tps"]) > 0
    assert int(row["peak_rss_bytes"]) > 0
    assert "0x" in row["throttled"]
