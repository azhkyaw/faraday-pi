#!/usr/bin/env bash
# Run ON the Pi AFTER the M4b baseline run. Re-runs ONLY the baseline config with
# grammar-constrained decoding into a separate raw dir, then writes the before/after.
set -euo pipefail
cd "$HOME/faraday"
# shellcheck disable=SC1091
source .venv/bin/activate
export FARADAY_USE_GRAMMAR=1
export FARADAY_EVAL_CONFIGS=k4_c1200_o200
export FARADAY_EVAL_RAW_DIR="$HOME/faraday/results/evals/raw_grammar"
python -m faraday.eval.runner
python -m faraday.eval.gbnf_compare
echo "Done — commit results/evals/gbnf_before_after.md"
