#!/usr/bin/env bash
# Run ON the Raspberry Pi, on a QUIET board (nothing else competing). Drives the
# M4c optimization sweep on 1.5B Q4_K_M: ablate each tuning lever vs baseline,
# stack the winners, measure speculative decoding + an Ollama baseline + the
# context curve. Resumable (re-running skips done cells). Records throttle per cell.
#
# Overclock is a SEPARATE step (see bottom) — it needs a reboot, so it is not part
# of this runner; run it after, then re-run to fill the OC rows.
set -euo pipefail
cd "$HOME/faraday"
# shellcheck disable=SC1091
source .venv/bin/activate
export PATH="$HOME/llama.cpp/build/bin:$PATH"

# 1) GNU time (peak RSS) — same prereq as M4a.
if ! /usr/bin/time -v true >/dev/null 2>&1; then sudo apt-get install -y time; fi

# 2) llama-speculative must be built (the Pi build lacks it — like llama-perplexity).
if [[ ! -x "$HOME/llama.cpp/build/bin/llama-speculative" ]]; then
  echo "Building llama-speculative (-j3, 4GB-safe)..."
  cmake --build "$HOME/llama.cpp/build" --target llama-speculative -j3
fi

# 3) Models: the 1.5B Q4_K_M target + a 0.5B draft (same Qwen2.5 family = shared tokenizer).
export FARADAY_OPT_MODEL="$(ls "$HOME"/faraday/models/*q4_k_m.gguf | head -1)"
DRAFT="$(ls "$HOME"/faraday/models/*0.5B*q4_k_m.gguf 2>/dev/null | head -1 || true)"
if [[ -z "$DRAFT" ]]; then
  echo "Fetching 0.5B draft model..."
  hf download bartowski/Qwen2.5-0.5B-Instruct-GGUF Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
    --local-dir "$HOME/faraday/models"
  DRAFT="$(ls "$HOME"/faraday/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf)"
fi
export FARADAY_OPT_DRAFT="$DRAFT"

# 4) Ollama baseline.
if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama..."; curl -fsSL https://ollama.com/install.sh | sh
fi
ollama pull qwen2.5:1.5b
export FARADAY_OPT_OLLAMA="qwen2.5:1.5b"

# 5) Health gate + run + plot.
echo "throttle (0x0 = healthy): $(vcgencmd get_throttled)"
python -m faraday.bench.optimize
python -m faraday.bench.optimize_plot
echo "Done. Commit results/optimize/{optimize.csv,waterfall.png,lever_gains.png,context_curve.png,leaderboard.md}."

# --- OVERCLOCK (separate, manual; needs a reboot) -----------------------------
# To add the overclock rows after the stock-clock sweep:
#   1) sudo sh -c 'printf "\n[all]\narm_freq=2000\nover_voltage=6\n" >> /boot/firmware/config.txt'
#   2) sudo reboot   (re-run scripts/30_run_servers.sh etc. after boot if needed)
#   3) Verify cooling + PSU: watch `vcgencmd measure_temp` / `get_throttled` under load.
#   4) Re-run this script — it appends OC-clock rows for the still-pending cells.
#   5) Revert: remove those lines from config.txt + reboot.
