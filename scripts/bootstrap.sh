#!/usr/bin/env bash
# Run ON a fresh Raspberry Pi OS (64-bit) after cloning this repo to ~/faraday.
# One shot: deps -> build llama.cpp (-j3, 4GB-safe) -> models -> venv -> systemd units
# -> smoke test. Idempotent: every stage skips itself if already done.
set -euo pipefail
cd "$HOME/faraday"

echo "[1/5] System deps + Pi setup"
bash scripts/00_pi_setup.sh

echo "[2/5] llama.cpp (skip if built)"
if [[ ! -x "$HOME/llama.cpp/build/bin/llama-server" ]]; then
  bash scripts/10_build_llama.sh
fi

echo "[3/5] Models (skip if present)"
if ! ls "$HOME"/faraday/models/*q4_k_m.gguf >/dev/null 2>&1; then
  bash scripts/20_download_models.sh
fi

echo "[4/5] Python venv + package"
if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -e ".[dev]"

echo "[5/5] systemd units + smoke"
bash deploy/systemd/install.sh
sleep 20   # model load
bash scripts/40_smoke_test.sh
echo "Bootstrap complete — Faraday is live on :8000 and survives reboots."
