#!/usr/bin/env bash
# Run ON the Raspberry Pi. Downloads the generation + embedding GGUF models
# into ~/faraday/models using the project venv.
#
# Debian 13 is PEP 668 "externally managed", so we never pip-install into the
# system Python — everything goes in ~/faraday/.venv.
set -euo pipefail
cd "$HOME/faraday"

[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q "huggingface-hub"   # provides the `hf` CLI (v1.x removed `huggingface-cli`)

mkdir -p models
# Generation model (~1.0 GB): Qwen2.5-1.5B-Instruct, Q4_K_M.
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --include "*q4_k_m.gguf" --local-dir models
# Embedding model (~130 MB): bge-small-en-v1.5, f16 (quality matters for retrieval).
hf download CompendiumLabs/bge-small-en-v1.5-gguf \
  --include "*f16.gguf" --local-dir models

echo "--- models ---"
ls -lh models
