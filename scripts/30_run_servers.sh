#!/usr/bin/env bash
# Run ON the Raspberry Pi. Launches two llama-server instances:
#   :8080  generation  (Qwen2.5-1.5B-Instruct Q4_K_M)
#   :8081  embeddings  (bge-small-en-v1.5 f16, --embeddings)
#
# One llama-server process serves one model, so generation and embeddings are
# separate instances. Both bind 0.0.0.0 so the dev machine can reach them.
# nohup detaches them so they survive this SSH session (a rehearsal for systemd).
set -euo pipefail
BIN="$HOME/llama.cpp/build/bin/llama-server"
M="$HOME/faraday/models"
THREADS="$(nproc)"
GEN="$(ls "$M"/*q4_k_m.gguf | head -1)"
EMB="$(ls "$M"/*f16.gguf | head -1)"

pkill -f 'llama-server' 2>/dev/null || true
sleep 1

nohup "$BIN" -m "$GEN" -c 4096 -t "$THREADS" --host 0.0.0.0 --port 8080 \
  >/tmp/gen.log 2>&1 &
echo "gen   server pid $! (:8080)  model: $(basename "$GEN")"

nohup "$BIN" -m "$EMB" --embeddings -t "$THREADS" --host 0.0.0.0 --port 8081 \
  >/tmp/embed.log 2>&1 &
echo "embed server pid $! (:8081)  model: $(basename "$EMB")"

echo "logs: /tmp/gen.log  /tmp/embed.log  (give them ~10-20s to load)"
