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
# Gen context window. The appliance default is 4096; the M4b eval grid's largest
# cell (top_k=8 x chunk 2400 ~= 4.7k prompt tokens + 512 generated) needs more,
# so 80_run_evals.sh exports GEN_CTX=8192 (KV cache cost ~+117 MB, fine on 4 GB).
GEN_CTX="${GEN_CTX:-4096}"
GEN="$(ls "$M"/*q4_k_m.gguf | head -1)"
EMB="$(ls "$M"/*f16.gguf | head -1)"

pkill -f 'llama-server' 2>/dev/null || true
sleep 1

nohup "$BIN" -m "$GEN" -c "$GEN_CTX" -t "$THREADS" --metrics --host 0.0.0.0 --port 8080 \
  >/tmp/gen.log 2>&1 &
echo "gen   server pid $! (:8080, ctx $GEN_CTX)  model: $(basename "$GEN")"

nohup "$BIN" -m "$EMB" --embeddings --metrics -t "$THREADS" --host 0.0.0.0 --port 8081 \
  >/tmp/embed.log 2>&1 &
echo "embed server pid $! (:8081)  model: $(basename "$EMB")"

echo "logs: /tmp/gen.log  /tmp/embed.log  (give them ~10-20s to load)"
