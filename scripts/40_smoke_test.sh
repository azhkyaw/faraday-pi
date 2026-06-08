#!/usr/bin/env bash
# Run ON the Raspberry Pi. Waits for both llama-servers to be healthy, then
# smoke-tests the generation + embedding APIs. Exit 0 = the appliance is up.
# Doubles as a post-boot health check. JSON lives in this file (one shell to
# escape through) rather than inline over SSH (four shells).
set -euo pipefail
GEN=http://localhost:8080
EMB=http://localhost:8081

echo "--- waiting for servers to load (up to 120s) ---"
for i in $(seq 1 60); do
  if curl -sf "$GEN/health" >/dev/null 2>&1 && curl -sf "$EMB/health" >/dev/null 2>&1; then
    echo "both healthy after $((i*2))s"; break
  fi
  sleep 2
done

echo "--- generation (/v1/chat/completions) ---"
curl -s "$GEN/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly one word: ready"}],"max_tokens":8,"temperature":0}' \
  | python3 -c 'import sys,json; print("reply:", json.load(sys.stdin)["choices"][0]["message"]["content"].strip())'

echo "--- embeddings (/v1/embeddings) ---"
curl -s "$EMB/v1/embeddings" -H 'Content-Type: application/json' \
  -d '{"input":"hello world"}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print("embedding dims:", len(d["data"][0]["embedding"]))'

echo "--- memory after both models loaded ---"
free -h | awk '/Mem:/{print "  RAM  used/total: "$3"/"$2}  /Swap:/{print "  Swap used/total: "$3"/"$2}'
