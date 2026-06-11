#!/usr/bin/env bash
# Run ON the Raspberry Pi. Drives the M4b RAG eval RUN phase:
#   for each (top_k x chunk_size) config -> ingest corpus at that chunk-size ->
#   RagEngine.answer each golden question -> record results/evals/raw/<slug>.jsonl.
# Resumable (re-running skips done (config, question)). Needs the gen + embed
# servers up. Scoring (judge + scorecard) is a separate dev/Pi step (report.py).
set -euo pipefail
cd "$HOME/faraday"
# shellcheck disable=SC1091
source .venv/bin/activate

# Servers must be up (embed :8081 for ingest/retrieval, gen :8080 for answers).
if ! curl -sf http://localhost:8081/health >/dev/null 2>&1; then
  echo "Embed/gen servers not healthy — starting them..."
  bash scripts/30_run_servers.sh
  echo "Waiting ~20s for models to load..." && sleep 20
fi

echo "throttle (0x0 = healthy): $(vcgencmd get_throttled)"
python -m faraday.eval.runner
echo "Done. Raw rows in results/evals/raw/. Next: score with report.py (needs ANTHROPIC_API_KEY)."
