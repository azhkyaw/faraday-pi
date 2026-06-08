#!/usr/bin/env bash
# Run ON the Raspberry Pi. Ensures the llama-servers are up, then serves the
# Faraday web app on 0.0.0.0:8000 (reachable from the dev machine's browser).
set -euo pipefail
cd "$HOME/faraday"
bash scripts/30_run_servers.sh
# shellcheck disable=SC1091
. .venv/bin/activate
exec faraday serve --host 0.0.0.0 --port 8000
