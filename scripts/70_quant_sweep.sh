#!/usr/bin/env bash
# Run ON the Raspberry Pi. Drives the M4a quantization sweep:
#   for each (Qwen2.5 size x quant) cell -> download GGUF -> time -v llama-bench
#   -> llama-perplexity -> append results/sweep/sweep.csv -> delete the GGUF.
# Resumable: re-running skips cells already in sweep.csv. Expect an overnight run.
#
# Ensures prereqs: GNU time (/usr/bin/time -v) and the wikitext perplexity corpus.
set -euo pipefail
cd "$HOME/faraday"
# shellcheck disable=SC1091
source .venv/bin/activate

# 1) GNU time — the shell builtin `time` has no -v (needed for peak-RSS capture).
if ! /usr/bin/time -v true >/dev/null 2>&1; then
  echo "Installing GNU time..."
  sudo apt-get update -qq && sudo apt-get install -y time
fi

# 2) Perplexity corpus: wikitext-2 raw test split (one-time, then cached).
CORPUS="$HOME/faraday/bench_data/wiki.test.raw"
if [[ ! -f "$CORPUS" ]]; then
  mkdir -p "$HOME/faraday/bench_data"
  if [[ -x "$HOME/llama.cpp/scripts/get-wikitext-2.sh" ]]; then
    ( cd "$HOME/faraday/bench_data" && "$HOME/llama.cpp/scripts/get-wikitext-2.sh" )
    found="$(find "$HOME/faraday/bench_data" -name 'wiki.test.raw' | head -1)"
    [[ -n "$found" && "$found" != "$CORPUS" ]] && cp "$found" "$CORPUS"
  else
    echo "ERROR: no wikitext getter found at ~/llama.cpp/scripts/get-wikitext-2.sh;" >&2
    echo "       place a perplexity corpus at $CORPUS manually and re-run." >&2
    exit 1
  fi
fi

# 3) Put llama.cpp bench tools on PATH (the Python calls them by bare name).
export PATH="$HOME/llama.cpp/build/bin:$PATH"

# 4) Health gate: don't trust numbers from a throttled board.
echo "throttle state (0x0 = healthy): $(vcgencmd get_throttled)"

# 5) Run the resumable sweep, then render the deliverables.
python -m faraday.bench.sweep
python -m faraday.bench.plot

echo "Done. Commit results/sweep/{sweep.csv,frontier.png,leaderboard.md} from the dev box."
