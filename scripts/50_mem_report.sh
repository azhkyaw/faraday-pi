#!/usr/bin/env bash
# Run ON the Raspberry Pi (with the servers running). Reports the HONEST memory
# footprint of the llama-servers — process RSS/PSS — next to what `free` shows,
# demonstrating why `free`'s "used" column undercounts mmap'd model weights
# (they live in buff/cache, not used). Reused as an M4 memory-measurement tool.
set -uo pipefail
MODELS="$HOME/faraday/models"

# [q]wen / [b]ge brackets prevent the pattern from matching this script's own
# process in `pgrep -f`.
gen_pid=$(pgrep -f '[q]wen' || true)
emb_pid=$(pgrep -f '[b]ge'  || true)

echo "=== model files on disk ==="
ls -lh "$MODELS"/*.gguf | awk '{print "  "$5"  "$9}'

echo
echo "=== what 'free' shows  (compare 'used' vs 'buff/cache') ==="
free -h

report() {
  local name=$1 pid=$2
  if [ -z "$pid" ]; then echo "  $name: not running"; return; fi
  local rss pss
  rss=$(awk '/^VmRSS:/{printf "%.0f MiB", $2/1024}' "/proc/$pid/status")
  pss=$(awk '/^Pss:/{sum+=$2} END{printf "%.0f MiB", sum/1024}' "/proc/$pid/smaps_rollup" 2>/dev/null)
  echo "  $name (pid $pid):  RSS=$rss  PSS=$pss"
}

echo
echo "=== what each process ACTUALLY holds (the honest footprint) ==="
report "gen   (qwen 1.5B)" "$gen_pid"
report "embed (bge)      " "$emb_pid"
