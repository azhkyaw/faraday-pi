#!/usr/bin/env bash
# Record the Faraday Pi airplane-mode CLI demo GIF — RUN ON THE PI (via: ssh -tt pi ...).
#
# Light path (no headless browser, unlike VHS — keeps the 4 GB board safe next to the
# llama-servers): asciinema records the session, agg renders the cast to GIF. Severs the
# internet by pulling the default route — loopback (app<->llama-server) and the on-link LAN
# subnet (this SSH session) keep their own routes, so both survive; only off-subnet traffic
# (the internet) loses its path. Records scripts/demo_cli.sh so `curl` VISIBLY fails on
# camera, renders to docs/assets/demo-cli.gif, then ALWAYS restores the route via trap on
# EXIT (a crash never leaves the board offline; routes are in-memory, so a reboot self-heals
# as a backstop too).
set -euo pipefail

REPO="$HOME/faraday"
CAST="$(mktemp --suffix=.cast)"
GIF="$REPO/docs/assets/demo-cli.gif"
AGG="$HOME/.local/bin/agg"

# --- ensure recorders exist (net still up) ---
command -v asciinema >/dev/null || { echo "→ installing asciinema…"; sudo apt-get update -qq && sudo apt-get install -y -qq asciinema fonts-dejavu-core; }
if ! { [ -x "$AGG" ] || command -v agg >/dev/null; }; then
  echo "→ fetching agg (arm64)…"; mkdir -p "$HOME/.local/bin"
  curl -fsSL https://github.com/asciinema/agg/releases/latest/download/agg-aarch64-unknown-linux-gnu -o "$AGG"; chmod +x "$AGG"
fi
command -v agg >/dev/null && AGG="$(command -v agg)"

# --- snapshot the default route(s) so restore puts them back exactly ---
gw_dev() { awk -v k1=via -v k2=dev '{for(i=1;i<=NF;i++){if($i==k1)g=$(i+1);if($i==k2)d=$(i+1)}; print g, d}' <<<"$1"; }
read -r GW  DEV  < <(gw_dev "$(ip -4 route show default | head -1)")
read -r GW6 DEV6 < <(gw_dev "$(ip -6 route show default 2>/dev/null | head -1 || true)")

restore() {
  [ -n "${GW:-}"  ] && [ -n "${DEV:-}"  ] && sudo ip    route replace default via "$GW"  dev "$DEV"  2>/dev/null || true
  [ -n "${GW6:-}" ] && [ -n "${DEV6:-}" ] && sudo ip -6 route replace default via "$GW6" dev "$DEV6" 2>/dev/null || true
  rm -f "$CAST"
  echo "→ internet restored."
}
trap restore EXIT

echo "→ severing internet (drop default route; LAN + loopback stay up)…"
sudo ip route del default
[ -n "${GW6:-}" ] && sudo ip -6 route del default 2>/dev/null || true
curl --max-time 5 -s https://example.com >/dev/null && echo "  WARN: still online" || echo "  external egress : blocked ✓"
curl --max-time 5 -sf http://127.0.0.1:8080/health >/dev/null && echo "  gen   :8080     : up ✓" || echo "  WARN gen :8080 down"
curl --max-time 5 -sf http://127.0.0.1:8081/health >/dev/null && echo "  embed :8081     : up ✓" || echo "  WARN embed :8081 down"

echo "→ recording (asciinema)…"
cd "$REPO"
stty rows 18 cols 100 2>/dev/null || true       # an ssh -tt pty with no real terminal records 0x0
asciinema rec --overwrite -c "bash scripts/demo_cli.sh" "$CAST"
sed -i '1{s/"width": *0/"width": 100/; s/"height": *0/"height": 18/}' "$CAST"  # backstop the 0x0 header

echo "→ rendering $GIF (agg)…"
"$AGG" --font-size 20 --font-family "DejaVu Sans Mono" --theme dracula --idle-time-limit 1 "$CAST" "$GIF"
echo "→ done: $(du -h "$GIF" | cut -f1)  $GIF"
