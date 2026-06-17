#!/usr/bin/env bash
# Run ON the Pi. Installs + enables the three Faraday units (idempotent).
set -euo pipefail
cd "$(dirname "$0")"
sudo cp faraday-llama-gen.service faraday-llama-embed.service faraday-app.service \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now faraday-llama-gen faraday-llama-embed faraday-app
systemctl --no-pager --type=service --state=running | grep faraday || true
echo "Installed. After 'git push pi', restart with: sudo systemctl restart faraday-app"
