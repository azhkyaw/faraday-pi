# Faraday monitoring (dev machine)

Prometheus + Grafana that scrape the Pi and render the Faraday dashboard.

## Setup
1. Ensure Docker Desktop is running.
2. `prometheus.yml` scrapes the Pi by **LAN IP** (currently `192.168.100.59`). If
   your Pi's IP differs, edit all three targets. We use the IP, not
   `raspberrypi.local`, because mDNS `.local` names don't resolve inside the
   Prometheus container. Find the IP with `ssh pi@raspberrypi.local "hostname -I"`.
3. On the Pi, (re)start the servers + app so `/metrics` is live:
   `bash scripts/30_run_servers.sh && bash scripts/60_run_app.sh`
4. From this folder: `docker compose up -d`

## Use
- Prometheus targets: http://localhost:9090/targets (all three should be UP)
- Grafana dashboard: http://localhost:3000 (anonymous admin) -> "Faraday"

## Stop
`docker compose down`  (add `-v` to also wipe stored metrics)
