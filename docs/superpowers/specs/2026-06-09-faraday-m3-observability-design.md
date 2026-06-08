# Faraday M3 — Observability
## Design Document

| | |
|---|---|
| **Status** | Approved (brainstorming) — ready for implementation planning |
| **Date** | 2026-06-09 |
| **Milestone** | M3 (extends the [main design spec](./2026-06-08-faraday-edge-rag-appliance-design.md) §6.6 / §14) |
| **Builds on** | M0–M2 (the RAG core, CLI, and streaming web server, on `main`) |

---

## 1. Overview

M3 makes Faraday's runtime observable: the operational numbers we measured by hand (TTFT, decode tok/s, retrieval latency) become **live, continuous metrics**. The FastAPI app exposes a Prometheus `/metrics` endpoint covering RAG-specific metrics and Pi host gauges; the two llama-servers expose their native `/metrics`; **Prometheus + Grafana run off-box on the dev machine** (Docker Compose) and scrape the Pi over the LAN.

M3 is **additive** — it instruments the existing M2 serving path and adds a monitoring stack; the RAG logic (`RagEngine`, retrieval, generation) is unchanged.

## 2. Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Monitoring stack location | **Off-Pi**, on the dev machine via Docker Compose | Keeps the 4 GB Pi budget intact (already ~1.6 GB used); the standard production pattern (workload node exposes, monitoring node scrapes) |
| Metrics scope | **App RAG metrics + llama-server native + Pi host gauges** | Richest picture; host temp/throttle/RSS tie directly to the project's thermal/`mmap` findings |
| Host metrics agent | **App-exposed custom collector** (no `node_exporter`) | No extra Pi process; `node_exporter` doesn't expose Pi temp/throttle anyway |
| Alerting / Grafana auth / retention tuning | **Deferred / none** | YAGNI for a single-user appliance |

## 3. Goals / non-goals

**Goals**
- Expose a Prometheus `/metrics` endpoint from the FastAPI app: RAG metrics + Pi host gauges.
- Launch the llama-servers with their native Prometheus metrics enabled.
- Provide a committed, reproducible Prometheus + Grafana stack (Docker Compose) that scrapes the Pi and renders a provisioned dashboard.
- Full test coverage of the instrumentation with fakes + one integration check on the Pi.

**Non-goals (YAGNI)**
- Alerting / alertmanager, Grafana auth/users, long-term retention tuning, `node_exporter`, distributed tracing, log aggregation.

## 4. Architecture

```
  Windows dev machine (Docker Compose)          Raspberry Pi (lean — exposes metrics only)
  ┌─────────────────────────────┐               ┌──────────────────────────────────────┐
  │ Prometheus ──scrape (LAN)───────────────────▶ FastAPI :8000 /metrics  (RAG + host)  │
  │     │                       ───────────────▶ llama-server gen   :8080 /metrics      │
  │     ▼                       ───────────────▶ llama-server embed :8081 /metrics      │
  │ Grafana ◀── PromQL ──── Prometheus           └──────────────────────────────────────┘
  └─────────────────────────────┘
```

The Pi adds near-zero overhead: `/metrics` is an in-process registry render; the llama-server metrics are a launch flag. All storage and visualization live on the dev machine.

## 5. Components

### 5.1 `metrics.py` (new)
Single responsibility: define the metric objects and the host collector.

**RAG metrics** (`prometheus_client`):
- `faraday_requests_total{outcome}` — Counter (outcome = `ok` | `error`).
- `faraday_requests_in_flight` — Gauge (concurrency / queue-depth proxy).
- `faraday_retrieval_seconds` — Histogram (embed query + vector search).
- `faraday_ttft_seconds` — Histogram (request received → first token).
- `faraday_request_seconds` — Histogram (full end-to-end).
- `faraday_answer_tokens` — Histogram (tokens generated per answer).
- `faraday_decode_tps` — Histogram (decode tokens/sec per request).
- `faraday_sources_retrieved` — Histogram (sources returned per query).
- `faraday_citations_total{validity}` — Counter (validity = `valid` | `invalid`); the live hallucination-rate signal.

**Host collector** (a custom `prometheus_client` collector, read lazily on each scrape):
- `faraday_pi_temp_celsius` — Gauge, from `/sys/class/thermal/thermal_zone0/temp`.
- `faraday_pi_throttled` — Gauge, the integer bitmask from `vcgencmd get_throttled` (0 = healthy).
- `faraday_pi_under_voltage` — Gauge (0/1), derived from the bitmask's under-voltage bits (current or sticky).
- `faraday_llama_rss_bytes{server}` — Gauge (server = `gen` | `embed`), honest RSS from `/proc/<pid>/status` (PID found by matching the model name in the cmdline, like `50_mem_report.sh`).

Reads are defensive: any failed source (e.g. `vcgencmd` missing) is skipped, never crashing `/metrics`. Collector logic takes injectable readers so it is unit-testable off-Pi.

### 5.2 `server.py` (modified)
- Mount the Prometheus ASGI app at `/metrics` (`prometheus_client.make_asgi_app()`).
- Instrument the `/chat` streaming generator by observing the event stream (no change to `RagEngine`):
  - on `SourcesEvent`: record `retrieval_seconds`, observe `sources_retrieved`.
  - on first `TokenEvent`: record `ttft_seconds`.
  - on `DoneEvent`: record `request_seconds`, `answer_tokens`, `decode_tps`; increment `citations_total{validity}` by the valid/invalid counts.
  - `requests_in_flight` incremented/decremented around the request; `requests_total{outcome}` on completion/error.
- Instrumentation is wrapped so a metrics failure never breaks a `/chat` response.

### 5.3 `scripts/30_run_servers.sh` (modified)
Add `--metrics` to both `llama-server` launch lines so each exposes a native Prometheus `/metrics` endpoint (prompt/generation token counters, KV-cache usage, etc.).

### 5.4 `monitoring/` (new — runs on the dev machine)
- `docker-compose.yml` — `prometheus` + `grafana` services, named volumes, ports 9090 / 3000.
- `prometheus.yml` — scrape config with three Pi targets (`:8000`, `:8080`, `:8081`). The Pi is addressed by its **LAN IP**, not `raspberrypi.local`: mDNS `.local` names usually don't resolve from inside a Docker container, so the IP is set via a documented one-line edit (a `PI_HOST` placeholder in `prometheus.yml` + a note in `monitoring/README.md`).
- `grafana/provisioning/datasources/prometheus.yml` — auto-wire the Prometheus datasource.
- `grafana/provisioning/dashboards/dashboards.yml` + `faraday.json` — the committed dashboard, auto-loaded on start.
- `README.md` — how to set the Pi address and `docker compose up`.

### 5.5 `pyproject.toml` (modified)
Add `prometheus-client` to dependencies.

## 6. Dashboard (Grafana, committed JSON)

Panels: TTFT (p50/p95), decode tok/s, retrieval latency, requests/min + in-flight, citations valid vs invalid (hallucination rate), Pi CPU temp (with an 80 °C throttle reference line), throttle/under-voltage state, llama-server RSS (gen + embed), and a few from the llama-server native metrics (prompt vs generation tokens, KV-cache usage).

## 7. Data flow

```
/chat request → instrumented generator updates in-process metric objects
Prometheus (dev machine) → scrapes Pi :8000/:8080/:8081 every ~10s → stores time series
Grafana → PromQL queries Prometheus → renders dashboard panels
```

## 8. Privacy / air-gap

Metrics are exposed on the LAN and contain **only low-cardinality counters and timings** — never query text or document content in any label. Document data still never leaves the device; this is operational telemetry, not user data. (Low cardinality is also mandatory Prometheus hygiene — query-text labels would explode the TSDB.)

## 9. Error handling

- **Host collector:** each source (thermal, `vcgencmd`, `/proc`) is read in a try/except; a failure omits that gauge rather than failing the scrape.
- **Instrumentation:** wrapped so a metrics error cannot break a `/chat` response.
- **Prometheus can't reach the Pi** (Pi off / wrong address): the target simply shows down in Prometheus — expected, no Pi-side impact.
- **llama-server RSS PID not found** (server restarting): omit that gauge for the scrape.

## 10. Testing

- **Unit (off-Pi, fakes):**
  - Instrumented `/chat` records the right metrics given a fake event stream (assert histogram/counter samples via the registry).
  - Host collector parses injected `/sys` + `vcgencmd` sample strings into the right gauge values; a failing reader is skipped gracefully.
  - `GET /metrics` returns 200 and Prometheus text exposition containing the `faraday_*` metric names.
- **Integration (Pi, real servers):** issue a `/chat`, scrape `/metrics`, assert `faraday_requests_total` moved and `faraday_pi_temp_celsius` is present and plausible.
- **Monitoring stack:** config, not unit-tested; verified by Prometheus targets-up + Grafana rendering data (manual, documented in `monitoring/README.md`).
- TDD throughout; ruff clean. Same remote loop: author on Windows → `git push pi` → `pytest` on the Pi.

## 11. File structure (delta from M2)

```
src/faraday/
  metrics.py       # NEW: metric defs + Pi host collector
  server.py        # MODIFY: mount /metrics, instrument /chat generator
tests/
  test_metrics.py        # NEW: host collector + metric recording
  test_server.py         # MODIFY: /metrics endpoint + instrumentation
  test_integration_pi.py # MODIFY: + scrape-after-chat case
scripts/
  30_run_servers.sh      # MODIFY: --metrics on both servers
monitoring/              # NEW (dev machine)
  docker-compose.yml
  prometheus.yml
  grafana/provisioning/datasources/prometheus.yml
  grafana/provisioning/dashboards/dashboards.yml
  grafana/provisioning/dashboards/faraday.json
  README.md
pyproject.toml           # MODIFY: +prometheus-client
```

## 12. Definition of done

- The FastAPI app exposes `/metrics` with the RAG metrics + Pi host gauges; both llama-servers expose native `/metrics`.
- `docker compose up` in `monitoring/` brings up Prometheus + Grafana; Prometheus shows all three Pi targets up; the provisioned Grafana dashboard renders live TTFT, decode tok/s, temp, RSS, and citation counts while traffic flows.
- All unit tests green (fakes) + the integration scrape test green on the Pi; ruff clean.
- The CLI and web app still work unchanged.

## 13. Deferred to later milestones

- **M4:** systematic latency/throughput profiling using these metrics (TTFT vs corpus size / `top_k`; the prefill-rate question from the M2 sanity reading); quantization sweep; RAG evals; GBNF citations.
- **M5:** systemd auto-start for the app + metrics, Docker packaging of the appliance, security hardening; optional alerting.
