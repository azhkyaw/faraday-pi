# Faraday — M3 As-Built & Findings

**Status:** ✅ M3 complete — merged to `main` and on GitHub
**Plan:** [2026-06-09-faraday-m3-observability.md](./2026-06-09-faraday-m3-observability.md)
**Spec:** [2026-06-09-faraday-m3-observability-design.md](../specs/2026-06-09-faraday-m3-observability-design.md)

Live observability over the air-gapped appliance, built test-first and **additively** over M2.

## Delivered

- **`/metrics` on the FastAPI app** (`metrics.py`): RAG metrics (TTFT, retrieval, decode tok/s, requests in-flight, answer tokens, sources, `citations{validity}`) recorded by instrumenting the `/chat` event stream, plus a **Pi host collector** (CPU temp, throttle bitmask, under-voltage, per-server llama RSS).
- **llama-server native metrics** via `--metrics` on both instances.
- **Off-Pi monitoring stack** (`monitoring/`): Prometheus + Grafana via Docker Compose on the dev machine, scraping the Pi by LAN IP; a provisioned 8-panel Grafana dashboard (`uid: faraday`).
- **40 tests** (38 unit with fakes + 2 hardware integration), ruff clean. ~348 lines added; the Pi gains near-zero runtime overhead.

## Verified live (real data)

```
Pi /chat (2 requests) → app /metrics → Prometheus (3 targets UP) → Grafana
  faraday_pi_temp_celsius     = 41.868   (live Pi temp)
  faraday_requests_total      = 2        (both counted)
  faraday_ttft_seconds_count  = 2        (TTFT recorded per request)
  Grafana dashboard 'Faraday' : 8 panels (provisioned, loaded)
```

## Findings worth keeping

1. **A running server can serve stale code that passes every test.** During live bring-up the app's `/metrics` returned 404 even though the integration test was green. Cause: a `faraday serve` process from an earlier demo was still running **M2 bytecode** — `git push` updates files, not the memory of a long-running process. `TestClient` always builds a *fresh* app from current code, so tests can't catch this; only hitting the real deployed endpoint did. → motivates the **systemd restart-on-deploy** work in M5.
2. **mDNS `.local` doesn't resolve inside Docker containers.** Prometheus (containerized, on the dev machine) must scrape the Pi by **LAN IP**, not `raspberrypi.local`. Caught in spec review, confirmed in practice.
3. **The streaming event seam is also the telemetry seam.** Instrumentation observes the M2 `Sources`/`Token`/`Done` events in the server's `/chat` generator and never touches `RagEngine` — the same boundary that made M2 cheap made M3 cheap (`answer_stream` unchanged).
4. **`free` lies, RSS doesn't — now continuously.** `faraday_llama_rss_bytes` exposes the honest per-server resident memory (the M0 `mmap` insight) as a live gauge, alongside temp/throttle (the M0 thermal finding). The project's hand-measured findings are now dashboards.

## Deliberate calls

- **No `node_exporter`** — the app emits the three host gauges we care about (temp/throttle/RSS); avoids another Pi process and surfaces Pi-specific metrics `node_exporter` doesn't.
- **Stack off-Pi** — keeps the 4 GB budget intact; the workload node exposes, the monitoring node scrapes (standard production split).
- **Air-gap intact** — metric labels are low-cardinality only (`outcome`, `validity`, `server`); no query text or document content, ever.

## Process notes (recurring lesson)

Inline commands with nested quotes do **not** survive PowerShell→ssh→bash; this session hit it repeatedly (json validation, version prints, `pkill` self-match). Reliable patterns: committed `.sh`/`.py` files, filename args (`python -m json.tool <file>`), piping a file to `python3 -`, and the `[f]araday` bracket trick to avoid `pkill` self-match. Long-running Pi processes are best held by an attached background task, not `nohup … &` over a one-shot ssh.

## Lint note

Strict per-step TDD scattered imports mid-file (ruff E402); consolidated to the top in a final `style(m3)` commit. "Tests green" ≠ "done" — the linter is part of the completion gate.

**Next:** M4 — the inference lab: quantization sweep, RAG evals, optimization study, GBNF citations; profile TTFT vs corpus/`top_k` and the prefill-rate question (hand-measured baseline: TTFT ~3.25 s, decode ~3.6 tok/s, retrieval ~96 ms).
