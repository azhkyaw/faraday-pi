# Faraday M3 — Observability Implementation Plan

> **✅ STATUS: COMPLETE — merged to `main` and on GitHub.** 40 tests green; live Prometheus/Grafana dashboard verified with real traffic. See the [M3 as-built record](./2026-06-09-faraday-m3-as-built.md).
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Faraday observable — expose Prometheus metrics (RAG metrics + Pi host gauges) from the FastAPI app and native metrics from the llama-servers, and provide a committed off-Pi Prometheus + Grafana stack that scrapes the Pi and renders a live dashboard.

**Architecture:** Additive over M2. A new `metrics.py` defines `prometheus_client` metric objects + a Pi host collector (temp/throttle/RSS). `server.py` mounts `/metrics` and instruments the `/chat` streaming generator by observing the existing `Sources`/`Token`/`Done` event seam (no change to `RagEngine`). Prometheus + Grafana run on the dev machine via Docker Compose, scraping the Pi by LAN IP.

**Tech Stack:** prometheus-client · FastAPI ASGI mount · Prometheus · Grafana · Docker Compose · the M2 `faraday` package.

**Spec:** [2026-06-09-faraday-m3-observability-design.md](../specs/2026-06-09-faraday-m3-observability-design.md)

---

## Development Environment

Same remote loop as M1/M2: author on Windows (`C:\projects\piai`), `git push pi`, run `pytest` in the Pi venv. M3 work goes on a new branch **`m3-observability`** (off `main`). Task 1 re-points the Pi's repo at that branch and installs `prometheus-client`. The `monitoring/` stack (Tasks 6–7) runs on the **dev machine** and needs **Docker Desktop running**; the Pi's **LAN IP** is captured in Task 1 for the Prometheus scrape config.

## File Structure (delta from M2)

```
src/faraday/
  metrics.py       # NEW: prometheus metric objects + Pi host collector
  server.py        # MODIFY: mount /metrics, instrument /chat generator
tests/
  test_metrics.py        # NEW: host collector parsing + metric recording
  test_server.py         # MODIFY: /metrics endpoint test
  test_integration_pi.py # MODIFY: + scrape-after-chat case
scripts/
  30_run_servers.sh      # MODIFY: --metrics on both llama-servers
monitoring/              # NEW (runs on the dev machine)
  docker-compose.yml
  prometheus.yml
  grafana/provisioning/datasources/prometheus.yml
  grafana/provisioning/dashboards/dashboards.yml
  grafana/provisioning/dashboards/faraday.json
  README.md
pyproject.toml           # MODIFY: +prometheus-client
```

---

### Task 1: M3 setup — branch, dep, Pi deploy, capture LAN IP

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the M3 branch** `[on dev machine]`

Run: `git checkout -b m3-observability`
Expected: `Switched to a new branch 'm3-observability'`

- [ ] **Step 2: Add prometheus-client to dependencies**

In `pyproject.toml`, change the `dependencies` list to add the package (keep the others):

```toml
dependencies = [
  "httpx>=0.27",
  "sqlite-vec>=0.1.3",
  "pypdf>=4.0",
  "typer>=0.12",
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "prometheus-client>=0.20",
]
```

- [ ] **Step 3: Commit, push, re-point the Pi, install the dep**

```bash
git add pyproject.toml
git commit -m "chore(m3): add prometheus-client dep; start m3-observability"
git push pi m3-observability
ssh pi@raspberrypi.local "cd ~/faraday && git checkout m3-observability && . .venv/bin/activate && pip install -q -e '.[dev]' && python -c 'import prometheus_client' && echo DEPS_OK"
```
Expected: Pi reports `Switched to branch 'm3-observability'` and `DEPS_OK`.

- [ ] **Step 4: Capture the Pi's LAN IP (for the Prometheus scrape config)**

Run: `ssh pi@raspberrypi.local "hostname -I"`
Expected: one or more space-separated addresses (e.g. `192.168.1.42 172.17.0.1`). **Record the first IPv4** — it is used verbatim in `monitoring/prometheus.yml` (Task 6), because `raspberrypi.local` (mDNS) usually will not resolve from inside the Prometheus container.

---

### Task 2: Metric definitions

**Files:**
- Create: `src/faraday/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_metrics.py`:
```python
from faraday import metrics


def test_rag_metric_objects_exist_with_expected_names():
    # Names are what Prometheus exposes; assert the registry knows them.
    names = {
        metrics.REQUESTS._name,            # Counter -> faraday_requests (––_total exposed)
        metrics.RETRIEVAL_SECONDS._name,
        metrics.TTFT_SECONDS._name,
        metrics.REQUEST_SECONDS._name,
        metrics.ANSWER_TOKENS._name,
        metrics.DECODE_TPS._name,
        metrics.SOURCES_RETRIEVED._name,
        metrics.CITATIONS._name,
    }
    assert "faraday_requests" in names
    assert "faraday_ttft_seconds" in names
    assert "faraday_citations" in names


def test_in_flight_is_a_gauge_that_moves():
    metrics.IN_FLIGHT.set(0)
    metrics.IN_FLIGHT.inc()
    assert metrics.IN_FLIGHT._value.get() == 1.0
    metrics.IN_FLIGHT.dec()
    assert metrics.IN_FLIGHT._value.get() == 0.0
```

- [ ] **Step 2: Run to confirm failure**

Run: `git push pi m3-observability` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_metrics.py -v"`
Expected: FAIL — cannot import `faraday.metrics`.

- [ ] **Step 3: Implement the metric objects in `src/faraday/metrics.py`**

```python
from __future__ import annotations
from prometheus_client import Counter, Gauge, Histogram

# --- RAG metrics (updated by the server's /chat instrumentation) ---
REQUESTS = Counter("faraday_requests", "Total /chat requests", ["outcome"])
IN_FLIGHT = Gauge("faraday_requests_in_flight", "In-flight /chat requests")
RETRIEVAL_SECONDS = Histogram("faraday_retrieval_seconds", "Embed query + vector search time")
TTFT_SECONDS = Histogram("faraday_ttft_seconds", "Time to first generated token")
REQUEST_SECONDS = Histogram("faraday_request_seconds", "End-to-end /chat time")
ANSWER_TOKENS = Histogram("faraday_answer_tokens", "Tokens generated per answer",
                          buckets=(8, 16, 32, 64, 128, 256, 512))
DECODE_TPS = Histogram("faraday_decode_tps", "Decode tokens/sec per request",
                       buckets=(1, 2, 3, 4, 5, 6, 8, 12))
SOURCES_RETRIEVED = Histogram("faraday_sources_retrieved", "Sources returned per query",
                              buckets=(0, 1, 2, 3, 4, 5))
CITATIONS = Counter("faraday_citations", "Citations by validity", ["validity"])
```

- [ ] **Step 4: Run to confirm pass**

Run: `git push pi m3-observability` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_metrics.py -v"`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/faraday/metrics.py tests/test_metrics.py
git commit -m "feat(m3): prometheus rag metric definitions"
```

---

### Task 3: Pi host collector

**Files:**
- Modify: `src/faraday/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Add the failing test** `[on dev machine]`

Append to `tests/test_metrics.py`:
```python
from faraday.metrics import read_host_gauges


def test_host_gauges_parse_injected_readers():
    g = read_host_gauges(
        temp_reader=lambda: "48324\n",                 # millidegrees C
        throttled_reader=lambda: "throttled=0x50005",  # under-voltage now + occurred
        rss_reader=lambda: {"gen": 1305000000, "embed": 95000000},
    )
    assert abs(g["faraday_pi_temp_celsius"] - 48.324) < 0.001
    assert g["faraday_pi_throttled"] == 0x50005
    assert g["faraday_pi_under_voltage"] == 1.0
    assert g["faraday_llama_rss_bytes"]["gen"] == 1305000000


def test_host_gauges_skip_failing_readers():
    def boom():
        raise OSError("no vcgencmd here")
    g = read_host_gauges(temp_reader=boom, throttled_reader=boom,
                         rss_reader=lambda: {})
    assert "faraday_pi_temp_celsius" not in g     # failed read omitted, no crash
    assert g["faraday_llama_rss_bytes"] == {}
```

- [ ] **Step 2: Run to confirm failure**

Run: `git push pi m3-observability` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_metrics.py::test_host_gauges_parse_injected_readers -v"`
Expected: FAIL — cannot import `read_host_gauges`.

- [ ] **Step 3: Add the host collector to `src/faraday/metrics.py`**

Append:
```python
import re
import subprocess
from pathlib import Path

_UNDER_VOLT_BITS = 0x1 | 0x10000   # under-voltage now | under-voltage occurred


def _default_temp_reader() -> str:
    return Path("/sys/class/thermal/thermal_zone0/temp").read_text()


def _default_throttled_reader() -> str:
    return subprocess.run(["vcgencmd", "get_throttled"],
                          capture_output=True, text=True, timeout=3).stdout


def _default_rss_reader() -> dict[str, int]:
    """RSS bytes per llama-server, found by matching the model name in cmdline."""
    out: dict[str, int] = {}
    for name, needle in (("gen", "qwen"), ("embed", "bge")):
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                cmd = (proc / "cmdline").read_bytes().decode("utf-8", "replace").lower()
                if "llama-server" in cmd and needle in cmd:
                    status = (proc / "status").read_text()
                    m = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.MULTILINE)
                    if m:
                        out[name] = int(m.group(1)) * 1024
                    break
            except (OSError, ValueError):
                continue
    return out


def read_host_gauges(temp_reader=_default_temp_reader,
                     throttled_reader=_default_throttled_reader,
                     rss_reader=_default_rss_reader) -> dict:
    """Read Pi host gauges defensively; a failing source is omitted, never raised."""
    g: dict = {}
    try:
        g["faraday_pi_temp_celsius"] = int(temp_reader().strip()) / 1000.0
    except Exception:
        pass
    try:
        raw = throttled_reader()
        val = int(re.search(r"0x[0-9a-fA-F]+", raw).group(0), 16)
        g["faraday_pi_throttled"] = float(val)
        g["faraday_pi_under_voltage"] = 1.0 if (val & _UNDER_VOLT_BITS) else 0.0
    except Exception:
        pass
    try:
        g["faraday_llama_rss_bytes"] = rss_reader()
    except Exception:
        g["faraday_llama_rss_bytes"] = {}
    return g
```

- [ ] **Step 4: Run to confirm pass**

Run: `git push pi m3-observability` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_metrics.py -v"`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/faraday/metrics.py tests/test_metrics.py
git commit -m "feat(m3): pi host collector (temp/throttle/rss, defensive)"
```

---

### Task 4: Register the host collector with Prometheus

**Files:**
- Modify: `src/faraday/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Add the failing test** `[on dev machine]`

Append to `tests/test_metrics.py`:
```python
from prometheus_client import CollectorRegistry
from faraday.metrics import HostCollector


def test_host_collector_yields_prometheus_metrics():
    reg = CollectorRegistry()
    reg.register(HostCollector(
        temp_reader=lambda: "50000\n",
        throttled_reader=lambda: "throttled=0x0",
        rss_reader=lambda: {"gen": 1000, "embed": 200},
    ))
    sample = {m.name: m for m in reg.collect()}
    assert sample["faraday_pi_temp_celsius"].samples[0].value == 50.0
    assert sample["faraday_pi_throttled"].samples[0].value == 0.0
    rss = {s.labels["server"]: s.value for s in sample["faraday_llama_rss_bytes"].samples}
    assert rss == {"gen": 1000.0, "embed": 200.0}
```

- [ ] **Step 2: Run to confirm failure**

Run: `git push pi m3-observability` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_metrics.py::test_host_collector_yields_prometheus_metrics -v"`
Expected: FAIL — cannot import `HostCollector`.

- [ ] **Step 3: Add `HostCollector` to `src/faraday/metrics.py`**

Append (uses `read_host_gauges` from Task 3):
```python
from prometheus_client.core import GaugeMetricFamily


class HostCollector:
    """A custom Prometheus collector that reads Pi host gauges lazily on each scrape."""

    def __init__(self, temp_reader=_default_temp_reader,
                 throttled_reader=_default_throttled_reader,
                 rss_reader=_default_rss_reader):
        self._readers = (temp_reader, throttled_reader, rss_reader)

    def collect(self):
        g = read_host_gauges(*self._readers)
        for name in ("faraday_pi_temp_celsius", "faraday_pi_throttled",
                     "faraday_pi_under_voltage"):
            if name in g:
                yield GaugeMetricFamily(name, name.replace("_", " "), value=g[name])
        rss = g.get("faraday_llama_rss_bytes", {})
        if rss:
            fam = GaugeMetricFamily("faraday_llama_rss_bytes",
                                    "llama-server resident memory", labels=["server"])
            for server, value in rss.items():
                fam.add_metric([server], value)
            yield fam
```

- [ ] **Step 4: Run to confirm pass**

Run: `git push pi m3-observability` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_metrics.py -v"`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/faraday/metrics.py tests/test_metrics.py
git commit -m "feat(m3): host collector as a prometheus custom collector"
```

---

### Task 5: Instrument the server + mount /metrics

**Files:**
- Modify: `src/faraday/server.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: Add the failing tests** `[on dev machine]`

Append to `tests/test_server.py`:
```python
def test_metrics_endpoint_exposes_faraday_metrics():
    client = TestClient(server.app)
    body = client.get("/metrics").text
    assert "faraday_requests" in body
    assert "faraday_ttft_seconds" in body


def test_chat_records_metrics(tmp_path, monkeypatch, fake_embedder, make_llm):
    from faraday import metrics
    engine, store = _engine(tmp_path, fake_embedder, make_llm, "Answer [1].")
    monkeypatch.setattr(server, "make_engine", lambda settings: (engine, store))
    monkeypatch.setattr(server, "_preflight_ok", lambda settings: True)
    client = TestClient(server.app)

    before = metrics.CITATIONS.labels(validity="valid")._value.get()
    client.get("/chat", params={"q": "how much ram?"}).text
    after = metrics.CITATIONS.labels(validity="valid")._value.get()
    assert after == before + 1          # one valid citation ([1]) was recorded
```

- [ ] **Step 2: Run to confirm failure**

Run: `git push pi m3-observability` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_server.py::test_metrics_endpoint_exposes_faraday_metrics -v"`
Expected: FAIL — `/metrics` 404 (not mounted yet).

- [ ] **Step 3: Mount /metrics and instrument /chat in `src/faraday/server.py`**

Add imports near the top (after the existing imports). Note: `server.py` already imports `SourcesEvent, TokenEvent, DoneEvent, ErrorEvent` from M2 — do **not** re-import them (a duplicate import trips ruff F811). Only add what's new:
```python
import time
from prometheus_client import make_asgi_app, REGISTRY
from faraday import metrics

REGISTRY.register(metrics.HostCollector())
app.mount("/metrics", make_asgi_app())
```

Replace the existing `chat()` route's `gen()` inner function with an instrumented version (the route signature and preflight check are unchanged):
```python
    def gen():
        engine, store = make_engine(settings)
        metrics.IN_FLIGHT.inc()
        t0 = time.perf_counter()
        t_first = None
        ntok = 0
        outcome = "ok"
        try:
            for ev in engine.answer_stream(q):
                if isinstance(ev, SourcesEvent):
                    metrics.RETRIEVAL_SECONDS.observe(time.perf_counter() - t0)
                    metrics.SOURCES_RETRIEVED.observe(len(ev.sources))
                elif isinstance(ev, TokenEvent):
                    if t_first is None:
                        t_first = time.perf_counter()
                        metrics.TTFT_SECONDS.observe(t_first - t0)
                    ntok += 1
                elif isinstance(ev, DoneEvent):
                    metrics.CITATIONS.labels(validity="valid").inc(len(ev.cited_indices))
                    metrics.CITATIONS.labels(validity="invalid").inc(len(ev.invalid_citations))
                yield _format(ev)
            total = time.perf_counter() - t0
            metrics.REQUEST_SECONDS.observe(total)
            metrics.ANSWER_TOKENS.observe(ntok)
            if t_first is not None and ntok > 1 and total > (t_first - t0):
                metrics.DECODE_TPS.observe((ntok - 1) / (total - (t_first - t0)))
        except Exception as exc:                      # mid-stream failure
            outcome = "error"
            yield _format(ErrorEvent(str(exc)))
        finally:
            store.close()
            metrics.IN_FLIGHT.dec()
            metrics.REQUESTS.labels(outcome=outcome).inc()
```

Note: the existing top-of-file imports already include `SourcesEvent, TokenEvent, DoneEvent, ErrorEvent`; the new import line is harmless (idempotent) but you may instead rely on the existing import. Keep one import only — if a duplicate-import lint fires, drop the new `from faraday.events import ...` line.

- [ ] **Step 4: Run to confirm pass**

Run: `git push pi m3-observability` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_server.py -v"`
Expected: PASS (all server tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add src/faraday/server.py tests/test_server.py
git commit -m "feat(m3): mount /metrics + instrument /chat via the event seam"
```

---

### Task 6: llama-server native metrics + monitoring stack

**Files:**
- Modify: `scripts/30_run_servers.sh`
- Create: `monitoring/docker-compose.yml`, `monitoring/prometheus.yml`,
  `monitoring/grafana/provisioning/datasources/prometheus.yml`,
  `monitoring/grafana/provisioning/dashboards/dashboards.yml`, `monitoring/README.md`

- [ ] **Step 1: Enable native metrics on both llama-servers** `[on dev machine]`

In `scripts/30_run_servers.sh`, add `--metrics` to both launch lines (gen and embed). The gen line becomes:
```bash
nohup "$BIN" -m "$GEN" -c 4096 -t "$THREADS" --metrics --host 0.0.0.0 --port 8080 \
  >/tmp/gen.log 2>&1 &
```
and the embed line becomes:
```bash
nohup "$BIN" -m "$EMB" --embeddings --metrics -t "$THREADS" --host 0.0.0.0 --port 8081 \
  >/tmp/embed.log 2>&1 &
```

- [ ] **Step 2: Create `monitoring/docker-compose.yml`**

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: faraday-prometheus
    ports: ["9090:9090"]
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prom-data:/prometheus
  grafana:
    image: grafana/grafana:latest
    container_name: faraday-grafana
    ports: ["3000:3000"]
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
      - GF_SECURITY_ALLOW_EMBEDDING=true
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana-data:/var/lib/grafana
volumes:
  prom-data:
  grafana-data:
```

- [ ] **Step 3: Create `monitoring/prometheus.yml`** (replace `PI_HOST` with the Pi's LAN IP from Task 1, Step 4)

```yaml
global:
  scrape_interval: 10s
scrape_configs:
  - job_name: faraday-app
    static_configs:
      - targets: ["PI_HOST:8000"]
  - job_name: llama-gen
    static_configs:
      - targets: ["PI_HOST:8080"]
  - job_name: llama-embed
    static_configs:
      - targets: ["PI_HOST:8081"]
```

- [ ] **Step 4: Create `monitoring/grafana/provisioning/datasources/prometheus.yml`**

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

- [ ] **Step 5: Create `monitoring/grafana/provisioning/dashboards/dashboards.yml`**

```yaml
apiVersion: 1
providers:
  - name: Faraday
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
```

- [ ] **Step 6: Create `monitoring/README.md`**

```markdown
# Faraday monitoring (dev machine)

Prometheus + Grafana that scrape the Pi and render the Faraday dashboard.

## Setup
1. Ensure Docker Desktop is running.
2. Edit `prometheus.yml` — replace every `PI_HOST` with your Pi's LAN IP
   (find it with `ssh pi@raspberrypi.local "hostname -I"`). We use the IP, not
   `raspberrypi.local`, because mDNS `.local` names don't resolve inside the
   Prometheus container.
3. On the Pi, (re)start the servers + app so `/metrics` is live:
   `bash scripts/30_run_servers.sh && bash scripts/60_run_app.sh`
4. From this folder: `docker compose up -d`

## Use
- Prometheus targets: http://localhost:9090/targets (all three should be UP)
- Grafana dashboard: http://localhost:3000 (anonymous admin) -> "Faraday"

## Stop
`docker compose down`  (add `-v` to also wipe stored metrics)
```

- [ ] **Step 7: Commit**

```bash
git add scripts/30_run_servers.sh monitoring/
git commit -m "feat(m3): llama-server --metrics + prometheus/grafana compose stack"
```

---

### Task 7: Grafana dashboard JSON

**Files:**
- Create: `monitoring/grafana/provisioning/dashboards/faraday.json`

- [ ] **Step 1: Create the dashboard** `monitoring/grafana/provisioning/dashboards/faraday.json`

A provisioned dashboard with timeseries + stat panels. Paste exactly:
```json
{
  "uid": "faraday",
  "title": "Faraday",
  "schemaVersion": 39,
  "time": {"from": "now-30m", "to": "now"},
  "refresh": "10s",
  "panels": [
    {"id": 1, "title": "TTFT (s)", "type": "timeseries", "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
     "targets": [
       {"expr": "histogram_quantile(0.5, rate(faraday_ttft_seconds_bucket[5m]))", "legendFormat": "p50"},
       {"expr": "histogram_quantile(0.95, rate(faraday_ttft_seconds_bucket[5m]))", "legendFormat": "p95"}
     ]},
    {"id": 2, "title": "Decode tok/s (avg)", "type": "timeseries", "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
     "targets": [
       {"expr": "rate(faraday_decode_tps_sum[5m]) / rate(faraday_decode_tps_count[5m])", "legendFormat": "decode tok/s"}
     ]},
    {"id": 3, "title": "Retrieval (s, avg)", "type": "timeseries", "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
     "targets": [
       {"expr": "rate(faraday_retrieval_seconds_sum[5m]) / rate(faraday_retrieval_seconds_count[5m])", "legendFormat": "retrieval"}
     ]},
    {"id": 4, "title": "Requests/min + in-flight", "type": "timeseries", "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
     "targets": [
       {"expr": "60 * rate(faraday_requests_total[5m])", "legendFormat": "{{outcome}}/min"},
       {"expr": "faraday_requests_in_flight", "legendFormat": "in-flight"}
     ]},
    {"id": 5, "title": "Citations (valid vs invalid)", "type": "timeseries", "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
     "targets": [
       {"expr": "faraday_citations_total", "legendFormat": "{{validity}}"}
     ]},
    {"id": 6, "title": "Pi CPU temp (°C)", "type": "timeseries", "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
     "fieldConfig": {"defaults": {"max": 90, "thresholds": {"mode": "absolute", "steps": [
       {"color": "green", "value": null}, {"color": "red", "value": 80}]}}},
     "targets": [
       {"expr": "faraday_pi_temp_celsius", "legendFormat": "temp"}
     ]},
    {"id": 7, "title": "llama-server RSS (bytes)", "type": "timeseries", "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24},
     "targets": [
       {"expr": "faraday_llama_rss_bytes", "legendFormat": "{{server}}"}
     ]},
    {"id": 8, "title": "Throttle / under-voltage", "type": "stat", "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
     "targets": [
       {"expr": "faraday_pi_throttled", "legendFormat": "throttled bitmask"},
       {"expr": "faraday_pi_under_voltage", "legendFormat": "under-voltage"}
     ]}
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add monitoring/grafana/provisioning/dashboards/faraday.json
git commit -m "feat(m3): grafana dashboard (ttft, decode, temp, rss, citations)"
```

---

### Task 8: Integration — scrape after a real chat (on the Pi)

**Files:**
- Modify: `tests/test_integration_pi.py`

- [ ] **Step 1: Add the integration test** `[on dev machine]`

Append to `tests/test_integration_pi.py`:
```python
@pytest.mark.integration
def test_metrics_endpoint_after_chat(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from faraday import server

    db = str(tmp_path / "m.sqlite")
    monkeypatch.setenv("FARADAY_DB", db)
    s = Settings.from_env()
    store = SqliteVecStore(db, dim=s.embed_dim)
    ingest("examples/corpus", store=store, embedder=HttpEmbedder(s))
    store.close()

    client = TestClient(server.app)
    client.get("/chat", params={"q": "How much RAM can a Raspberry Pi 4 have?"}).text
    body = client.get("/metrics").text
    print("\nMETRICS SAMPLE:\n" + "\n".join(
        l for l in body.splitlines() if l.startswith("faraday_") and "pi_" in l))
    assert "faraday_requests_total" in body
    assert "faraday_pi_temp_celsius" in body          # host collector live on the Pi
    assert "faraday_ttft_seconds_count" in body
```

- [ ] **Step 2: Run on the Pi with servers up**

Run:
```bash
git push pi m3-observability
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && bash scripts/30_run_servers.sh >/dev/null && bash scripts/40_smoke_test.sh >/dev/null 2>&1 && pytest -m integration -v -s"
```
Expected: `test_metrics_endpoint_after_chat PASSED` and the printed sample shows `faraday_pi_temp_celsius` with a real value. (The two M1/M2 integration tests run too and stay green.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_pi.py
git commit -m "test(m3): integration — /metrics live after a chat on the Pi"
```

---

### Task 9: Bring up the monitoring stack (manual verification)

**Files:** none (uses `monitoring/` from Tasks 6–7)

- [ ] **Step 1: Set the Pi IP in the scrape config** `[on dev machine]`

Edit `monitoring/prometheus.yml`, replacing every `PI_HOST` with the LAN IP captured in Task 1 Step 4. Commit:
```bash
git add monitoring/prometheus.yml
git commit -m "chore(m3): set Pi LAN IP in prometheus scrape config"
```

- [ ] **Step 2: Ensure the Pi is serving metrics** `[on Pi]`

Run: `ssh pi@raspberrypi.local "cd ~/faraday && bash scripts/30_run_servers.sh && nohup bash -c 'cd ~/faraday && . .venv/bin/activate && faraday serve --host 0.0.0.0 --port 8000' >/tmp/app.log 2>&1 & sleep 6; curl -s localhost:8000/metrics | grep -c faraday_"`
Expected: a non-zero count (the app's `/metrics` is live).

- [ ] **Step 3: Start Prometheus + Grafana** `[on dev machine, Docker Desktop running]`

Run: `docker compose -f monitoring/docker-compose.yml up -d`
Then open `http://localhost:9090/targets` — expected: `faraday-app`, `llama-gen`, `llama-embed` all **UP**.

- [ ] **Step 4: Verify the dashboard** `[on dev machine]`

Open `http://localhost:3000` → dashboard **Faraday**. Generate traffic (open `http://<PI_IP>:8000` and ask a question, or curl `/chat`), and confirm TTFT, decode tok/s, Pi temp, and RSS panels populate. **This is the M3 done state.**

- [ ] **Step 5: (optional) Tear down**

Run: `docker compose -f monitoring/docker-compose.yml down`

---

## Final verification

- [ ] Full suite green on the Pi: `pytest -m 'integration or not integration' -q` and `ruff check src tests`.
- [ ] Prometheus shows all three Pi targets UP; the Grafana **Faraday** dashboard renders live TTFT, decode tok/s, temp, RSS, and citation counts under traffic.
- [ ] CLI and web app still work unchanged.

## Plan done criteria

Live observability over the air-gapped appliance: the FastAPI app and llama-servers expose Prometheus metrics; an off-Pi Docker stack scrapes and visualizes them; the dashboard shows the exact numbers (TTFT, decode tok/s, temp, RSS, hallucination counts) we previously measured by hand — now continuous. Then: finish the branch (merge to `main`) and push to GitHub.

## Deferred (unchanged from spec)

- **M4:** systematic profiling using these metrics (TTFT vs corpus/`top_k`; prefill-rate question); quantization sweep; RAG evals; GBNF citations.
- **M5:** systemd auto-start, Docker packaging of the appliance, security hardening; optional alerting.
