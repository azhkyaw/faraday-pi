from __future__ import annotations
import re
import subprocess
from pathlib import Path
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily

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
