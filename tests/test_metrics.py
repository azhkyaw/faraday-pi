from prometheus_client import CollectorRegistry
from faraday import metrics
from faraday.metrics import read_host_gauges, HostCollector


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
