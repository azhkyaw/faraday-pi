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
