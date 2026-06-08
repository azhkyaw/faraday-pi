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
