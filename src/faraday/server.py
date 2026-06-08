from __future__ import annotations
import json
import time
from pathlib import Path
import httpx
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from prometheus_client import make_asgi_app, REGISTRY
from faraday import metrics
from faraday.config import Settings
from faraday.embedder import HttpEmbedder
from faraday.events import SourcesEvent, TokenEvent, DoneEvent, ErrorEvent
from faraday.index_store import SqliteVecStore
from faraday.llm_client import HttpLLMClient
from faraday.rag import RagEngine
from faraday.retriever import Retriever

app = FastAPI(title="Faraday")
STATIC = Path(__file__).parent / "static"

REGISTRY.register(metrics.HostCollector())
app.mount("/metrics", make_asgi_app())


def make_engine(settings: Settings):
    """Build a per-request engine + store (caller closes the store)."""
    store = SqliteVecStore(settings.db_path, dim=settings.embed_dim)
    engine = RagEngine(Retriever(HttpEmbedder(settings), store), HttpLLMClient(settings),
                       top_k=settings.top_k, max_tokens=settings.max_tokens)
    return engine, store


def _preflight_ok(settings: Settings) -> bool:
    """Both llama-servers reachable?"""
    try:
        for url in (settings.embed_url, settings.gen_url):
            httpx.get(url + "/health", timeout=2.0).raise_for_status()
        return True
    except Exception:
        return False


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _format(ev) -> str:
    if isinstance(ev, SourcesEvent):
        return _sse("sources", [{"n": i + 1, "source": rc.chunk.source, "score": round(rc.score, 3)}
                                 for i, rc in enumerate(ev.sources)])
    if isinstance(ev, TokenEvent):
        return _sse("token", {"text": ev.text})
    if isinstance(ev, DoneEvent):
        return _sse("done", {"cited": ev.cited_indices, "invalid": ev.invalid_citations})
    if isinstance(ev, ErrorEvent):
        return _sse("error", {"message": ev.message})
    return ""


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/chat")
def chat(q: str):
    settings = Settings.from_env()
    if not _preflight_ok(settings):
        return Response(status_code=503, content="llama-servers unavailable")

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

    return StreamingResponse(gen(), media_type="text/event-stream")
