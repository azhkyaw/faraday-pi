# Faraday M2 — Streaming Web Serving Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the M1 `RagEngine` in a FastAPI service that streams grounded, cited answers token-by-token over SSE to a minimal vanilla-JS web chat, fully offline.

**Architecture:** Purely additive over M1. A new streaming path (`LLMClient.stream()` → `RagEngine.answer_stream()` yielding `Sources`/`Token`/`Done` events) feeds a FastAPI `/chat` SSE endpoint that a static HTML page consumes via `EventSource`. The store is opened per-request (single-user appliance). The CLI's `ask`/`ingest` are unchanged; a new `faraday serve` launches the app.

**Tech Stack:** FastAPI · uvicorn · httpx streaming · Server-Sent Events · vanilla JS (`EventSource`) · pytest (`TestClient`) · the M1 `faraday` package.

**Spec:** [2026-06-09-faraday-m2-serving-design.md](../specs/2026-06-09-faraday-m2-serving-design.md)

---

## Development Environment

Same remote loop as M1: author on Windows (`C:\projects\piai`), `git push pi`, run `pytest` in the Pi venv (`~/faraday/.venv`). M2 work goes on a new branch **`m2-serving`** (off `main`). Task 1 re-points the Pi's repo at that branch and installs the new deps. The two llama-servers must be running on the Pi for the integration test (`bash scripts/30_run_servers.sh`).

## File Structure (delta from M1)

```
src/faraday/
  events.py        # NEW: SourcesEvent, TokenEvent, DoneEvent, ErrorEvent, Event
  server.py        # NEW: FastAPI app (/, /healthz, /chat SSE) + helpers
  static/
    index.html     # NEW: minimal streaming chat (inline JS + CSS)
  llm_client.py    # MODIFY: +stream() on protocol + HttpLLMClient + _tokens_from_sse helper
  rag.py           # MODIFY: +answer_stream()
  cli.py           # MODIFY: +serve command
tests/
  conftest.py            # MODIFY: FakeLLM.stream()
  test_llm_stream.py     # NEW
  test_rag_stream.py     # NEW
  test_server.py         # NEW
  test_integration_pi.py # MODIFY: + streaming /chat case
scripts/
  60_run_app.sh    # NEW: servers + faraday serve
pyproject.toml     # MODIFY: +fastapi, +uvicorn
```

---

### Task 1: M2 setup — branch, deps, Pi deploy

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the M2 branch** `[on dev machine]`

Run: `git checkout -b m2-serving`
Expected: `Switched to a new branch 'm2-serving'`

- [ ] **Step 2: Add FastAPI + uvicorn to dependencies**

In `pyproject.toml`, change the `dependencies` list to add the two packages:

```toml
dependencies = [
  "httpx>=0.27",
  "sqlite-vec>=0.1.3",
  "pypdf>=4.0",
  "typer>=0.12",
  "fastapi>=0.115",
  "uvicorn>=0.30",
]
```

- [ ] **Step 3: Commit, push the branch, re-point the Pi at it**

```bash
git add pyproject.toml
git commit -m "chore(m2): add fastapi + uvicorn deps; start m2-serving"
git push pi m2-serving
ssh pi@raspberrypi.local "cd ~/faraday && git checkout m2-serving"
```
Expected: push succeeds; the Pi reports `Switched to branch 'm2-serving'`. (The Pi's `receive.denyCurrentBranch=updateInstead` now tracks `m2-serving`, so future pushes update its working tree.)

- [ ] **Step 4: Install the new deps into the Pi venv + verify**

Run:
```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pip install -q -e '.[dev]' && python -c 'import fastapi, uvicorn' && echo DEPS_OK"
```
Expected: `DEPS_OK`.

---

### Task 2: Streaming LLM client

**Files:**
- Modify: `src/faraday/llm_client.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_llm_stream.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_llm_stream.py`:
```python
from faraday.llm_client import _tokens_from_sse


def test_tokens_from_sse_parses_deltas_and_stops_on_done():
    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        '',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: [DONE]',
        'data: {"choices":[{"delta":{"content":"ignored"}}]}',
    ]
    assert list(_tokens_from_sse(lines)) == ["Hel", "lo"]


def test_tokens_from_sse_skips_empty_and_non_data_lines():
    lines = ['', ': comment', 'data: {"choices":[{"delta":{}}]}',
             'data: {"choices":[{"delta":{"content":"x"}}]}']
    assert list(_tokens_from_sse(lines)) == ["x"]
```

- [ ] **Step 2: Run to confirm failure**

Run: `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_llm_stream.py -v"` (after `git push pi m2-serving`)
Expected: FAIL — cannot import `_tokens_from_sse`.

- [ ] **Step 3: Implement the parser + `stream()`**

In `src/faraday/llm_client.py`, add `import json` and `from typing import Iterator` at the top, add `stream` to the protocol, and add the helper + method:

```python
from __future__ import annotations
import json
from typing import Iterator, Protocol, runtime_checkable
import httpx
from faraday.config import Settings


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, messages: list[dict], max_tokens: int = 512) -> str: ...
    def stream(self, messages: list[dict], max_tokens: int = 512) -> Iterator[str]: ...


def _tokens_from_sse(lines: Iterator[str]) -> Iterator[str]:
    """Parse llama-server's OpenAI-style SSE lines into content tokens."""
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        content = json.loads(data)["choices"][0].get("delta", {}).get("content")
        if content:
            yield content


class HttpLLMClient:
    """Chat completion against a local llama-server (streaming + non-streaming)."""

    def __init__(self, settings: Settings | None = None, timeout: float = 120.0):
        self.settings = settings or Settings()
        self._client = httpx.Client(base_url=self.settings.gen_url, timeout=timeout)

    def complete(self, messages: list[dict], max_tokens: int = 512) -> str:
        resp = self._client.post(
            "/v1/chat/completions",
            json={"messages": messages, "max_tokens": max_tokens, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def stream(self, messages: list[dict], max_tokens: int = 512) -> Iterator[str]:
        with self._client.stream(
            "POST", "/v1/chat/completions",
            json={"messages": messages, "max_tokens": max_tokens, "stream": True},
        ) as resp:
            resp.raise_for_status()
            yield from _tokens_from_sse(resp.iter_lines())

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Add `FakeLLM.stream()`** in `tests/conftest.py`

Add this method to the existing `FakeLLM` class (keep `complete` as-is):
```python
    def stream(self, messages: list[dict], max_tokens: int = 512):
        self.last_messages = messages
        mid = len(self.reply) // 2          # two chunks that rejoin to the exact reply
        yield self.reply[:mid]
        yield self.reply[mid:]
```

- [ ] **Step 5: Run to confirm pass**

Run: `git push pi m2-serving` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_llm_stream.py tests/test_llm_client.py -v"`
Expected: PASS (the new parser tests + the existing `test_fake_llm_satisfies_protocol`, which still holds because `FakeLLM` now has both `complete` and `stream`).

- [ ] **Step 6: Commit**

```bash
git add src/faraday/llm_client.py tests/conftest.py tests/test_llm_stream.py
git commit -m "feat(m2): streaming llm client (stream() + SSE parser)"
```

---

### Task 3: Streaming RAG engine (events + answer_stream)

**Files:**
- Create: `src/faraday/events.py`
- Modify: `src/faraday/rag.py`
- Test: `tests/test_rag_stream.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_rag_stream.py`:
```python
from faraday.rag import RagEngine
from faraday.retriever import Retriever
from faraday.index_store import SqliteVecStore
from faraday.models import Chunk
from faraday.events import SourcesEvent, TokenEvent, DoneEvent


def _store(tmp_path, embedder):
    store = SqliteVecStore(str(tmp_path / "s.sqlite"), dim=embedder.dim)
    chunks = [Chunk(doc_id="d", ord=0, text="The Pi 4 has 4GB RAM.", source="pi.txt"),
              Chunk(doc_id="d", ord=1, text="It uses an ARM CPU.", source="pi.txt")]
    store.add_chunks(chunks, embedder.embed([c.text for c in chunks]))
    return store


def test_answer_stream_emits_sources_tokens_then_done(tmp_path, fake_embedder, make_llm):
    store = _store(tmp_path, fake_embedder)
    engine = RagEngine(Retriever(fake_embedder, store), make_llm("Answer [1]."), top_k=2)
    events = list(engine.answer_stream("how much ram?"))
    store.close()

    assert isinstance(events[0], SourcesEvent)
    assert len(events[0].sources) == 2
    tokens = [e.text for e in events if isinstance(e, TokenEvent)]
    assert "".join(tokens) == "Answer [1]."          # exact reconstruction
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].cited_indices == [1]
    assert events[-1].invalid_citations == []


def test_answer_stream_done_flags_hallucinated_citation(tmp_path, fake_embedder, make_llm):
    store = _store(tmp_path, fake_embedder)
    engine = RagEngine(Retriever(fake_embedder, store), make_llm("Nope [9]."), top_k=2)
    events = list(engine.answer_stream("q"))
    store.close()
    assert events[-1].invalid_citations == [9]
```

- [ ] **Step 2: Run to confirm failure**

Run: `git push pi m2-serving` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_rag_stream.py -v"`
Expected: FAIL — cannot import `faraday.events` / `answer_stream` missing.

- [ ] **Step 3: Implement `src/faraday/events.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from faraday.models import RetrievedChunk


@dataclass(frozen=True)
class SourcesEvent:
    sources: list[RetrievedChunk]


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class DoneEvent:
    cited_indices: list[int]
    invalid_citations: list[int]


@dataclass(frozen=True)
class ErrorEvent:
    message: str


Event = SourcesEvent | TokenEvent | DoneEvent | ErrorEvent
```

- [ ] **Step 4: Add `answer_stream()` to `src/faraday/rag.py`**

Add the import and the method (keep the existing `answer()`):
```python
from typing import Iterator
from faraday.events import Event, SourcesEvent, TokenEvent, DoneEvent
```
```python
    def answer_stream(self, query: str) -> Iterator[Event]:
        sources = self.retriever.search(query, k=self.top_k)
        yield SourcesEvent(sources)
        messages = build_messages(query, sources)
        parts: list[str] = []
        for token in self.llm.stream(messages, max_tokens=self.max_tokens):
            parts.append(token)
            yield TokenEvent(token)
        valid, invalid = classify_citations("".join(parts), n_sources=len(sources))
        yield DoneEvent(cited_indices=valid, invalid_citations=invalid)
```

- [ ] **Step 5: Run to confirm pass**

Run: `git push pi m2-serving` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_rag_stream.py -v"`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add src/faraday/events.py src/faraday/rag.py tests/test_rag_stream.py
git commit -m "feat(m2): streaming rag engine (events + answer_stream)"
```

---

### Task 4: FastAPI server

**Files:**
- Modify: `src/faraday/index_store.py` (relax the sqlite thread check)
- Create: `src/faraday/server.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_server.py`:
```python
from fastapi.testclient import TestClient
from faraday import server
from faraday.rag import RagEngine
from faraday.retriever import Retriever
from faraday.index_store import SqliteVecStore
from faraday.models import Chunk


def _engine(tmp_path, fake_embedder, make_llm, reply):
    store = SqliteVecStore(str(tmp_path / "srv.sqlite"), dim=fake_embedder.dim)
    chunks = [Chunk(doc_id="d", ord=0, text="The Pi 4 has 4GB RAM.", source="pi.txt"),
              Chunk(doc_id="d", ord=1, text="It uses an ARM CPU.", source="pi.txt")]
    store.add_chunks(chunks, fake_embedder.embed([c.text for c in chunks]))
    return RagEngine(Retriever(fake_embedder, store), make_llm(reply), top_k=2), store


def test_healthz_ok():
    client = TestClient(server.app)
    assert client.get("/healthz").status_code == 200


def test_chat_streams_sse_events(tmp_path, monkeypatch, fake_embedder, make_llm):
    engine, store = _engine(tmp_path, fake_embedder, make_llm, "Answer [1].")
    monkeypatch.setattr(server, "make_engine", lambda settings: (engine, store))
    monkeypatch.setattr(server, "_preflight_ok", lambda settings: True)
    client = TestClient(server.app)

    body = client.get("/chat", params={"q": "how much ram?"}).text
    assert "event: sources" in body
    assert "event: token" in body
    assert "event: done" in body
    assert "Answer [1]." in body          # the streamed token text appears


def test_chat_503_when_servers_down(monkeypatch):
    monkeypatch.setattr(server, "_preflight_ok", lambda settings: False)
    client = TestClient(server.app)
    assert client.get("/chat", params={"q": "x"}).status_code == 503
```

- [ ] **Step 2: Run to confirm failure**

Run: `git push pi m2-serving` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_server.py -v"`
Expected: FAIL — cannot import `faraday.server`.

- [ ] **Step 3: Relax the sqlite thread check, then implement the server**

First, in `src/faraday/index_store.py`, change the connection line so the store can be opened and closed across threads — FastAPI iterates the streaming generator in a threadpool that may switch worker threads between `next()` calls, and the default would raise *"SQLite objects created in a thread can only be used in that same thread."* We use the connection serially (no concurrent access), so relaxing the check is safe:

```python
        self.db = sqlite3.connect(path, check_same_thread=False)
```

Then create `src/faraday/server.py`:

```python
from __future__ import annotations
import json
from pathlib import Path
import httpx
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from faraday.config import Settings
from faraday.embedder import HttpEmbedder
from faraday.events import SourcesEvent, TokenEvent, DoneEvent, ErrorEvent
from faraday.index_store import SqliteVecStore
from faraday.llm_client import HttpLLMClient
from faraday.rag import RagEngine
from faraday.retriever import Retriever

app = FastAPI(title="Faraday")
STATIC = Path(__file__).parent / "static"


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
        try:
            for ev in engine.answer_stream(q):
                yield _format(ev)
        except Exception as exc:                      # mid-stream failure
            yield _format(ErrorEvent(str(exc)))
        finally:
            store.close()

    return StreamingResponse(gen(), media_type="text/event-stream")
```

- [ ] **Step 4: Run to confirm pass**

Run: `git push pi m2-serving` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_server.py -v"`
Expected: PASS (3 tests). *(The `/chat` test injects fakes via `monkeypatch` and skips the real preflight; `TestClient` collects the full streamed body into `.text`.)*

- [ ] **Step 5: Commit**

```bash
git add src/faraday/index_store.py src/faraday/server.py tests/test_server.py
git commit -m "feat(m2): fastapi server (/, /healthz, /chat SSE)"
```

---

### Task 5: Web UI

**Files:**
- Create: `src/faraday/static/index.html`
- Test: `tests/test_server.py` (add one test)

- [ ] **Step 1: Add the failing test** `[on dev machine]`

Append to `tests/test_server.py`:
```python
def test_index_page_served():
    client = TestClient(server.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "EventSource" in r.text and "Faraday" in r.text
```

- [ ] **Step 2: Run to confirm failure**

Run: `git push pi m2-serving` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_server.py::test_index_page_served -v"`
Expected: FAIL — `index.html` does not exist (`FileNotFoundError`).

- [ ] **Step 3: Implement `src/faraday/static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Faraday</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.4rem; }
  #q { width: 78%; padding: .5rem; } button { padding: .5rem 1rem; }
  #answer { white-space: pre-wrap; margin: 1rem 0; min-height: 3rem; line-height: 1.5; }
  #sources { color: #555; font-size: .9rem; }
  .bad { color: #c00; }
</style>
</head>
<body>
  <h1>Faraday — private RAG appliance</h1>
  <form id="f"><input id="q" placeholder="Ask about your documents..." autofocus autocomplete="off">
  <button>Ask</button></form>
  <div id="answer"></div>
  <div id="sources"></div>
<script>
const f = document.getElementById('f'), q = document.getElementById('q');
const answer = document.getElementById('answer'), sources = document.getElementById('sources');
f.onsubmit = (e) => {
  e.preventDefault();
  if (!q.value.trim()) return;
  answer.textContent = ''; sources.textContent = '';
  const es = new EventSource('/chat?q=' + encodeURIComponent(q.value));
  es.addEventListener('sources', (ev) => {
    const s = JSON.parse(ev.data);
    sources.textContent = s.length
      ? 'Sources: ' + s.map(x => `[${x.n}] ${x.source} (${x.score})`).join('   ')
      : 'No matching sources.';
  });
  es.addEventListener('token', (ev) => { answer.textContent += JSON.parse(ev.data).text; });
  es.addEventListener('done', (ev) => {
    const d = JSON.parse(ev.data);
    if (d.invalid && d.invalid.length) {
      const span = document.createElement('span');
      span.className = 'bad';
      span.textContent = `  ! hallucinated citations: ${d.invalid.join(', ')}`;
      sources.appendChild(span);
    }
    es.close();
  });
  es.addEventListener('error', () => {
    const span = document.createElement('span');
    span.className = 'bad'; span.textContent = ' [stream error]';
    answer.appendChild(span); es.close();
  });
};
</script>
</body>
</html>
```

- [ ] **Step 4: Run to confirm pass**

Run: `git push pi m2-serving` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_server.py -v"`
Expected: PASS (4 tests in the file now).

- [ ] **Step 5: Commit**

```bash
git add src/faraday/static/index.html tests/test_server.py
git commit -m "feat(m2): minimal streaming web chat UI"
```

---

### Task 6: CLI `serve` command + run script

**Files:**
- Modify: `src/faraday/cli.py`
- Create: `scripts/60_run_app.sh`
- Test: `tests/test_cli.py` (add one test)

- [ ] **Step 1: Add the failing test** `[on dev machine]`

Append to `tests/test_cli.py`:
```python
def test_cli_help_lists_serve():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.output
```

- [ ] **Step 2: Run to confirm failure**

Run: `git push pi m2-serving` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_cli.py::test_cli_help_lists_serve -v"`
Expected: FAIL — `serve` not in help output.

- [ ] **Step 3: Add the `serve` command to `src/faraday/cli.py`**

Append this command (after `ask`):
```python
@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000):
    """Run the Faraday web app (FastAPI + uvicorn)."""
    import uvicorn
    uvicorn.run("faraday.server:app", host=host, port=port)
```

- [ ] **Step 4: Run to confirm pass**

Run: `git push pi m2-serving` then `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_cli.py -v"`
Expected: PASS (both CLI tests).

- [ ] **Step 5: Create `scripts/60_run_app.sh`**

```bash
#!/usr/bin/env bash
# Run ON the Raspberry Pi. Ensures the llama-servers are up, then serves the
# Faraday web app on 0.0.0.0:8000 (reachable from the dev machine's browser).
set -euo pipefail
cd "$HOME/faraday"
bash scripts/30_run_servers.sh
# shellcheck disable=SC1091
. .venv/bin/activate
exec faraday serve --host 0.0.0.0 --port 8000
```

- [ ] **Step 6: Commit**

```bash
git add src/faraday/cli.py scripts/60_run_app.sh tests/test_cli.py
git commit -m "feat(m2): faraday serve command + run script"
```

---

### Task 7: End-to-end streaming integration test (on the Pi)

**Files:**
- Modify: `tests/test_integration_pi.py`

- [ ] **Step 1: Add the streaming integration test** `[on dev machine]`

Append to `tests/test_integration_pi.py`:
```python
import json as _json


def _collect_tokens(sse_body: str) -> str:
    """Join the text of all `event: token` frames in an SSE body."""
    out = []
    lines = sse_body.splitlines()
    for i, line in enumerate(lines):
        if line == "event: token" and i + 1 < len(lines) and lines[i + 1].startswith("data:"):
            out.append(_json.loads(lines[i + 1][len("data:"):].strip())["text"])
    return "".join(out)


@pytest.mark.integration
def test_chat_endpoint_streams_grounded_answer(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from faraday import server

    db = str(tmp_path / "srv.sqlite")
    monkeypatch.setenv("FARADAY_DB", db)
    s = Settings.from_env()
    store = SqliteVecStore(db, dim=s.embed_dim)
    ingest("examples/corpus", store=store, embedder=HttpEmbedder(s))
    store.close()

    client = TestClient(server.app)
    body = client.get("/chat", params={"q": "How much RAM can a Raspberry Pi 4 have?"}).text
    answer = _collect_tokens(body)
    print("\nSTREAMED ANSWER:", answer)
    assert "event: done" in body
    assert "8gb" in answer.lower() or "8 gb" in answer.lower()
```

- [ ] **Step 2: Run on the Pi with servers up**

Run:
```bash
git push pi m2-serving
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && bash scripts/30_run_servers.sh >/dev/null && bash scripts/40_smoke_test.sh >/dev/null 2>&1 && pytest -m integration -v -s"
```
Expected: `test_chat_endpoint_streams_grounded_answer PASSED` (the streamed answer mentions 8GB). If the 1.5B model phrases it without a literal "8GB", read the printed answer and loosen the assertion to the fact actually stated.

- [ ] **Step 3: Manual browser check (the demo)** `[on Pi]` + `[on dev machine]`

Run on the Pi: `ssh pi@raspberrypi.local "cd ~/faraday && bash scripts/60_run_app.sh"` (leave running), then open `http://raspberrypi.local:8000` in your browser, type a question, and watch the answer stream in with sources. **This is the M2 done state.** Ctrl-C to stop.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_pi.py
git commit -m "test(m2): end-to-end streaming /chat integration test"
```

---

## Final verification

- [ ] Full suite green on the Pi: `pytest -m 'integration or not integration' -q` (M1 + M2 tests) and `ruff check src tests`.
- [ ] `faraday serve` → browser at `http://raspberrypi.local:8000` streams a grounded, cited answer, fully offline.
- [ ] The CLI (`faraday ingest` / `faraday ask`) still works unchanged.

## Plan done criteria

A browser-based, token-streaming chat over the air-gapped RAG core, with live source citations and the citation guardrail — built additively on M1, fully tested, CLI intact. Then: finish the branch (merge to `main`), and the M2 work is ready to push to GitHub alongside M0/M1.

## Deferred (unchanged from spec)

- **M4:** GBNF grammar-constrained citations.
- **M5:** systemd auto-start, Docker, security hardening.
- **Later:** OpenAI-compatible grounded endpoint, web-based ingestion.
