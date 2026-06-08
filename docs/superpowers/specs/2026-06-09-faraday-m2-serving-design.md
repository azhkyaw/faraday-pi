# Faraday M2 — Streaming Web Serving Layer
## Design Document

| | |
|---|---|
| **Status** | Approved (brainstorming) — ready for implementation planning |
| **Date** | 2026-06-09 |
| **Milestone** | M2 (extends the [main design spec](./2026-06-08-faraday-edge-rag-appliance-design.md), §6.3 / §14) |
| **Builds on** | M0 + M1 (the RAG core + CLI, on `main`) |

---

## 1. Overview

M2 turns the M1 command-line RAG core into a **streaming web application**: a FastAPI service wraps the existing `RagEngine` and streams answers token-by-token to a minimal browser chat, with live source citations. Nothing about M1's retrieve → ground → generate → verify logic changes — M2 is **purely additive**. The CLI remains a first-class interface; both the CLI and the web service share one `RagEngine`.

The appliance stays air-gapped at serve time: the browser talks to FastAPI on the Pi, which talks to the two local llama-servers. No external network.

## 2. Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Web UI | **Plain HTML + vanilla JS** (browser `EventSource`) | Zero build step, no Node toolchain on the Pi; the UI is intentionally minimal |
| Citations | **Keep M1 regex `[n]`** extract-then-verify | Reads as natural prose, streams cleanly, already tested |
| OpenAI-compatible endpoint | **Deferred** | Keep M2 focused on the web product |
| GBNF grammar citations | **Deferred to M4** | Fiddly grammar + complicates streaming; better as an inference-control study |
| Packaging (systemd/docker) | **Deferred to M5** | M2 is the serving layer, not deployment hardening |

## 3. Goals / non-goals

**Goals**
- Stream a grounded, cited answer to a browser, token-by-token, over SSE.
- Reuse `RagEngine`/`Retriever`/store unchanged; add a streaming path alongside the existing synchronous `answer()`.
- Keep the CLI working (shared engine).
- Full test coverage with fakes (no server needed) + one streaming integration test on the Pi.

**Non-goals (YAGNI)**
- OpenAI-compatible endpoint, GBNF grammars, systemd/docker, web-based ingestion (CLI stays), auth/multi-user, a heavy UI framework.

## 4. Architecture

Additive layer over M1:

```
  browser (static chat page, EventSource)
        │  GET /chat?q=…   (SSE)
        ▼
  FastAPI service (server.py)
        │  RagEngine.answer_stream(q)  →  Sources / Token… / Done events
        ▼
  RagEngine (M1, reused)
    ├─ Retriever → SqliteVecStore + HttpEmbedder   (:8081)
    └─ HttpLLMClient.stream()  →  llama-server gen  (:8080)
```

## 5. Components

New and modified units, each small and independently testable.

### 5.1 `events.py` (new)
Tiny frozen dataclasses for the streaming protocol (`Event` = their union):
- `SourcesEvent(sources: list[RetrievedChunk])`
- `TokenEvent(text: str)`
- `DoneEvent(cited_indices: list[int], invalid_citations: list[int])`
- `ErrorEvent(message: str)` — emitted if generation fails after streaming has begun

### 5.2 `llm_client.py` (modified)
Add streaming to the `LLMClient` protocol and `HttpLLMClient`:
- `stream(messages: list[dict], max_tokens: int) -> Iterator[str]` — POST with `"stream": true`, iterate llama-server's SSE lines (`data: {...}`), extract `choices[0].delta.content`, yield non-empty token strings, stop on `data: [DONE]`.
- The existing non-streaming `complete()` stays (CLI uses it).
- `FakeLLM` (in `conftest.py`) gains a `stream()` that yields its canned reply in chunks, for deterministic tests.

### 5.3 `rag.py` (modified)
Add a streaming method alongside `answer()`:
- `answer_stream(query) -> Iterator[Event]`: retrieve → `yield SourcesEvent` → stream tokens (`yield TokenEvent` each) while accumulating the full text → `classify_citations(full, n_sources)` → `yield DoneEvent`. Reuses `retriever`, `build_messages`, `classify_citations`.

### 5.4 `server.py` (new)
FastAPI application:
- `GET /` — serve the static chat page.
- `GET /healthz` — liveness (and a quick check that the llama-servers are reachable).
- `GET /chat?q=…` — `StreamingResponse(media_type="text/event-stream")`; runs `engine.answer_stream(q)`, formatting each event as `event: <sources|token|done>\ndata: <json>\n\n`.
- Opens the `SqliteVecStore` **per request** (sqlite connections aren't shared across threads; the appliance is single-user, so per-request open is the simplest safe choice). Builds the engine with the streaming `HttpLLMClient`.

### 5.5 `static/index.html` (new)
One self-contained page (inline JS + CSS, no build): a question input, an answer area, a sources panel. JS opens `EventSource('/chat?q=…')` and handles the named events — `sources` (render the panel), `token` (append to the answer), `done` (mark which citations are valid / flag hallucinated ones, then close), and `error` (show the message, close).

### 5.6 `cli.py` (modified)
Add a `faraday serve` command that launches uvicorn on `server.app` (so the appliance starts from the same CLI). The existing `ingest` / `ask` commands are unchanged.

### 5.7 `pyproject.toml` (modified)
Add `fastapi` and `uvicorn` to dependencies.

### 5.8 `scripts/60_run_app.sh` (new)
Convenience: ensure the two llama-servers are up (`30_run_servers.sh`), then launch `faraday serve` (uvicorn) bound to `0.0.0.0:8000` so the dev machine's browser can reach it.

## 6. Data flow (the streaming path)

```
browser EventSource('/chat?q=…')
  └─▶ FastAPI /chat  (sync generator, run in FastAPI's threadpool)
        └─▶ RagEngine.answer_stream(q):
              retrieve (sqlite-vec + embed :8081)  → yield SourcesEvent → SSE  event: sources
              stream gen tokens (Qwen :8080)       → yield TokenEvent…  → SSE  event: token  (live)
              classify_citations(full answer)      → yield DoneEvent    → SSE  event: done
  ◀── browser: render sources panel · append streaming answer · confirm citations on done
```

**Stream-optimistically-then-confirm.** Citations can only be verified *after* the full answer exists (you can't know `[2]` is valid until every marker is extracted), yet streaming shows text before it's complete. The three-event protocol resolves this: `sources` is known right after retrieval; tokens stream live and raw; the final `done` event carries the verification verdict. The browser renders citations provisionally and confirms them on `done`. This is how a fundamentally *batch* guardrail coexists with *streaming* UX.

## 7. Error handling

- **Empty retrieval** → `SourcesEvent([])`; the grounded prompt makes the model abstain ("not in your documents").
- **llama-server unreachable** → `/chat` pre-flight check returns **503** before the stream starts (headers not yet sent); `/healthz` also reports it. A failure *after* streaming has begun emits an `ErrorEvent` and closes.
- **Client disconnect mid-stream** → the generator's cleanup closes the httpx streaming response so no llama-server slot leaks.
- **Store open / query failure** → 500 with a logged reason.

## 8. Testing

- **Unit (fakes, no server):**
  - `HttpLLMClient.stream()` SSE parsing — feed canned `data: {...}` chunks (incl. `[DONE]`) → assert the token sequence.
  - `RagEngine.answer_stream()` — with `FakeLLM.stream()` → assert the event order `SourcesEvent`, `TokenEvent×N`, `DoneEvent`, and that `DoneEvent` carries correct valid/invalid citations.
  - `server.py` via FastAPI `TestClient` — inject fakes (dependency override), `GET /chat?q=…`, assert the SSE body contains the `sources`/`token`/`done` events; `GET /healthz` → 200.
- **Integration (Pi, real models):** `GET /chat` streams a grounded, cited answer end-to-end; assert the concatenated tokens contain the expected fact and citations are valid.
- TDD throughout; ruff clean. Same remote loop: author on Windows → `git push pi` → `pytest` on the Pi.

## 9. File structure (delta from M1)

```
src/faraday/
  events.py        # NEW: SourcesEvent, TokenEvent, DoneEvent
  server.py        # NEW: FastAPI app (/, /healthz, /chat SSE)
  static/
    index.html     # NEW: minimal streaming chat (inline JS/CSS)
  llm_client.py    # +stream()
  rag.py           # +answer_stream()
  cli.py           # +serve command
tests/
  test_llm_stream.py     # NEW
  test_rag_stream.py     # NEW
  test_server.py         # NEW
  test_integration_pi.py # + a streaming case
scripts/
  60_run_app.sh    # NEW: servers + uvicorn
pyproject.toml     # +fastapi, +uvicorn
```

## 10. Definition of done

- `faraday serve` runs the appliance; a browser at `http://raspberrypi.local:8000` streams a grounded, cited answer to a typed question, fully offline.
- All unit tests green (fakes) + the streaming integration test green on the Pi; ruff clean.
- The CLI still works unchanged.

## 11. Decisions deferred to later milestones

- **M4:** GBNF grammar-constrained citations (structured inference control, measured).
- **M5:** systemd auto-start service, Docker packaging, security hardening (disable password SSH, drop NOPASSWD sudo).
- **Later:** OpenAI-compatible grounded endpoint; web-based ingestion / document upload.
