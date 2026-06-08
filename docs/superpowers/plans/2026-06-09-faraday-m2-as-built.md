# Faraday — M2 As-Built & Findings

**Status:** ✅ M2 complete — merged to `main` and on GitHub
**Plan:** [2026-06-09-faraday-m2-serving.md](./2026-06-09-faraday-m2-serving.md)
**Spec:** [2026-06-09-faraday-m2-serving-design.md](../specs/2026-06-09-faraday-m2-serving-design.md)

The streaming web serving layer, built test-first and **additively** over M1.

## Delivered

A token-streaming web chat over the air-gapped RAG core:

- `GET /chat?q=…` streams Server-Sent Events: `sources` → `token`×N → `done` (plus `error`).
- A self-contained vanilla-JS page (browser `EventSource`); `faraday serve` (uvicorn) launches it; `scripts/60_run_app.sh` brings up the llama-servers + the app.
- Reuses the M1 `RagEngine` **untouched** — `answer_stream()` (~9 lines) + a streaming `LLMClient.stream()` are the only new logic. The CLI is unchanged.
- **32 tests** (30 unit with fakes + 2 hardware integration), ruff clean. ~360 lines added.

## Proven on hardware (offline, streaming)

```
GET /chat?q=How much RAM can a Raspberry Pi 4 have?
event: sources → [{ "n": 1, "source": "pi-facts.md", "score": 0.79 }]
event: token   → "According" · "to" · "the" · …   (streamed live)
event: done    → { "cited": [1], "invalid": [] }
```
Answer: *"…a Raspberry Pi 4 Model B can have 1GB, 2GB, 4GB, or 8GB of RAM … up to 8GB."* — grounded, cited, citation-verified.

## Design notes & what review / tests caught

1. **Stream-optimistically-then-confirm.** Citations can only be verified after the *full* answer exists, yet streaming shows text early. Resolved with the three-event protocol: `sources` after retrieval, raw `token`s live, a final `done` carrying the verification verdict.
2. **Cross-thread sqlite (caught in plan review).** FastAPI iterates the streaming generator in a threadpool that may switch worker threads between `next()` calls; `sqlite3`'s thread-affinity would crash mid-stream. Fixed with `check_same_thread=False` — safe because access is serial.
3. **Test assertion bug (caught in execution).** `FakeLLM.stream` splits its reply into two chunks, so the full answer string is never contiguous in the SSE body — the server test asserts the `done` event's `"cited": [1]` instead.

## Dev-loop note

Same develop-against-the-Pi flow as M1; the only setup wrinkle was re-pointing the Pi's repo at the new `m2-serving` branch (it was still on the now-deleted `m0-m1-rag-core`).

## The architecture payoff

M2 added a whole streaming web product in ~360 lines because M1's boundaries held: the LLM behind a `Protocol`, the RAG flow in a focused `RagEngine`. Streaming became a small additive change, not a rewrite, and the same fakes tested it with zero servers.

## Deferred (unchanged)

- **M4:** GBNF grammar-constrained citations (structured inference control).
- **M5:** systemd auto-start, Docker, security hardening.
- **Later:** OpenAI-compatible grounded endpoint, web-based ingestion.

**Next:** M3 — Prometheus metrics + a Grafana dashboard (tok/s, TTFT, RAM).
