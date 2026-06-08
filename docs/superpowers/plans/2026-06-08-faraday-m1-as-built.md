# Faraday — M1 As-Built & Findings

**Status:** ✅ M1 complete — merged to `main`
**Plan:** [2026-06-08-faraday-m0-m1-rag-core.md](./2026-06-08-faraday-m0-m1-rag-core.md) (Milestone M1)

The RAG core, built test-first. What was delivered, what the tests caught, and how the remote dev loop worked.

## Delivered

A fully-offline document-Q&A pipeline + CLI:
`ingest → chunk → embed → sqlite-vec → retrieve → ground → generate → verify-citations`

- `faraday ingest <dir>` and `faraday ask "<q>"` (typer CLI).
- **21 unit tests** (deterministic fakes for the embedder/LLM, no server needed) + **1 hardware integration test**. All green; `ruff` clean.
- Components: `models`, `parsers`, `chunker`, `embedder`, `index_store` (sqlite-vec), `retriever`, `prompt`, `citations`, `llm_client`, `ingest`, `rag`, `cli`.

## Proven on hardware (offline)

```
$ faraday ask "What CPU does the Raspberry Pi 4 use?"
The Raspberry Pi 4 uses a Broadcom BCM2711 with a quad-core ARM Cortex-A72 (64-bit) CPU. [1]
Sources:
  [1] pi-facts.md (score 0.855)
```
Grounded, cited, citation-verified — no network egress.

## What the tests caught (the value of TDD + a real integration test)

1. **Chunker off-by-one.** A sliding window (size=100, overlap=20) over 250 chars emitted a 4th tiny sliver chunk fully contained in chunk #3's tail. Fix: stop once a chunk reaches end-of-text. Caught by a unit test *before* the impl shipped.
2. **`.gitignore` over-broad pattern.** Unanchored `corpus/` also ignored `examples/corpus/`, so the demo corpus never deployed and `ingest` found 0 docs. Caught **only by the integration test** — the one exercising real deploy + filesystem. Fix: anchor to `/corpus/`.
3. **sqlite-vec KNN shape.** Used a subquery (`MATCH` + `LIMIT`) joined to the metadata table, rather than an inline JOIN, to reliably satisfy vec0's KNN-bound requirement.

## The remote dev loop

Develop-against-the-Pi: author tests + code on Windows → `git push pi` → `pytest` in the Pi venv. Because the deploy is a git push, every test ran against an exact committed SHA. Strict per-step red→green was adapted to **test+impl-per-commit (verified green)** to suit the remote loop, with clean per-module commits preserved.

## M1 task completion

- [x] 5 scaffold · [x] 6 models · [x] 7 parsers · [x] 8 chunker · [x] 9 embedder
- [x] 12 prompt · [x] 13 citations · [x] 14 llm_client · [x] 10 store · [x] 11 retriever
- [x] 15 ingest · [x] 16 rag · [x] 17 cli · [x] 18 integration (on hardware)

**Next:** M2 — serving (FastAPI + SSE), a minimal web UI, and GBNF grammar-structured citations.
