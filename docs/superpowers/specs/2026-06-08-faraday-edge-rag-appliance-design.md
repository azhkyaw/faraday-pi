# Faraday — Private RAG Appliance on Raspberry Pi 4
## Design Document

| | |
|---|---|
| **Status** | Approved (brainstorming) — ready for implementation planning |
| **Date** | 2026-06-08 |
| **Author** | azhk.dev@gmail.com |
| **Working name** | `Faraday` (a Faraday cage blocks signal egress — the privacy metaphor *is* the pitch) |

---

## 1. Overview

**Faraday** is a fully air-gapped personal RAG (Retrieval-Augmented Generation) appliance running entirely on a Raspberry Pi 4. A user points it at their own documents (PDF / Markdown / HTML / text); Faraday ingests them locally, and answers natural-language questions with **cited** answers — with **zero network egress**. No cloud, no API calls, nothing leaves the device at serve time.

The project is deliberately two things at once:

1. **A product** — a polished, demoable, privacy-first GenAI appliance whose value proposition *requires* the edge.
2. **An inference-engineering study** — a rigorous, reproducible body of measurement and optimization that treats the Pi 4 as a systems constraint to engineer against, not just a host.

## 2. Portfolio framing (what this proves)

This is a portfolio piece for an **AI / GenAI / LLM Engineer** role with an edge / on-device / inference-systems lean. It is engineered to demonstrate, with evidence:

- **Applied GenAI**: end-to-end RAG (chunking, embeddings, retrieval, grounding, citation, evaluation).
- **Inference engineering**: quantization, KV-cache tuning, serving, throughput/latency optimization on constrained hardware.
- **Production thinking**: observability, robustness, testing, reproducibility.
- **Senior judgment**: honest evaluation (every speed number paired with a *measured quality* number), and clear-eyed reporting of limits.

The headline differentiator versus the thousands of "I put an LLM on a Pi" posts is **measurement and optimization** — see §9.

## 3. Goals and non-goals

**Goals**
- Answer questions about a user's private document corpus, fully offline, with source citations.
- Run generation + embeddings on-device via a self-hosted, OpenAI-compatible API.
- Produce a reproducible benchmark + evaluation harness and a written technical report.
- Be clonable and runnable by a recruiter from a clean repo with minimal steps.

**Non-goals (YAGNI — explicitly out of scope)**
- Multimodality (vision / voice). Text-only; no peripherals. (Candidate v2.)
- Multi-user accounts, auth, or hosting for others. Single-user appliance.
- Training / fine-tuning a model. We *use and optimize* off-the-shelf small models. (Candidate v2.)
- A beautiful, framework-heavy UI. The UI is minimal by design; the engineering is the point.
- Beating cloud LLM quality. The point is *how much capability fits on the edge*, measured honestly.

## 4. Constraints and assumptions

- **Hardware**: Raspberry Pi 4, **4GB RAM (confirmed)**. This is the tighter, more common variant — and the constraint *is* the story: fitting a useful RAG appliance in 4GB is a stronger engineering signal than doing it with headroom. The default generation model is therefore a **1.5B-class** model (comfortable in 4GB); larger 3B models are explored in the sweep to chart exactly where the 4GB wall hits. Peak RAM is recorded per cell, making the published leaderboard directly useful to the large population of ≤4GB single-board computers.
- **No GPU / no CUDA**: ARM Cortex-A72 CPU inference only. Local models are small and quantized.
- **Text-only**: no camera/mic/sensors. The product is document Q&A.
- **Offline at serve time**: the appliance makes no network calls when answering. (Eval tooling may use a cloud LLM-judge — at *eval time only*, off the serving path; see §9/§10.)
- **Effort**: uncapped; optimize for standout. Staged so each milestone is independently demoable.

## 5. Architecture

Five independently-testable layers. Runtime layers (1–5) ship in the appliance; the **Lab** (§9) is a separate harness that exercises them.

```
┌──────────────────────────────────────────────────────────┐
│  Interface:  local web chat (SSE)  +  OpenAI-compat API   │
├──────────────────────────────────────────────────────────┤
│  RAG Orchestrator (FastAPI): embed→search→ground→stream   │
│                              citation-verify + guardrail   │
├───────────────────────────────┬──────────────────────────┤
│  Retrieval                    │  Inference (llama.cpp)     │
│  sqlite-vec (single-file DB)  │  llama-server (OpenAI API) │
│  chunks + vectors + metadata  │  gen LLM  +  embed model   │
├───────────────────────────────┴──────────────────────────┤
│  Ingestion (batch): parse → chunk → embed → index         │
├──────────────────────────────────────────────────────────┤
│  Observability: Prometheus + Grafana (tok/s, TTFT, RAM)   │
└──────────────────────────────────────────────────────────┘
       all on-device · Raspberry Pi OS Lite (64-bit) · no GPU
```

## 6. Components

Each component has one clear purpose, a defined interface, and explicit dependencies — so it can be understood, replaced, and tested independently.

### 6.1 `ingest` (batch CLI/module)
- **Purpose**: turn a folder of documents into a searchable index.
- **Interface**: `ingest(source_dir, collection) -> IndexStats`; CLI: `faraday ingest ./docs --collection mydocs`.
- **Depends on**: `parsers`, `chunker`, `embedder`, `index_store`.
- **Sub-units**:
  - `parsers` — PDF/Markdown/HTML/text → plain text (+ page/section metadata).
  - `chunker` — text → chunks (~512 tokens, configurable overlap/strategy).
  - `embedder` — chunks → vectors via the llama-server embeddings endpoint (batched).
  - `index_store` — read/write vectors + chunk text + metadata to `sqlite-vec`. Incremental (content-hash skip), idempotent.

### 6.2 `retriever`
- **Purpose**: given a query embedding, return the top-k relevant chunks with metadata.
- **Interface**: `search(query_vec, k, filters) -> list[Chunk]`.
- **Depends on**: `sqlite-vec`, `embedder`.

### 6.3 `orchestrator` (FastAPI app — the brain + serving surface)
- **Purpose**: the RAG control flow and the HTTP API.
- **Interface (HTTP)**: `POST /chat` (SSE stream of answer + sources) and OpenAI-compatible `POST /v1/chat/completions` — **both RAG-grounded** (retrieval + citations), so editors/tools can consume the appliance via a standard client; `GET /healthz`; `GET /metrics`. Raw, *ungrounded* model access (for benchmarking) goes directly to the llama-server instance, not through this API.
- **Depends on**: `retriever`, `embedder`, `llm_server`, `prompt_builder`, `citation_verifier`, `llm_client`.
- **Sub-units**:
  - `prompt_builder` — assembles grounded prompt: retrieved context + instructions + a **GBNF citation grammar** forcing valid JSON citations.
  - `llm_client` — streams tokens from llama-server.
  - `citation_verifier` — validates that every cited chunk ID exists and was actually retrieved (anti-hallucination guardrail).

### 6.4 `llm_server` (deployment unit, not our code)
- **Purpose**: serve the generation model and the embedding model over an OpenAI-compatible API.
- **Implementation**: `llama.cpp`'s `llama-server`, built from source with ARM NEON / optimization flags. Because one process serves one model, generation and embeddings run as **two separate `llama-server` instances** (the embedding instance launched with `--embedding`); both sit behind the orchestrator. Config-managed: model path, quant, thread count, KV-cache settings, grammar support. The bench harness may also hit a raw generation instance directly (bypassing retrieval) for model-only benchmarks.

### 6.5 `web_ui` (minimal SPA)
- **Purpose**: chat with streaming tokens, show source citations, select a collection.
- **Interface**: talks only to the `orchestrator`. Default stack: HTMX (or a small Svelte build) served statically.

### 6.6 `metrics` (observability)
- **Purpose**: live operational visibility.
- **Implementation**: Prometheus exporter embedded in the orchestrator + llama-server's own metrics; Grafana dashboards provisioned as JSON. Tracks tok/s (prefill vs decode), TTFT, RAM (RSS), retrieval latency, queue depth, CPU temp.

### 6.7 `bench` (the Lab — separate from runtime)
- **Purpose**: run reproducible sweeps and evaluations; produce CSVs + plots.
- **Interface (CLI)**: `faraday-bench sweep --models ... --quants ... --out results/`; `faraday-bench eval-rag --dataset eval/qa.jsonl`.
- **Depends on**: `llama-bench`, `llama_server`, eval datasets, an LLM-judge client (cloud, eval-time only).
- **Sub-units**: `sweep_runner`, `quality_eval` (perplexity + judge), `rag_eval` (recall / citation / answer), `plots`.

## 7. Data flows

**Query path:**
`question → embed(query) → sqlite-vec top-k → prompt_builder (context + citation grammar) → stream from llama-server → citation_verifier → SSE answer + sources → log metrics`

**Ingest path:**
`docs/ → parse → chunk → embed (batched) → write to sqlite-vec (incremental, hash-skip) → IndexStats`

## 8. Tech stack

| Layer | Choice | Rationale (edge-appropriate) |
|---|---|---|
| OS | Raspberry Pi OS Lite (64-bit) | Headless, ARM64, minimal RAM overhead |
| Inference runtime | **`llama.cpp`** (source build, NEON flags) | Full control of quant, KV-cache, grammars — the levers an inference engineer is hired to pull |
| Generation model | **Qwen2.5-1.5B-Instruct** GGUF @ Q4_K_M (default) | Reliable in 4GB with room for KV-cache + embeddings; sweep still covers 1B/1.5B/3B × Q8→Q2, with 3B as the frontier-ceiling model |
| Embedding model | `bge-small-en-v1.5` (or `nomic-embed-text`) | Tiny, high-quality, runs on-device |
| Vector DB | **`sqlite-vec`** | Single file, zero daemon, survives power-cycle — the edge-native choice |
| Orchestrator / API | **FastAPI** (async, SSE) | Clean OpenAI-compatible surface, streaming |
| Structured output | llama.cpp **GBNF grammars** | Forces valid JSON citations — advanced inference-control showcase |
| UI | Minimal SPA (HTMX default) | UI is not the point; keep it lean |
| Observability | Prometheus + Grafana | Run on-box or push off-box |
| Packaging | Docker Compose + systemd + one-shot bootstrap | Reproducible, always-on, recruiter-clonable |
| Baseline (comparison only) | **Ollama** | A managed default to benchmark hand-tuning against |

## 9. The rigor — experimental methodology (portfolio gold)

Six studies. Each produces committed CSVs + plots and a section of the written report.

1. **Quantization sweep → the headline chart.** For each candidate model (Qwen2.5-1.5B/3B, Llama-3.2-1B/3B, optionally Gemma-2-2B), build GGUF at Q8_0 → Q6_K → Q5_K_M → Q4_K_M → Q3_K_M → Q2_K. Measure per cell: disk size, peak RAM (RSS), prefill tok/s, decode tok/s, TTFT (via `llama-bench` + custom end-to-end timing). On 4GB, higher-quant 3B cells are expected to hit a hard RAM wall (OOM) — **charting that wall is itself a headline finding**, not a gap. **Output**: the quality-vs-footprint frontier plot with the sweet-spot knee and the 4GB ceiling marked.
2. **Quality eval.** Perplexity (wikitext) per quant + task quality on a held-out QA set scored by an **LLM-judge** (1–5 correctness + faithfulness). Judge runs via a cloud API **at eval time only**; the appliance stays air-gapped.
3. **RAG-specific eval.** Label ~50–100 `(question, gold-answer, gold-source)` triples → measure retrieval **recall@k**, **citation accuracy**, **end-to-end answer quality**. Ablate chunk size, top-k, embedding model, rerank on/off — each ablation a table row with a takeaway.
4. **Optimization study.** Before/after deltas for each lever: thread count, CPU governor + safe overclock, KV-cache quantization, flash-attention, mmap vs load, batch/ubatch sizes, prompt/prefix caching, and **speculative decoding** (a ~0.5B draft model for the 1.5B target) as the showpiece. Under 4GB, both draft and target must stay resident, so whether spec-decoding's speedup survives the added memory pressure is itself a finding. **Output**: a throughput waterfall ("baseline → +X → +Y → final").
5. **Sustained-load & thermal reality.** The Pi 4 throttles when hot. Measure tok/s over a 30-min sustained run, with/without a heatsink, logging CPU temp. Report **steady-state vs burst**.
6. **Energy (optional stretch).** A ~$10 inline USB power meter → **tokens/sec per watt**. Flagged optional (a meter, not a build peripheral).

## 10. Evaluation datasets & metric definitions

**Datasets**
- **Corpus**: a public document set (e.g., a fixed set of papers/manuals) for reproducibility, plus the user's own docs for the demo.
- **QA eval set**: ~50–100 `(question, gold-answer, gold-source)` triples over the public corpus.
- **Abstention set**: questions deliberately *unanswerable* from the corpus, to measure the faithfulness guardrail.

**Metric definitions (unambiguous)**
- **TTFT** — time from request received to first generated token streamed.
- **Prefill tok/s** — prompt tokens ÷ prefill time.
- **Decode tok/s** — generated tokens ÷ (total time − prefill time).
- **Peak RAM** — max RSS of llama-server during a fixed workload.
- **recall@k** — fraction of eval questions whose gold-source chunk appears in the top-k retrieved.
- **citation accuracy** — fraction of answer citations that point to a chunk actually supporting the claim (judged).
- **answer quality** — LLM-judge score (1–5) on correctness + faithfulness, averaged.
- **abstention rate** — on the abstention set, fraction where the system correctly declines to answer.

## 11. Error handling & robustness

- **Faithfulness guardrail**: low retrieval confidence → "not in your documents" (behavior is *measured* via the abstention set, not hoped).
- **OOM protection**: memory budgeting, model-size guard at startup, graceful fallback to a smaller quant, zram/swap tuning.
- **Stability**: systemd auto-restart, `GET /healthz`, thermal-throttle awareness (surfaced in metrics).
- **Robust ingestion**: handle malformed PDFs, odd encodings, and duplicates; skip-and-log rather than crash.
- **Streaming**: handle client disconnects / connection drops in the orchestrator without leaking llama-server slots.

## 12. Testing strategy

- **Unit**: `chunker`, `parsers`, `citation_verifier`, `retriever` wrapper — deterministic via a tiny fixture corpus + a stubbed embedder.
- **Integration**: full RAG path on a fixed mini-corpus with golden `question→answer` assertions.
- **Eval-as-test**: a retrieval-recall regression guard in CI when chunking/embedding changes (a real quality gate, with a threshold).
- **Bench reproducibility**: a smoke test that the sweep runs end-to-end on one tiny model.
- **Approach**: TDD where it fits (orchestrator logic, citation verification).

## 13. Deliverables (portfolio artifacts)

- **GitHub repo**: documented, one-command setup; appliance + bench/eval harness; results (CSVs + plots) committed.
- **Technical report / blog post**: *"Engineering a private RAG appliance on a 4GB Raspberry Pi"* — frontier charts, optimization waterfall, eval methodology.
- **Demo GIF**: the appliance answering document questions in airplane mode (the offline flex).
- **Pi-4 quant leaderboard**: a reusable community table ("what runs well on a Pi 4").
- **Architecture diagram + an honest "limits & what's next"** section.

## 14. Milestone plan

Each milestone is an independently demoable checkpoint.

| # | Milestone | Proves |
|---|---|---|
| **M0** | Bring-up: flash OS, build llama.cpp, run llama-server, curl it, baseline tok/s | Foundation works |
| **M1** | Minimal RAG: ingest → sqlite-vec → orchestrator → CLI query with citations | First offline answer |
| **M2** | Serving & UI: FastAPI gateway, streaming, OpenAI-compat passthrough, web chat + sources | It's a product |
| **M3** | Observability: Prometheus + Grafana live metrics | Production thinking |
| **M4** | The Lab: quant sweep + quality/perplexity + frontier charts; RAG evals; optimization study | The rigor |
| **M5** | Polish & ship: report, demo GIF, README, leaderboard, "next steps" | The narrative |

M0–M3 deliver a working product; M4 is the rigor; M5 is the narrative. M4 studies are independently prioritizable/droppable if time-constrained (though effort is uncapped here).

## 15. Risks & mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | 1.5B-class quality too low to be useful | RAG grounding shifts the task from recall to reading provided context (a small-model strength); quant sweep finds the best fit; abstention guardrail; honest reporting |
| R2 | 4GB is tight — 3B won't fit at good quant; KV-cache + long context risk OOM even for 1.5B | Default to 1.5B; cap context length; quantize the KV-cache; memory-budget guard at startup; treat 3B as exploratory and record where it OOMs |
| R3 | Throughput too slow for pleasant UX | Optimization study (speculative decoding, KV-cache, threads); streaming masks latency; framed as an appliance, not a speed race |
| R4 | Thermal throttling skews benchmarks | Control with heatsink; report steady-state; log temps |
| R5 | Eval set too small to generalize | ~50–100 triples; report caveats/confidence honestly |
| R6 | Scope creep across six studies | Ship working product (M0–M3) first; M4 studies independently droppable |

## 16. Open questions / deferred decisions

- **Reranker as a first-class component?** Deferred — start without; add in the M4 ablation if recall is weak.
- **UI framework** (HTMX vs small Svelte) — deferred to M2; HTMX default.
- **Public eval corpus selection** — pick a concrete, license-clean document set during M4.

## 17. Future work / known limits

- Multimodal v2 (camera/mic): local vision Q&A, offline voice assistant.
- Pi 5 or NPU accelerators (Hailo-8, Coral) for a step-change in throughput.
- Fine-tuning / domain-adapting a small model on the corpus.
- Multi-user, auth, and a hardened deployment.

## 18. Decisions made during brainstorming (traceability)

- Showcase: **Edge AI / on-device inference**.
- Pi's role: **Central — edge is the point**.
- Hardware: **Pi 4 (4GB) only, text-only** (no peripherals).
- Effort: **uncapped — optimize for standout**.
- Angle: **Both — product + rigor**.
- Concept: **Approach A — Private RAG appliance backed by an inference lab** (chosen over a pure benchmark leaderboard and a self-hosted copilot).
