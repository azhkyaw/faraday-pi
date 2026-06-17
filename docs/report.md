# Engineering a private RAG appliance on a 4 GB Raspberry Pi

Faraday is an air-gapped retrieval-augmented-generation appliance: ask questions of your own
documents, get cited answers, entirely offline, on a $55 computer. This report is the
engineering story — the constraint that makes it interesting, the four measurement studies
that turned guesses into numbers, and what it took to ship.

Every numeric claim below links the committed artifact it came from. Nothing here is a vibe.

---

## 1. The constraint

The privacy pitch for RAG — "your documents never leave the room" — only means something if
the hardware is *cheap and self-contained*. A 4 GB Raspberry Pi 4 is the honest test of that
claim: no GPU, ~3.8 GB of usable LPDDR4, a quad-core ARM CPU, and a power supply that will
under-volt if you look at it wrong. If a useful RAG appliance fits *here*, the privacy story
is real; if it needs a workstation, it isn't.

The 4 GB ceiling is the whole game. A generation model, an embedding model, a vector store,
and a web app must coexist in the RAM a single browser tab wastes — and stay fast enough to
talk to. Everything that follows is a consequence of that budget.

## 2. Architecture

Faraday is a conventional RAG pipeline with an unconventional amount of dependency injection:

```
documents → chunk → embed (bge-small) → sqlite-vec store
query → embed → vector search (top_k) → prompt w/ sources → llama-server → cited answer
```

Two llama.cpp servers run **native** on the Pi (generation on :8080, embeddings on :8081);
a FastAPI app (:8000) does retrieval and streams tokens over Server-Sent Events. The design
rule is that every component depends on a `Protocol` (`Embedder`, `LLMClient`), with real
HTTP implementations injected at the edges (`cli.py`/`server.py`) and fakes injected by
tests. That one decision is why **well over a hundred unit tests run with no server and no model**
(only a handful of integration tests need the live stack) — the impure parts (subprocess,
HTTP, `/proc`) are pushed to the boundary, leaving the decision logic pure and testable. Streaming flows through a `Sources → Token → Done` event
seam, which doubles as the metrics instrumentation point (TTFT, decode tok/s, citation
validity all observed there).

The baseline pipeline (M0) measured end-to-end RAG TTFT ~3.25 s, retrieval ~96 ms, decode
~3.6 tok/s — the starting point the four studies below set out to understand and improve.

## 3. Choosing the model — the quality/footprint frontier (M4a)

*Which model and quantization?* We swept **18 cells** — Qwen2.5 {0.5B, 1.5B, 3B} × six
quants {Q8_0…Q2_K} — measuring perplexity, peak RSS, and speed end-to-end on power that held
`get_throttled=0x0` for ~15 hours.
([findings](../results/sweep/findings.md) · [frontier.png](../results/sweep/frontier.png) ·
[leaderboard](../results/sweep/leaderboard.md))

**The verdict: 1.5B at Q4_K_M.** It sits at the *knee* of the perplexity-vs-RSS frontier —
11.32 perplexity, 1.07 GB, 3.86 tok/s — nearly all of the 1.5B model's quality at an
interactive speed and a comfortable footprint. Three findings made the call non-obvious:

- **Decode is memory-bandwidth-bound, and we measured the constant.** Across all 18 cells,
  `decode tok/s × model bytes` lands in a narrow band — **≈3.8 GB/s**, which *is* the board's
  effective LPDDR4 read bandwidth, measured 18 independent ways. The practical gift:
  **`decode ≈ 3.8 GB/s ÷ model_bytes`** lets you size any GGUF for a Pi 4 *without running
  it*.
- **The low-bit cliff arrives earlier as the model grows** — the opposite of the folk wisdom
  that big models compress better. Q8→Q3 degradation: 0.5B +3.1%, 1.5B +12.4%, **3B +48%**.
  The 3B model is flatly broken below Q4_K_M — its Q3/Q2 are *worse than 1.5B-Q4_K_M while
  using more RAM*.
- **Prefill is kernel-bound, not byte-bound** — Q8_0 and Q4_K_M are the two *fastest* prefill
  quants in every size, because their NEON dequant paths are the most optimized. Q4_K_M wins
  twice.

**The 3B verdict — "fits in RAM, fails at interactivity":** every 3B quant ran to completion
(peak RSS ≤3.3 GB, swap untouched), but the only healthy quant decodes at **1.92 tok/s ≈ 86
words/min** — below reading speed. *Can* a 4 GB Pi run a 3B model? Yes. *Should it serve one
interactively?* No. (It remains real frontier for batch/offline jobs.)

## 4. Is it actually good? — RAG quality (M4b + GBNF)

A fast wrong answer is worthless, so we built `faraday.eval`: a record-then-judge harness
that drives the real engine over a 47-question Apollo-corpus golden set with machine-verified
source spans, scores deterministic retrieval metrics on all 9 configs, and runs an LLM judge
at the baseline.
([scorecard](../results/evals/scorecard.md) ·
[ablations.png](../results/evals/ablations.png) · [findings](../results/evals/findings.md))

**Verdict: chunk 1200 / top_k 4** — the shipped defaults, validated. The headline finding is
a trap worth its own sentence: **the largest chunks (2400) have the *worst* recall at every
top_k.** Why? 52% of 2400-char chunks exceed bge-small's hard 512-token limit, so their
embeddings are computed from a truncated view that silently drops the tail — *a chunk size
past the embedder's context is self-defeating*. Top_k shows a clean knee at k4 (k2→k4 buys
+0.146 recall; k4→k8 only +0.049). Answer quality at the baseline: faithfulness **4.22**,
correctness **3.95** (1–5 judge).

**The weak spot was abstention** — the model declined only **2 of 6** out-of-corpus questions,
emitting confident answers where it should have refused. M5 fixed this structurally with
**GBNF grammar-constrained citations**: the grammar is generated *per request* from the
retrieved-source count, so an out-of-range citation like `[5]` (when only 4 sources exist) is
*undecodable* — impossible by construction, not merely discouraged by the prompt. A live
smoke confirmed llama-server's OpenAI endpoint honors the grammar, and the before/after
measurement ([gbnf_before_after.md](../results/evals/gbnf_before_after.md)) shows citation
validity held at **1.000 — now guaranteed** — with recall (0.805) and abstention (0.872)
*unchanged*: the grammar constrains without distorting.

## 5. Making it fast (M4c)

*Can we tune our way to more speed?* We ablated every CPU/inference lever on 1.5B-Q4_K_M —
governor, threads, batch size, KV-cache quant, flash-attention — stacked the winners, and
compared speculative decoding and an Ollama baseline.
([leaderboard](../results/optimize/leaderboard.md) ·
[lever_gains.png](../results/optimize/lever_gains.png) ·
[waterfall.png](../results/optimize/waterfall.png) ·
[context_curve.png](../results/optimize/context_curve.png) ·
[findings](../results/optimize/findings.md))

**The answer is the finding: the appliance already ships at its throughput optimum.** Decode
sits at **3.83–3.91 tok/s no matter what you change** — governor, batch, KV-quant,
flash-attention, *and context depth* all leave it flat, because (per M4a) decode is gated by
RAM bandwidth ÷ model size, and no CPU knob changes the memory bus. The "stacked best" config
(`-t 3`) is actually *net-negative* — it nudges decode by a rounding error while starving
prefill. Two cross-runtime experiments confirm it:

- **Speculative decoding is counterproductive on CPU: 0.942 tok/s — ~4× *slower* — at 21.6%
  draft acceptance.** It wins on GPUs because batch-verifying N draft tokens is nearly free;
  on a bandwidth-bound CPU you pay for draft compute you mostly discard. The right tool, wrong
  machine.
- **Ollama** lands ~4% slower on decode — a different operating point on the same wall, not a
  win.

Prefill *does* respond — it collapses cleanly with context depth (7.74 → 6.22 tok/s from
128 → 4096 tokens), because attention is O(n²) and compute-bound. The lesson for an edge RAG
box: **price long-context requests by prompt tokens, not by question.** M4a and M4c are the
same truth from opposite ends — vary the model and decode tracks `1/bytes`; hold the model
and vary everything else and decode *doesn't move*. The only throughput lever is the model
you choose.

## 6. Shipping it

A study is not an appliance. M5 hardened Faraday into something you can plug in and forget:

- **Always-on via systemd** — three units (gen → embed → app, with dependency ordering)
  survive both crashes and power-cycles with *zero intervention*. Verified live: SIGKILL the
  app → systemd restarts it in ~9 s (`RESTART-OK`); full reboot → all three services
  auto-recover (`BOOT-OK`). This retires the project's oldest gotcha — "nothing auto-starts
  on boot."
- **A startup memory guard** (`ExecStartPre`) checks the model + a 700 MB headroom against
  `MemAvailable` and refuses to start loudly rather than OOM the board — sized from the
  measured peak-RSS-vs-file-size deltas.
- **One-shot bootstrap** — `scripts/bootstrap.sh` takes a fresh Pi to a running, reboot-proof
  appliance in a single idempotent command (deps → build → models → venv → systemd → smoke).
- **A Docker portability artifact** — the *app* containerizes and runs anywhere against the
  Pi's servers; the **llama-servers deliberately stay native** (NEON-tuned, no container RAM
  tax on a 4 GB board). Shipping the judgment, not just the Dockerfile.
- **A retrieval regression gate** — an eval-as-test that fails CI if recall@4 drops below 0.70
  (the M4b baseline minus a margin).

## 7. What to run on a Pi 4

The one-paragraph answer: **Qwen2.5-1.5B at Q4_K_M, top_k 4, chunk 1200, with GBNF citations,
served native under systemd.** It delivers ~3.9 tok/s decode (interactive), recall@4 ~0.80,
faithfulness 4.22/5, guaranteed-valid citations, and survives a power cut — in ~2.4 GB total
resident, leaving headroom on a 4 GB board. Don't run a 3B model interactively (1.9 tok/s),
don't chunk past your embedder's context window (the c2400 trap), and don't reach for
speculative decoding on a CPU. The full measured menu is the [Pi-4 leaderboard](./pi4-leaderboard.md).

## 8. Lessons learned

- **Power is a first-class variable.** The M4a core run under-volted (`get_throttled=0x50000`,
  `Undervoltage detected!` at 55 °C — power, not heat) on a marginal PSU; the official 5.1V/3A
  supply held `0x0` across 15 h. A Pi can't report its input voltage in software — only the
  comparator flag — so: **check `get_throttled` before trusting any benchmark.**
- **Read process RSS, not `free`.** mmap'd model weights hide in `buff/cache`; the prompt
  cache once grew until it *evicted those weights from page cache*, collapsing decode
  3.6 → 0.07 tok/s — a 50× cliff that looked like a leak and wasn't.
- **Measure the whole distribution, not one sample.** A run dies on the *first* breach, so a
  budget tuned to one chunk (the embed 512-token ceiling) hides the denser ones behind it.
- **`pkill -f foo` matches its own command line** — and the bracket trick (`[f]oo`) must be
  applied to *every* occurrence, or a later unbracketed copy still kills your SSH session.
- **Live tooling drifts from the plan.** M4c hit three llama.cpp CLI/parser changes mid-run
  (`-fa` now needs an arg; `--draft-max` → `--spec-draft-n-max`; two `speed:` lines, not one).
  The resilient-runner pattern (record failures, never crash) plus committed raw logs turned
  each into a re-parse, not a re-run.
- **Persist the raw audit trail.** Every benchmark commits its per-cell raw output, so a
  parser bug fixes by re-parsing and an analysis re-runs with no Pi time and no API spend.

## 9. Next steps

A reranker over the top_k (recall is good; precision could be sharper); an energy/watts axis
(this is an *edge* box — joules-per-answer matters); a Pi 5 / NPU port (the bandwidth wall
moves); light fine-tuning for the citation format; and a security pass for untrusted-network
deployment. The frontier is mapped; these are the next ridgelines.

---

*Faraday is built across model and account switches as a portfolio piece. Specs, plans, and
per-milestone as-builts live in [`docs/superpowers/`](./superpowers/); the operational source
of truth is [`CLAUDE.md`](../CLAUDE.md).*
