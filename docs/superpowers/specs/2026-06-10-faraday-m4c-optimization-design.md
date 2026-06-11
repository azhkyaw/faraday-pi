# Faraday M4c — Inference Optimization Study
## Design Document

| | |
|---|---|
| **Status** | Approved (brainstorming) — ready for implementation planning |
| **Date** | 2026-06-10 |
| **Milestone** | M4c (third of three M4 sub-studies; M4a quant sweep · M4b RAG evals · **M4c optimization**) |
| **Builds on** | M0–M3; M4a (`faraday.bench` harness — reused); M4a's model pick (1.5B Q4_K_M) |

---

## 1. Overview

M4c is the **inference-optimization study** — the design spec's "optimization waterfall" (§9, §195). It treats the Pi 4 as a systems constraint to engineer against: starting from the M4a-recommended model (**Qwen2.5-1.5B Q4_K_M**), it measures how much throughput a sequence of tuning levers buys, whether speculative decoding helps on CPU, and how the hand-tuned result compares to a managed default (Ollama). It answers: *"What does inference engineering actually buy you on a 4 GB Pi — and which levers matter?"*

Three components + a characterization:
- **A. Tuning waterfall** — ablate each lever independently vs a fixed baseline, then stack the winners.
- **B. Speculative decoding** — an alternative decode strategy, vs the best-tuned config.
- **C. Ollama baseline** — a managed default vs the best-tuned config.
- **Characterization** — TTFT + decode tok/s vs context length.

## 2. Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Scope | **Full**: software levers + overclock + speculative decoding + Ollama baseline | The complete inference-engineering narrative; the standout-portfolio tier |
| Waterfall method | **Ablate-then-stack** | Each lever's marginal value is measured independently of the others (avoids the greedy-order attribution trap), then winners are combined into a "best config" |
| Target model | **1.5B Q4_K_M** (the M4a frontier pick) | M4c optimizes the model the project actually ships; M4a's recommendation is already final |
| Harness | **Extend `faraday.bench`** | Reuse M4a's tested `parse_llama_bench`/`parse_time_v`, resumable-CSV, plotting (DRY); add M4c config/runner/plot |
| Profiling lead | **Include TTFT/throughput-vs-context**; **defer the prefill-rate puzzle** | The context curve motivates KV-cache quant and is clearly valuable; the prefill puzzle is an analytical rabbit hole |

## 3. Goals / non-goals

**Goals**
- A per-lever marginal-gain measurement + a baseline→best **waterfall** for 1.5B Q4_K_M.
- A speculative-decoding verdict on CPU (decode tok/s + token acceptance rate vs best-tuned).
- An Ollama-default vs hand-tuned comparison.
- A TTFT/decode-vs-context curve.
- Committed: `optimize.csv`, `waterfall.png`, `context_curve.png`, `leaderboard.md`, `findings.md`. Pure parsing/config/planning logic unit-tested; ruff clean.

**Non-goals (YAGNI / later)**
- Quant/quality re-exploration (that's M4a); RAG answer quality (M4b); energy/watts (§17 future); a full lever-combination matrix (2^N — impractical); Pi-5/NPU (future). No new model training.

## 4. Architecture

```
A. TUNING WATERFALL  — ablate each lever vs a fixed baseline; stack winners → best config
B. ALT DECODE        — speculative decoding (0.5B draft + 1.5B target) vs best-tuned
C. EXTERNAL BASELINE — Ollama default vs best-tuned
   + CHARACTERIZATION — TTFT & decode tok/s vs context length (varying -p)
```

All measured with `time -v llama-bench` (reusing M4a's parsers) on the **1.5B Q4_K_M** model, on a **quiet board** (nothing else competing — the lesson from running M4b authoring while the M4a sweep measured). Results born on the Pi, committed from the dev box (the M4a reverse path).

## 5. The lever matrix (Part A)

**Baseline** = stock `llama-bench` on 1.5B Q4_K_M: `ondemand` governor, `-t 4`, default batch, no flash-attn, f16 KV cache, stock 1.5 GHz clock.

| Lever | Tested vs baseline | Mechanism |
|---|---|---|
| CPU governor | `performance` (pinned max freq) vs `ondemand` | `/sys/.../scaling_governor` |
| Threads `-t` | {2, 3, 4} | llama-bench flag |
| Batch `-b` / `-ub` | larger micro-batch vs default | llama-bench flag (prefill) |
| KV-cache quant | `q8_0` via `-ctk/-ctv` vs f16 | llama-bench flag (memory-bound decode) |
| Flash-attention | `-fa` on vs off | llama-bench flag |
| **Overclock** | `arm_freq` 1.8/2.0 GHz + `over_voltage` vs stock 1.5 | `/boot/config.txt` + **reboot** |

Winning levers (those with a positive, throttle-clean marginal gain) are **stacked into a "best config"**; the baseline→best delta is the **waterfall**.

## 6. The overclock execution wrinkle

Changing `arm_freq` requires a `config.txt` edit **and a reboot** — which kills any running Python. So:
- **Software levers + speculative + Ollama** run in **one resumable sweep at stock clock**.
- **Overclock** is a separate documented procedure: set `config.txt` → `sudo reboot` → re-run the key configs (baseline + best-software) at the OC setting → merge those rows into `optimize.csv`.
Recognizing one lever's different execution lifecycle upfront avoids a mid-run surprise.

## 7. Components (delta on `faraday/bench/`)

| Unit | Path | Responsibility |
|---|---|---|
| Opt config | `bench/optimize_config.py` | lever→flags/env mapping; the ablate-then-stack plan; the `optimize.csv` schema |
| Opt runner | `bench/optimize.py` | per cell: apply lever → `time -v llama-bench` → record; resumable; injected Runner (reuses `parsers`) |
| Extra parsers | `bench/parsers.py` (+) | `parse_ollama_bench`, `parse_speculative` (decode tok/s + acceptance rate) — pure, tested |
| Opt plots | `bench/optimize_plot.py` | per-lever bar chart + baseline→best waterfall + TTFT/decode-vs-context curve + leaderboard |
| On-Pi runner | `scripts/90_optimize.sh` | stock-clock sweep; ensure `llama-speculative` built + draft model + Ollama present; documented OC reboot steps |
| Results | `results/optimize/` | `optimize.csv`, `waterfall.png`, `context_curve.png`, `leaderboard.md`, `findings.md` (committed); raw logs git-ignored |

## 8. Special pieces

- **Speculative decoding:** needs `llama-speculative` built on the Pi (its build lacks it — the same `cmake --build ~/llama.cpp/build --target llama-speculative -j3` trick used for `llama-perplexity`) + a **Qwen2.5-0.5B-Instruct draft** model resident alongside the 1.5B target. The draft *must* share the target's tokenizer (a hard requirement of speculative decoding — same Qwen2.5 family satisfies it). Two models resident is memory-tight but fits in 4 GB. Record decode tok/s + **token acceptance rate**; compare to best-tuned decode.
- **Ollama baseline:** install Ollama on the Pi, `ollama pull qwen2.5:1.5b`, bench it. Ollama wraps llama.cpp, so this is framed honestly as **"managed defaults vs hand-tuned settings"** — the "what did engineering buy?" headline, not a different engine.
- **Characterization:** `llama-bench -p {128,512,1024,2048,4096}` → prefill tok/s + **TTFT** (= prompt_tokens ÷ prefill-rate) and decode tok/s at each context size → the curve that *motivates* KV-cache quant (decode slows as the KV stream grows).

## 9. CSV schema (the durable record)

`results/optimize/optimize.csv`, one row per measured config:
```
component, label, prefill_tps, decode_tps, peak_rss_bytes, accept_rate, throttled, notes
```
`component` ∈ {`baseline`, `lever`, `stacked_best`, `speculative`, `ollama`, `context`}; `label` names the cell (e.g. `governor=performance`, `ctx=2048`). `accept_rate` blank except speculative. Resumable: skip `(component, label)` already present.

## 10. Methodology & measurement hygiene

- Every cell records **`vcgencmd measure_temp` + `get_throttled`**; any thermally/under-volt-throttled cell is **flagged invalid** (`throttled` column) — doubly important for the OC lever.
- Reps via llama-bench's built-in averaging (± reported); a cell is trusted only at a clean clock.
- Run on a **quiet board** — no concurrent load (the lesson learned the hard way this milestone).
- Cross-check the baseline against the M4a clean 1.5B-Q4_K_M numbers (independent-measurement sanity, as M4a's smoke matched M0).

## 11. Error handling

- **Lever unsupported / flag rejected** (e.g. a KV-quant type needing `-fa`) → record `notes=unsupported`, continue.
- **OOM** (speculative two-model memory) → `notes=oom`, continue — a charted result.
- **Throttle mid-cell** → flag `throttled`, the cell is reported but excluded from the waterfall.
- **Parse failure** → raw saved to `results/optimize/raw/<cell>.log`, row `notes=error`.
- Resumable runner; always restore the governor to `ondemand` on exit.

## 12. Testing

- **Unit (dev, no Pi):** the lever→flags mapping (`optimize_config`); the new `parse_ollama_bench`/`parse_speculative` against captured sample outputs; the ablate-then-stack planner (given per-lever results, picks winners + builds best config); plot smoke tests. Mirrors M4a, with fakes/captured text.
- **Integration (Pi):** a 1-lever end-to-end smoke before the full run.
- TDD throughout; injected Runner (real subprocess at the edge, fake in tests); ruff clean. Author on Windows → `git push pi` → `pytest` on the Pi.

## 13. File structure (delta)

```
src/faraday/bench/
  optimize_config.py   # NEW: lever matrix + stack plan + CSV schema
  optimize.py          # NEW: resumable per-lever runner (injected Runner)
  optimize_plot.py     # NEW: waterfall + per-lever bars + context curve + leaderboard
  parsers.py           # +parse_ollama_bench, +parse_speculative
tests/
  test_optimize_config.py  test_optimize_parsers.py
  test_optimize_plan.py    test_optimize_plot.py   test_optimize_integration.py
scripts/90_optimize.sh
results/optimize/        # optimize.csv, waterfall.png, context_curve.png, leaderboard.md, findings.md
```

## 14. Definition of done

- Per-lever marginal gains + baseline→best waterfall produced on the Pi (resumably), at a clean clock.
- Speculative-decoding verdict (tok/s + acceptance) and Ollama-vs-tuned delta recorded.
- TTFT/decode-vs-context curve produced.
- `optimize.csv`, `waterfall.png`, `context_curve.png`, `leaderboard.md`, `findings.md` committed.
- Parser/config/planner/plot unit tests green; ruff clean.

## 15. Execution note

The biggest M4 study — multi-session on the Pi, and it requires a **quiet board** + the verified PSU, so execution naturally sequences **after** the M4a clean sweep and the M4b run. Design, spec, plan, and harness code-authoring all need no Pi (the "author offline, batch-verify later" mode).

## 16. Deferred / future

- The **prefill-rate puzzle** (real prefill beat the llama-bench pp128 prediction) — an analytical thread for §17 future-work, not M4c.
- Energy / tokens-per-watt (needs an inline USB power meter) — §17.
- Pi-5 / NPU accelerators — §17.
