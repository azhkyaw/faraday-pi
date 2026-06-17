# What runs well on a Raspberry Pi 4 — measured

A reference table for anyone putting an LLM on a 4 GB Pi 4. These are end-to-end measurements
on real hardware, not extrapolations. The short answer: **Qwen2.5-1.5B at Q4_K_M** — the knee
of the quality/footprint frontier and the only quant that is good, small, *and* interactive.

## Hardware & method

- **Board:** Raspberry Pi 4, 4 GB, Raspberry Pi OS 64-bit, quad-core Cortex-A72.
- **Power:** official 5.1 V / 3 A PSU. This matters — a marginal supply under-volts under
  sustained load (`vcgencmd get_throttled` → `0x50000`), capping the clock. **Every number
  below was taken with `get_throttled` reading `0x0`**; treat any Pi benchmark on an unverified
  PSU as preliminary.
- **Inference:** `llama.cpp` (`llama-bench`, `-p 512 -n 128`, threads = 4). Speed is reported
  tok/s; perplexity is `llama-perplexity` over WikiText-2; peak RSS is process resident set
  (not `free` "used", which hides mmap'd weights in `buff/cache`).
- Full method + raw logs: [M4a findings](../results/sweep/findings.md) ·
  [M4c findings](../results/optimize/findings.md).

## Table 1 — model × quantization (Qwen2.5, 18 cells)

Sorted by perplexity (lower = better). ★ = on the quality/footprint Pareto frontier.
Decode < ~3 tok/s reads slower than a person; **bold** = the shipped appliance default.

| Cell | Perplexity | Peak RSS (MB) | Decode tok/s | ★ |
|---|---|---|---|---|
| 3B-Q8_0 | 9.55 | 3288 | 1.15 | ★ |
| 3B-Q6_K | 9.72 | 2575 | 1.54 | ★ |
| 3B-Q5_K_M | 9.77 | 2276 | 1.72 | ★ |
| 3B-Q4_K_M | 9.92 | 1995 | 1.92 | ★ |
| 1.5B-Q8_0 | 10.93 | 1704 | 2.33 | ★ |
| 1.5B-Q5_K_M | 11.04 | 1207 | 3.15 | ★ |
| 1.5B-Q6_K | 11.05 | 1348 | 3.02 | |
| **1.5B-Q4_K_M** | **11.32** | **1074** | **3.86** | **★** |
| 1.5B-Q3_K_M | 12.29 | 920 | 4.06 | ★ |
| 3B-Q2_K | 13.10 | 1370 | 2.60 | |
| 3B-Q3_K_M | 14.15 | 1671 | 2.13 | |
| 1.5B-Q2_K | 15.06 | 779 | 4.84 | ★ |
| 0.5B-Q6_K | 16.32 | 578 | 7.63 | ★ |
| 0.5B-Q8_0 | 16.34 | 602 | 7.25 | |
| 0.5B-Q5_K_M | 16.52 | 496 | 9.05 | ★ |
| 0.5B-Q4_K_M | 16.61 | 475 | 9.47 | ★ |
| 0.5B-Q3_K_M | 16.84 | 435 | 10.47 | ★ |
| 0.5B-Q2_K | 19.03 | 418 | 10.66 | ★ |

**Reading it:** the 3B cluster owns the top of the perplexity column but every 3B quant
decodes below reading speed (≤1.92 tok/s) — it *fits* the board but can't *serve* it
interactively. The 0.5B cluster is fast but a quality cliff. **1.5B-Q4_K_M is the knee**:
~96% of the 3B model's job at 2× the speed and half the RAM. Two rules fall out of the data:
**decode ≈ 3.8 GB/s ÷ model_bytes** (size any GGUF for a Pi 4 without running it), and **don't
go below Q4_K_M on the 3B** — its Q3/Q2 are worse than 1.5B-Q4_K_M while using more RAM.

## Table 2 — tuning the shipped model (1.5B-Q4_K_M)

Can you tune past 3.9 tok/s? No. Decode is bandwidth-bound, so every CPU lever lands flat;
the two cross-runtime experiments lose. Sorted by decode tok/s.
([leaderboard](../results/optimize/leaderboard.md) ·
[waterfall.png](../results/optimize/waterfall.png))

| Config | decode tok/s | prefill tok/s | note |
|---|---|---|---|
| threads=3 | 3.91 | 5.79 | +0.3% (noise) |
| **baseline** (`-t 4`) | **3.90** | **7.52** | the shipped config |
| batch ubatch=1024 | 3.85 | 7.61 | no gain |
| governor=performance | 3.84 | 7.59 | no gain |
| stacked_best (`-t 3`) | 3.84 | 5.82 | net-negative (starves prefill) |
| kv=q8_0 | 3.83 | 7.41 | no gain (saves memory, not speed) |
| flash_attn | 3.83 | 7.60 | no gain on CPU |
| ollama-default | 3.74 | 8.07 | −4% decode (managed-runtime overhead) |
| threads=2 | 3.00 | 3.91 | starved |
| **speculative** (0.5B draft) | **0.942** | — | **4× slower**, 21.6% draft accept |

**Reading it:** no lever beats the baseline; the "best" stacked config is *worse* once you
weight prefill; and **speculative decoding — a GPU win — is a 4× *loss* on a bandwidth-bound
CPU** (you pay for draft compute you mostly discard). The appliance already ships at its
throughput ceiling. Prefill, unlike decode, does fall with context depth (7.74 → 6.22 tok/s
over 128 → 4096 prompt tokens — [context_curve.png](../results/optimize/context_curve.png)),
so price long-context requests by prompt tokens.

## Caveats

- **One board.** N=1 hardware; another Pi 4 (different PSU, SD card, silicon lottery) will
  vary by a few percent. The *trends* (bandwidth wall, the knee, the 3B interactivity floor)
  are robust; the third decimal is not.
- **WikiText-2 perplexity** is a generic language-modeling proxy, not a RAG-quality score —
  for that, see the [M4b evals](../results/evals/findings.md) (recall, faithfulness,
  citations) on a themed Apollo corpus.
- Corpus articles are Wikipedia (CC-BY-SA); the models are Qwen2.5 (Apache-2.0) and
  bge-small-en-v1.5 (MIT).

## Reproduce

On a Pi 4 with the repo cloned and built (`bash scripts/bootstrap.sh`):

- **Table 1:** `bash scripts/70_quant_sweep.sh` (~15 h, downloads each GGUF in turn).
- **Table 2:** `bash scripts/90_optimize.sh` (quiet board; ablate-then-stack + speculative +
  Ollama).
- **RAG quality:** `bash scripts/80_run_evals.sh`; **GBNF citations:**
  `bash scripts/95_gbnf_measure.sh`.

All runner scripts record raw per-cell logs, committed in `results/` — re-derive any number
here without re-running.
