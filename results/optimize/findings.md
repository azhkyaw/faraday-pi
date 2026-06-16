# Faraday M4c — Inference Optimization Study — Findings

**Verdict: the shipped appliance (Qwen2.5-1.5B Q4_K_M) is already at its practical
throughput optimum on the Raspberry Pi 4.** No CPU-tuning lever beats baseline decode,
speculative decoding is *counterproductive*, and the one knob that moves at all (threads)
trades prefill away for a rounding-error decode gain. Decode is memory-bandwidth-bound and
prefill is kernel-bound — both ceilings are hardware, not configuration. This is a negative
result, and a useful one: there is no free configuration win left on the table.

Run on a quiet board, `throttled=0x0` and ~55 °C throughout. Method: ablate each lever
independently vs a fixed baseline, stack the clean winners, then compare speculative decoding
(0.5B draft + 1.5B target) and an Ollama baseline. Raw per-cell tool logs are committed in
`raw/` (the audit trail behind every row).

## Headline numbers

| Rank | Cell | decode t/s | prefill t/s | vs baseline decode |
|---|---|---|---|---|
| 1 | threads=3 | **3.91** | 5.79 | +0.3% (noise) |
| 2 | baseline (`-t 4`) | 3.90 | 7.52 | — |
| 3 | ctx=128 | 3.87 | 7.74 | −0.8% |
| 4–8 | ubatch=1024 / ctx / governor / stacked_best | 3.84–3.85 | 5.82–7.61 | ~−1.5% |
| 9–12 | ctx=512/2048 / kv=q8_0 / flash_attn | 3.83 | 6.83–7.60 | −1.8% |
| 13 | ollama-default | 3.74 | 8.07 | −4.1% |
| 14 | threads=2 | 3.00 | 3.91 | −23% |
| 15 | **speculative** | **0.942** | — | **−76%** |

**Every CPU-tuning lever lands within ±2% of baseline decode.** The whole tunable surface is
a ~5% band; the cliffs (threads=2, speculative) are *losses*, not wins.

## 1. Why decode won't move — the bandwidth wall

Decode throughput is **flat at 3.83–3.91 tok/s regardless of governor, batch size, KV-cache
quantization, flash-attention, or context depth.** This is the signature of a
**memory-bandwidth-bound** decode: generating each token streams the full ~1 GB of quantized
weights from RAM, and no CPU-side knob changes RAM bandwidth.

It quantifies cleanly against M4a's model (`decode ≈ mem_bandwidth ÷ model_bytes`):
~3.8 GB/s ÷ ~1.0 GB ≈ 3.8 tok/s, measured 3.90. Confirmed here **15 more ways** — every cell
agrees. The corollary is decisive for the appliance: *the only lever that changes decode speed
is the one M4a already pulled* (model size / quant choice), because that's the only thing that
changes `model_bytes`. Runtime tuning cannot help.

- **governor=performance** (3.84): no gain — the CPU already boosts under load; pinning the
  governor just removes idle-downclock, irrelevant to a bandwidth-bound steady state.
- **ubatch=1024** (3.85): no gain — larger compute batches help prefill parallelism, not the
  inherently sequential, bandwidth-bound decode.
- **kv=q8_0** (3.83) and **flash_attn** (3.83): no gain — KV-cache quant saves *memory* (lets
  you hold more context), flash-attention saves attention *compute*; neither touches the
  weight-streaming bottleneck. Their value is capacity/latency at long context, not tok/s.
- **threads=2** (3.00, −23%): the only large decode move, and it's a *loss* — two cores can't
  saturate the memory bus that four (baseline) and three nearly do. Confirms the board needs
  ≥3 threads; beyond that, bandwidth saturates and more threads stop helping.

## 2. The prefill–context curve

Unlike decode, **prefill *does* respond — it collapses monotonically with context depth:**

| context (prompt tokens) | 128 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| prefill t/s | 7.74 | 7.55 | 7.19 | 6.83 | 6.22 |
| decode t/s | 3.87 | 3.83 | 3.85 | 3.83 | 3.85 |

Prefill falls ~20% from 128 → 4096 tokens because it is **compute-bound** (attention is
O(n²) in sequence length, so more context = more matmul per token). Decode over the same
sweep is **dead flat** — depth-independent, because each decode step still streams the same
weights regardless of how much context precedes it. This is the two-regime picture M4a
predicted, now drawn directly: *price long-context batches by prompt tokens, not by question.*
It also re-confirms why `GEN_CTX=8192` deep cells dominated the M4b eval runtime.

## 3. Speculative decoding — a GPU technique that backfires on CPU

The single most decisive result: **speculative decoding runs at 0.942 tok/s — ~4× *slower*
than plain decode (3.90)** — with a **21.6% draft-acceptance rate** (0.5B draft, 1.5B target,
same Qwen2.5 tokenizer).

Speculative decoding wins on GPUs because batch-verifying N draft tokens in one forward pass
is nearly free, so even modest acceptance amortizes the draft cost. On this **bandwidth-bound
CPU** that economics inverts:
- the draft model adds its own weight-streaming cost for every proposed token;
- low acceptance (~1 in 5) means ~80% of drafted tokens are computed and thrown away;
- the target's batch-verify is *not* meaningfully cheaper than sequential decode on CPU (no
  wide parallel units to hide the extra work).

Net: you pay for draft compute you mostly discard, and the bandwidth wall still gates the
target. **Speculative decoding is the wrong tool for a bandwidth-bound edge CPU** — a clean,
non-obvious finding worth the build.

## 4. Ollama vs llama.cpp

Ollama (the managed runtime, same 1.5B Q4_K_M) lands at **decode 3.74 / prefill 8.07** vs
llama.cpp's **3.90 / 7.52**: ~4% slower decode, ~7% faster prefill — a different operating
point on the same bandwidth wall, not a win. The decode delta is managed-runtime overhead; the
prefill edge is Ollama's batch defaults. For an appliance already standardized on llama.cpp
(`llama-server`), there's no throughput reason to switch.

## 5. The optimizer's "best" is net-negative

`stack_winners` selected `governor=ondemand -t 3` (the only lever whose decode nominally beat
baseline). Measured, **stacked_best is decode 3.84 / prefill 5.82** — *worse than baseline on
both axes*. The `threads=3` "win" (3.91 vs 3.90) was inside run-to-run noise, and forcing
`-t 3` **starves prefill** (5.82 vs 7.52, −23%). The ablate-then-stack method worked exactly
as designed — and correctly reported that **the best available configuration is the default**.
(The throttle-gating never had to fire: every cell ran clean at `0x0`.)

## 6. Run record & methodology notes

- **Clean run, `0x0` throughout, ~55 °C.** Resumable (`done_keys` on `(component,label)`); the
  monitor's failure-coverage filter and `run_cell`'s never-raises design meant bad cells were
  *recorded*, not fatal.
- **Three CLI/parser drifts caught against the live tooling** (the Pi's llama.cpp is newer than
  the plan assumed): (1) `llama-bench` now requires `-fa on` (bare `-fa` rejected → usage dump);
  (2) `llama-speculative` renamed `--draft-max` → `--spec-draft-n-max`; (3) `parse_speculative`
  grabbed the first `speed:` line (prefill) instead of the `decoded` line, recording 5.312
  instead of 0.942. The first two surfaced as recorded errors mid-run; the third was caught by
  reading the raw log before trusting the number.
- **Re-derive without re-benchmark.** Because per-cell raw stdout/stderr is persisted in `raw/`,
  the speculative parser bug was fixed by *re-parsing the saved log* — the true 0.942 was
  recovered with zero extra Pi inference. Same property that lets M4b re-score off committed raw
  rows: the audit trail is the source of truth, the CSV is a derived view.
- **Caveat — flash-attention baseline.** `llama-bench`'s default is `-fa auto`, so the baseline
  may already auto-enable FA where it helps; the `flash_attn` cell forces `-fa on`. The measured
  delta (≈0) is "forced-on vs auto," not "on vs off." Either way FA doesn't move decode.

## 7. Implications for the appliance

- **Ship the defaults.** `30_run_servers.sh` already runs `llama-server` at the measured
  optimum; M4c found nothing to change. The decode ceiling (~3.9 tok/s) is physics.
- **The real throughput lever is upstream** (model/quant — M4a's job), not runtime tuning. If
  more decode speed is ever needed, it comes from a smaller/faster-to-stream model, not config.
- **Overclock is the one untested escape hatch** (raising `arm_freq`/memory clock would raise
  the bandwidth ceiling itself) — deliberately left as a separate, reboot-gated manual step
  (see `scripts/90_optimize.sh`); not run here to keep the board's shipped, warranty-safe
  profile as the baseline of record.

**Artifacts:** `optimize.csv` (15 cells), `leaderboard.md`, `waterfall.png`,
`lever_gains.png`, `context_curve.png`, and `raw/` (per-cell tool logs). Re-render via
`python -m faraday.bench.optimize_plot` — no Pi run required.
