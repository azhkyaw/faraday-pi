# Faraday M4a — Quantization Sweep Findings

**Status:** ✅ **COMPLETE — all 18 cells** (Qwen2.5 {0.5B, 1.5B, 3B} × {Q8_0, Q6_K, Q5_K_M,
Q4_K_M, Q3_K_M, Q2_K}), measured end-to-end on **clean power** (official 5.1 V/3 A PSU,
`get_throttled=0x0` for the entire ~15 h run, 2026-06-10). Quality, footprint, **and speed**
are all final. The earlier throttled 12-cell core run and its resolution: §5; run record: §6.

**Deliverables:** [`frontier.png`](./frontier.png) · [`leaderboard.md`](./leaderboard.md) ·
[`sweep.csv`](./sweep.csv) (raw rows).

---

## 1. Headline & recommendation

On a 4 GB Pi 4, the quality-vs-footprint frontier says: **run Qwen2.5-1.5B at Q4_K_M.**
It sits at the **knee** — nearly all of the 1.5B model's quality at an interactive speed
and a comfortable footprint. The full matrix adds a real option above it and a hard floor
below it:

| Pick | Cell | Perplexity | Peak RSS | Decode | When |
|---|---|---|---|---|---|
| **The knee (appliance default)** | **1.5B Q4_K_M** | 11.32 | 1.07 GB | 3.86 tok/s | Best quality-per-byte that is still interactive |
| Quality ceiling | 3B Q4_K_M | 9.92 | 2.0 GB | 1.92 tok/s | Batch / non-interactive jobs where 2× the wait is fine |
| Speed/RAM floor | 0.5B Q4_K_M | 16.61 | 475 MB | 9.47 tok/s | Latency demos; accept a large quality drop |

(1.5B **Q5_K_M** is the co-knee: 11.04 ppl at 1.21 GB — slightly better quality for −18%
decode. Either is defensible; the product ships Q4_K_M.)

**The frontier (perplexity vs peak RSS)** is the textbook L-shape with a long upper arm:
a steep cliff across the 0.5B cluster, the big drop to 1.5B, then a slow, expensive crawl
toward 3B-Q8_0 (9.55 ppl at 3.3 GB — the model alone nearly fills the board). 14 of 18
cells sit on the frontier; the four dominated cells are called out in §2.

## 2. Quality vs footprint (FINAL)

**Size dominates quality — with diminishing returns.** At Q4_K_M: 0.5B → 1.5B → 3B is
16.61 → 11.32 → 9.92. The first 3× in parameters buys −5.3 perplexity; the next 2× buys
only −1.4 (and costs half the decode speed, §3).

| | 0.5B | 1.5B | 3B |
|---|---|---|---|
| Q8_0 | 16.34 | 10.93 | 9.55 |
| Q6_K | 16.32 | 11.05 | 9.72 |
| Q5_K_M | 16.52 | 11.04 | 9.77 |
| Q4_K_M | 16.61 | 11.32 | 9.92 |
| Q3_K_M | 16.84 | 12.29 | **14.15** |
| Q2_K | **19.03** | **15.06** | **13.10** |

**The low-bit cliff arrives *earlier* as the model grows** — the opposite of what the
12-cell core run extrapolated ("bigger models tolerate compression better"). Relative
degradation Q8_0 → Q3_K_M: 0.5B **+3.1%**, 1.5B **+12.4%**, 3B **+48.2%**. The 3B model is
flatly broken below Q4_K_M: its Q3_K_M (14.15 ± 0.58) and Q2_K (13.10 ± 0.53) are *worse
than 1.5B-Q4_K_M* while using more RAM. The Q3-worse-than-Q2 inversion is within ~2σ —
read it as "Q3_K_M ≈ Q2_K, both off the cliff," not as a confident ordering. Whether the
early cliff is intrinsic to the 3B weights or an artifact of these particular imatrix
calibrations is unresolved; the leaderboard reports what was measured, not what was expected.

**Four cells are dominated** (a strictly better cell exists — smaller *and* lower
perplexity), so never run them:
- **0.5B-Q8_0** (602 MB, 16.34) — beaten by **0.5B-Q6_K** (578 MB, 16.32).
- **1.5B-Q6_K** (1348 MB, 11.05) — beaten by **1.5B-Q5_K_M** (1207 MB, 11.04).
- **3B-Q3_K_M** (1671 MB, 14.15) and **3B-Q2_K** (1436 MB, 13.10) — both beaten by
  **1.5B-Q3_K_M** (920 MB, 12.29). A broken big model loses to a healthy small one.

**Reproducibility:** the re-run reproduced all 12 core-run perplexities **exactly** (fixed
GGUF + corpus + thread count → deterministic forward passes) and peak RSS within noise —
empirical confirmation that quality/footprint were power-immune, as claimed when the core
run was published. Footprint: peak RSS ≈ GGUF size + 100–160 MB (KV cache + compute
buffers), so the model file size is an excellent RAM predictor.

## 3. Speed (FINAL — clean power)

**Decode is memory-bandwidth-bound, and now we've measured the constant.** Across **all 18
cells**, `decode tok/s × model bytes` lands in a narrow band:

- ≈ **3.8–3.9 GB/s** at Q8_0/Q6_K (simple dequant),
- drifting down to ≈ **3.3 GB/s** at Q3/Q2 (super-block bit-unpacking starts to eat the
  bandwidth win).

That band *is* the Pi 4's effective LPDDR4 read bandwidth, measured 18 independent ways.
Consequences, verified: within a size, more compression → proportionally faster decode
(0.5B: 7.25 → 10.66 tok/s from Q8 to Q2; 1.5B: 2.33 → 4.84; 3B: 1.15 → 2.60); across
sizes at Q4_K_M, 2.48× the bytes → 2.45× slower (0.5B→1.5B) and 1.96× the bytes → 2.01×
slower (1.5B→3B). To first order on this board: **`decode ≈ 3.8 GB/s ÷ model_bytes`** —
you can now size any GGUF for the Pi 4 without running it.

| decode tok/s | 0.5B | 1.5B | 3B |
|---|---|---|---|
| Q8_0 | 7.25 | 2.33 | 1.15 |
| Q6_K | 7.63 | 3.02 | 1.54 |
| Q5_K_M | 9.05 | 3.15 | 1.72 |
| Q4_K_M | 9.47 | 3.86 | 1.92 |
| Q3_K_M | 10.47 | 4.06 | 2.13 |
| Q2_K | 10.66 | 4.84 | 2.60 |

**Prefill is compute-bound and *kernel*-bound — it does not follow byte count.** In every
size row, **Q8_0 and Q4_K_M are the two fastest prefill quants** (0.5B: 27.9/21.8 tok/s;
1.5B: 7.65/7.52; 3B: 3.54/3.58) while Q5_K_M/Q6_K are the slowest (their bit-packed
super-blocks cost the most ALU work to unpack). The core run showed the same pattern and
we wrote it off as throttle noise; the clean run **reproduced it almost digit-for-digit**
(1.5B Q8_0: 7.67→7.65; Q6_K: 5.68→5.69) — it's real. Lesson: quantization trades
*decode* bandwidth against *prefill* dequant compute, and Q4_K_M's heavily-optimized NEON
path is why it wins twice.

**Cross-validation versus M0:** the harness's 1.5B-Q4_K_M row reads **7.52 prefill / 3.86
decode** vs the hand-measured M0 baseline **7.71 / 3.87** — agreement within 2.5% / 0.3%,
on independent tooling.

**Postscript on the throttled core run:** its speed numbers turned out accurate to ≤~2%
(the under-voltage dips lasted seconds and llama-bench's averaging absorbed them). Flagging
them "preliminary" was still right — that accuracy was unknowable until measured on power
that held `0x0`.

## 4. The 3B verdict: fits in RAM, fails at interactivity

The core run's qualitative observation is now quantified:

- **It fits.** Every 3B quant ran to completion; peak RSS 1.44–3.29 GB, swap untouched.
  The 4 GB ceiling never triggered — `status=oom` went unused across the whole matrix.
- **It's slow in both phases.** Decode 1.15–2.60 tok/s; prefill 2.7–3.6 tok/s. At the only
  healthy quant (Q4_K_M): **1.92 tok/s ≈ 86 words/min** — well below reading speed, and
  RAG prompts (~1k tokens of context) make the prefill phase brutal too.
- **Its low-bit escape hatch is broken** (§2): below Q4_K_M, 3B loses to 1.5B on *both*
  axes. You cannot buy 3B's quality cheaply.
- **RAM-wise it would coexist** with the appliance stack (≈2.4 GB total resident with
  embed + app) — the deal-breaker is throughput, not capacity.

**Verdict:** 3B-Q4_K_M is real frontier (−1.4 ppl vs the knee) for **batch/offline** use;
the interactive appliance stays on **1.5B Q4_K_M**. "Can a 4 GB Pi run a 3B model?" —
yes; "should it serve one?" — no.

## 5. Measurement-hygiene incident: under-voltage throttling (resolved)

During the original core run the board **under-volted under sustained load**:
`vcgencmd get_throttled` → `0x50000` (under-voltage *and* throttling have occurred),
`dmesg` showed repeated `Undervoltage detected!` dips, each recovering in seconds, at
55 °C — **power, not heat**. The marginal PSU sagged below the ~4.63 V trip point; the
firmware capped the clock intermittently. Quality/footprint were unaffected (clock-
independent); speed was flagged preliminary; the 3B rows were deferred.

**Resolution:** official 5.1 V/3 A PSU, verified by a stress test before relaunch (held
`0x0`, ±0.00 V variance), then the full 18-cell re-run held **`0x0` across ~15 h of
sustained load**. The re-run reproduced quality exactly and core-run speeds within ~2%
(§3 postscript) and added the six 3B cells. A Pi 4 can't report its input voltage in
software — only the comparator flag — so the rule stands: **check `get_throttled` before
trusting any benchmark, and re-measure on power you trust before calling numbers final.**

## 6. Run record (the clean 18-cell run)

- **2026-06-10**, fresh start (cleared `sweep.csv` + `raw/` so the resumable runner
  wouldn't skip contaminated rows), via `scripts/70_quant_sweep.sh` detached over SSH.
- First cell artifact 04:39, last 19:31 → **~15 h wall-clock**: 0.5B ≈ 1.4 h, 1.5B ≈ 4.3 h,
  **3B ≈ 9.2 h (61% of the run** — each 3B perplexity pass alone is 55–69 min, matching
  the "~1 h" estimate that justified deferring 3B from the throttled run).
- ≈ **21.9 GB of GGUFs** downloaded to the scratch dir over the run, one model at a time,
  each deleted after its cell (SD-card- and disk-friendly by design).
- Throttle state `0x0` at launch, at every mid-run peek, and at completion.

## 7. Process notes (engineering candor)

- **`pkill -f` self-match has a second form.** The bracket trick (`[f]oo`) stops the
  pattern matching *itself*, but a **literal unbracketed copy of the pattern elsewhere in
  the same command** (here, in a later `pgrep` verify clause) still matches — and killed
  the SSH session. Lesson: bracket *every* occurrence, or kill by resolved PID
  (`kill $(pgrep -f '[f]oo')`).
- **Latent harness gap:** `run_cell` has no subprocess timeout. It didn't bite (3B fit and
  ran), but a model that genuinely thrashed instead of cleanly OOM-ing would hang the sweep
  indefinitely. A per-cell timeout (→ `status=timeout`) remains a worthwhile hardening item.
- **Re-measuring known cells is a free harness regression test.** Re-running the 12 core
  cells cost ~6 of the 15 h but reproduced perplexity *exactly* and speed within ~2% — that
  agreement is itself evidence the harness is sound, in the same spirit as the M0
  cross-validation. It also promoted one "noise" artifact (Q8_0's prefill advantage, §3)
  into a reproduced, real effect.
