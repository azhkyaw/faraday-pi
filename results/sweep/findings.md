# Faraday M4a — Quantization Sweep Findings

**Status:** Core run complete — **12 cells (Qwen2.5 0.5B + 1.5B × 6 quants)**. The 3B
row and clean speed numbers are deferred to a **re-run** (the run hit intermittent
**under-voltage throttling** on a marginal PSU — see §5). **Quality and footprint are
final** (power-immune); **speed is preliminary**.

**Deliverables:** [`frontier.png`](./frontier.png) · [`leaderboard.md`](./leaderboard.md) ·
[`sweep.csv`](./sweep.csv) (raw rows).

---

## 1. Headline & recommendation

On a 4 GB Pi 4, the quality-vs-footprint frontier says: **run Qwen2.5-1.5B at Q4_K_M (or
Q5_K_M).** It sits at the **knee** — perplexity ~11.3, peak RAM ~1.07 GB — capturing nearly
all of the 1.5B model's quality at a comfortable footprint. Drop to 0.5B only if you need
minimum RAM / maximum speed and can accept a large quality hit (~16 vs ~11 perplexity). The
3B model fits but is impractically slow (§4).

**The frontier (perplexity vs peak RSS):** a textbook L-shape — a steep quality cliff across
the 0.5B cluster, a big drop to the 1.5B cluster, then the curve flattens (extra RAM buys
almost no quality past Q4/Q5 of 1.5B).

## 2. Quality vs footprint (FINAL — unaffected by throttling)

**Size dominates quality.** Tripling parameters 0.5B → 1.5B drops perplexity from ~16–19 to
~11 — a far bigger quality lever than any quant choice within a size.

| | 0.5B perplexity | 1.5B perplexity |
|---|---|---|
| Q8_0 | 16.34 | 10.93 |
| Q6_K | 16.32 | 11.05 |
| Q5_K_M | 16.52 | 11.04 |
| Q4_K_M | 16.61 | 11.32 |
| Q3_K_M | 16.84 | 12.29 |
| Q2_K | **19.03** | **15.06** |

**The quant cliff is real, and bigger models tolerate compression better.** Quality drifts
only gently from Q8 down to Q3, then falls off a cliff at **Q2_K**. For 0.5B the cliff is the
single Q2 step (16.84 → 19.03). For 1.5B the gentle zone extends through Q4; Q3 starts to
slip (12.29) and Q2 falls hard (15.06). More parameters = more redundancy to absorb the
rounding noise.

**Two cells are dominated** (a strictly better option exists — smaller *and* lower
perplexity), so never run them:
- **0.5B-Q8_0** (602 MB, 16.34) — beaten by **0.5B-Q6_K** (578 MB, 16.32). Q8's extra
  precision bought *nothing* here.
- **1.5B-Q6_K** (1348 MB, 11.05) — beaten by **1.5B-Q5_K_M** (1207 MB, 11.04).

**Footprint scales with quant**, all comfortably within 4 GB: 0.5B 419–602 MB, 1.5B
779–1704 MB peak RSS.

## 3. Speed (PRELIMINARY — see throttling caveat, §5)

**Decode is memory-bandwidth-bound, exactly as theory predicts.** Within a size, more
compression → fewer bytes streamed per token → *faster* decode (0.5B: 7.24 → 10.61 tok/s as
Q8 → Q2; 1.5B: 2.33 → 4.74). Across sizes, 1.5B (~3× the weights of 0.5B) decodes ~3× slower
— the `decode ∝ 1/model_bytes` relationship, measured.

**Harness cross-validation:** 1.5B-Q4_K_M measured **7.53 prefill / 3.84 decode** here vs the
independent **M0 hand-measured baseline of 7.71 / 3.87** — agreement within ~2%, strong
evidence both measurements are sound.

⚠️ The prefill column is noisy (e.g. 1.5B Q8 7.67 > Q6 5.68) because of *intermittent* clock
capping from under-voltage. Absolute speeds for 1.5B+ are **conservative**; relative trends
hold. Clean numbers await the re-run.

## 4. The 3B observation: a *speed* wall, not a memory wall (qualitative)

We expected 3B to OOM against the 4 GB ceiling. It doesn't. Live diagnosis of `3B Q8_0`
(**3.06 GB** GGUF) showed it **fits and runs**: the mmap'd weights sat in the page cache
(`free`: used 409 MiB, buff/cache 3.4 GiB — the M0 mmap lesson live), swap untouched, all
four cores compute-bound at full clock. The real limit is **speed**: a single perplexity
pass on 3B runs ~1 h. So 3B is technically usable on a 4 GB Pi but **impractically slow** —
the wall is throughput, not capacity. Full 3B characterization is deferred to the re-run.

## 5. Measurement-hygiene incident: under-voltage throttling

Partway through the run the board began **under-volting under sustained load**:
- `vcgencmd get_throttled` → `0x50000` (under-voltage **and** throttling *have occurred*).
- `dmesg` logged real dips: `Undervoltage detected!` at 21:40, 21:57, 23:22 (each recovering
  in seconds).
- Temp was 55.5 °C — so this is **power, not heat**. The PSU sags below the ~4.63 V trip
  point under load, the firmware caps the clock, then it recovers.

**Impact:** speed columns for the heavier (1.5B+) cells are under-reported. **Perplexity and
peak RSS are unaffected** (they don't depend on clock speed) — which is why the frontier and
leaderboard are final.

**Note:** a Pi 4 cannot report its actual 5 V input voltage in software (only the
comparator flag + dmesg events); measuring the real voltage needs an inline USB-C power
meter. Fix before the re-run: the **official 5.1 V/3 A PSU + a good short cable**.

## 6. Re-run plan (for clean speed + the 3B row)

After the PSU fix, start **fresh** (the resumable CSV would otherwise skip the contaminated
rows):
```bash
ssh pi@raspberrypi.local
rm ~/faraday/results/sweep/sweep.csv; rm -f ~/faraday/results/sweep/raw/*
cd ~/faraday && bash scripts/70_quant_sweep.sh   # full 18 cells, clean clock
```
Expect: identical quality/footprint, *higher* (un-throttled) speeds, and the six 3B rows
completing the matrix (slowly — budget the full overnight + most of a day for the 3B
perplexity passes).

## 7. Process notes (engineering candor)

- **`pkill -f` self-match has a second form.** The bracket trick (`[f]oo`) stops the pattern
  matching *itself*, but a **literal unbracketed copy of the pattern elsewhere in the same
  command** (here, in a later `pgrep` verify clause) still matches and killed the SSH
  session. Lesson: when killing by pattern over SSH, bracket *every* occurrence, or kill by
  resolved PID (`kill $(pgrep -f '[f]oo')`).
- **Latent harness gap:** `run_cell` has no subprocess timeout. It didn't bite (3B fit and
  ran), but a model that genuinely thrashed instead of cleanly OOM-ing would hang the sweep
  indefinitely. A per-cell timeout (→ `status=timeout`) is a worthwhile hardening item.
