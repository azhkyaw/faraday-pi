# M4a — Quantization Sweep Results

Quality-vs-footprint frontier for Qwen2.5 {0.5B, 1.5B, 3B} × {Q8_0, Q6_K,
Q5_K_M, Q4_K_M, Q3_K_M, Q2_K} on a Raspberry Pi 4 (4 GB), all imatrix GGUFs
(bartowski). Produced by `scripts/70_quant_sweep.sh` →
`python -m faraday.bench.sweep` (+ `…plot`).

## Artifacts (committed after the run)

| File | What |
|---|---|
| `sweep.csv` | One row per cell: disk size, peak RSS, prefill/decode tok/s, perplexity, status. |
| `frontier.png` | Perplexity vs peak RSS scatter; the Pareto frontier and the 4 GB wall. |
| `leaderboard.md` | Cells ranked by perplexity; ★ marks the frontier; non-fitting cells listed. |
| `findings.md` | The narrative: the "knee", quant cliffs, what to actually run on a 4 GB Pi, throttle hygiene. |
| `raw/*.log` | Per-cell raw tool output (git-ignored via `*.log`; kept on the Pi for debugging). |

## Reproduce

```bash
# On the Pi (overnight, resumable):
bash scripts/70_quant_sweep.sh
# Then commit the curated outputs from the dev box.
```

`status` ∈ {`ok`, `oom`, `download_failed`, `error`}. A non-fitting 3B cell is
`oom` — that's a charted result (the 4 GB ceiling), not a bug.
