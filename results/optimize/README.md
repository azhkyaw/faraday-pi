# M4c — Inference Optimization Results

Throughput tuning of the shipped model (Qwen2.5-1.5B Q4_K_M) on a Raspberry Pi 4
(4 GB), via `scripts/90_optimize.sh` → `python -m faraday.bench.optimize` (+ `…optimize_plot`).

| File | What |
|---|---|
| `optimize.csv` | One row per config: prefill/decode tok/s, peak RSS, accept rate, throttle, notes. |
| `lever_gains.png` | Per-lever marginal decode gain (%) vs baseline. |
| `waterfall.png` | Baseline → best-tuned (stacked winners) decode throughput. |
| `context_curve.png` | Prefill/decode tok/s vs context length (motivates KV-cache quant). |
| `leaderboard.md` | All configs ranked by decode tok/s (incl. speculative + Ollama). |
| `findings.md` | The narrative: which levers won, the stacked speedup, speculative's CPU verdict, the Ollama delta. |
| `raw/` | Per-cell raw tool output — committed as the audit trail behind `optimize.csv` (llama-bench ± stderr + `time -v`). |

## Method
Ablate-then-stack: each lever measured independently vs a fixed baseline, winners
(clean `throttled=0x0` only) stacked into a best config. Speculative decoding (0.5B
draft + 1.5B target) and an Ollama default are compared to best-tuned. Overclock is a
separate reboot procedure (see `scripts/90_optimize.sh`). Run on a quiet board.

## Reproduce
`bash scripts/90_optimize.sh` on the Pi, then commit the curated outputs from the dev box.
