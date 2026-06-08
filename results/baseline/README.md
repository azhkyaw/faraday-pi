# Baseline — Qwen2.5-1.5B-Instruct Q4_K_M on Raspberry Pi 4 (4 GB)

Pre-optimization reference for the M4 study. Captured with `llama-bench` on a
clean board (no other model processes running).

## Throughput

| Test | tok/s |
|---|---|
| Prefill (pp128) | 7.71 ± 0.00 |
| Decode  (tg64)  | 3.87 ± 0.00 |

## Environment (validated — these numbers are trustworthy)

| | |
|---|---|
| Model | `qwen2.5-1.5b-instruct-q4_k_m.gguf` (1.04 GiB, 1.78 B params) |
| Engine | llama.cpp @ `715b86a`, built with `GGML_NATIVE` (NEON), GCC 14.2.0 |
| CPU | Cortex-A72 ×4 @ 1.5 GHz (stock, no overclock) |
| Threads | 4 |
| Governor | ondemand |
| Throttling | **none** (`get_throttled=0x0` — no under-voltage, no thermal cap) |
| Temp under load | 38.4°C |
| RAM (`free` "used") | 441 MiB / 3.7 GiB after load — see memory note |
| OS | Raspberry Pi OS Lite, Debian 13 (trixie), kernel 6.12, aarch64 |
| Date | 2026-06-08 |

## Notes

- **Trustworthy baseline**: `get_throttled=0x0` and the ARM clock held at 1.5 GHz,
  so these reflect the silicon at full speed — not a throttled/undervolted artifact.
  This is the honest "naive, un-tuned" reference the optimization study improves on.
- **Why modest**: the Cortex-A72 lacks the `dotprod`/`i8mm` int8 matmul instructions
  (introduced in A76+), so quantized inference falls back to plain NEON. Prefill
  (matmul-heavy) suffers more than decode, hence pp128 is only ~2× tg64.
- **Memory caveat**: `free` "used" (441 MiB) *undercounts* the real footprint because
  llama.cpp `mmap`s the weights as file-backed page cache (counted as buff/cache, not
  "used"). True resident footprint ≈ weights (1.04 GiB) + KV/compute (~240 MiB) + OS.
  M4 will measure **peak process RSS under sustained load** instead.

## M4 optimization targets (the levers that remain)

Since we can't add `dotprod` to the A72, throughput gains must come from:
- CPU governor → `performance` (remove ramp-up latency / ensure steady max clock)
- Safe overclock (1.5 → 1.8–2.0 GHz, cooling permitting)
- KV-cache quantization, flash-attention
- Thread / batch-size tuning
- Speculative decoding (0.5 B draft + 1.5 B target)
