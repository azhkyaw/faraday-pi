# Faraday M4c — Optimization Leaderboard

Sorted by decode tok/s (higher = better).

| Rank | Cell | decode t/s | prefill t/s | accept % | throttled |
|---|---|---|---|---|---|
| 1 | threads=3 | 3.91 | 5.79 | - | throttled=0x0 |
| 2 | baseline | 3.9 | 7.52 | - | throttled=0x0 |
| 3 | ctx=128 | 3.87 | 7.74 | - | throttled=0x0 |
| 4 | ubatch=1024 | 3.85 | 7.61 | - | throttled=0x0 |
| 5 | ctx=1024 | 3.85 | 7.19 | - | throttled=0x0 |
| 6 | ctx=4096 | 3.85 | 6.22 | - | throttled=0x0 |
| 7 | governor=performance | 3.84 | 7.59 | - | throttled=0x0 |
| 8 | stacked_best | 3.84 | 5.82 | - | throttled=0x0 |
| 9 | ctx=512 | 3.83 | 7.55 | - | throttled=0x0 |
| 10 | ctx=2048 | 3.83 | 6.83 | - | throttled=0x0 |
| 11 | kv=q8_0 | 3.83 | 7.41 | - | throttled=0x0 |
| 12 | flash_attn | 3.83 | 7.6 | - | throttled=0x0 |
| 13 | ollama-default | 3.74 | 8.07 | - | throttled=0x0 |
| 14 | threads=2 | 3.0 | 3.91 | - | throttled=0x0 |
| 15 | speculative | 0.942 |  | 21.552 | throttled=0x0 |
