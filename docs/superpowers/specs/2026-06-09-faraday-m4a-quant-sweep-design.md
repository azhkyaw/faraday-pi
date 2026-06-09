# Faraday M4a — Quantization Sweep & Footprint Frontier
## Design Document

| | |
|---|---|
| **Status** | Approved (brainstorming) — ready for implementation planning |
| **Date** | 2026-06-09 |
| **Milestone** | M4a (first of three M4 sub-studies; extends main spec §9 "the rigor") |
| **Builds on** | M0–M3 (RAG core, serving, observability, on `main`) |

---

## 1. Overview

M4a is the first of three independent M4 "inference lab" sub-studies (M4a quant sweep · M4b RAG-quality evals · M4c optimization study). It builds a **reproducible benchmark harness** that, for each `(model size, quantization)` cell, measures footprint, speed, and quality on the Pi, then plots the **quality-vs-footprint Pareto frontier** and a Pi-4 leaderboard. The headline output is one chart that says, with evidence: *on a 4 GB Pi, run this size at this quant — here's why.*

The model axis caps at ~3B (the 4 GB ceiling); high-quant 3B cells that **don't fit** are a charted finding (the frontier visibly hits the RAM wall), not a gap.

## 2. Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Model axis | **Qwen2.5 {0.5B, 1.5B, 3B}** × quant ladder | A clean 2D size×quant frontier within one architecture; ~18 cells; answers "best size/quant for a 4 GB Pi" |
| Quant ladder | **Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K** | Standard K-quant rungs from near-lossless to aggressive |
| Quant source | **Download best-practice imatrix GGUFs** (e.g. bartowski on HF) | Measures the frontier people actually deploy; effort goes into the *harness*, not a quant pipeline |
| Quality metric | **Perplexity** (wikitext, fixed small chunk sample) via `llama-perplexity` | Cheap, standard, comparable; shows quant-degradation cliffs. Task/answer quality is M4b. |
| Memory metric | **Peak RSS** via `/usr/bin/time -v llama-bench` | Honest footprint (mmap'd weights hide from `free` — the M0 lesson) |
| Execution | **Serial** download → bench → delete, **resumable** | SD-card space + multi-hour overnight run on flaky hardware |

## 3. Goals / non-goals

**Goals**
- Per `(size, quant)` cell: GGUF disk size, peak RSS, prefill tok/s, decode tok/s, perplexity.
- A `sweep.csv`, a frontier PNG, a markdown leaderboard, and a findings writeup — committed.
- Resumable, reproducible harness; pure-Python parsing/plotting is unit-tested.

**Non-goals (YAGNI; later milestones)**
- Task-accuracy eval, energy/watts, multi-family models, self-quantization (M4a is download-only), and **all tuning levers** (governor/overclock/KV-cache/flash-attn/speculative → **M4c**), RAG answer quality (→ **M4b**).

## 4. Architecture

```
config (3 sizes × 6 quants = 18 cells)
  └─ for each cell  [on Pi, SERIAL, resumable: skip if already in CSV]:
       hf download GGUF
       → /usr/bin/time -v llama-bench   → prefill/decode tok/s + peak RSS
       → llama-perplexity (N chunks)    → perplexity
       → append row to results/sweep/sweep.csv
       → rm the GGUF   (keep ≤ ~1 model on disk)
  → scp sweep.csv back to dev machine
  → plot.py → frontier.png + leaderboard.md → commit
```

Results are **born on the Pi but committed from the dev machine** (the git-push deploy is one-way; the harness writes CSV/PNG on the Pi, then `scp` back to commit — the one workflow that reverses our usual direction).

## 5. Components

| Unit | Path | Responsibility |
|---|---|---|
| Bench parsers | `src/faraday/bench/parsers.py` | Pure fns: `parse_llama_bench(text)→(pp_tps, tg_tps)`, `parse_perplexity(text)→float`, `parse_time_v(text)→peak_rss_bytes`. The unit-tested core. |
| Sweep config | `src/faraday/bench/config.py` | The cell matrix: sizes, quant rungs, HF repo/filename templates, perplexity chunk count, the CSV schema/path. |
| Sweep orchestrator | `src/faraday/bench/sweep.py` | Per cell: download → bench → perplexity → record → delete; resumable (read CSV, skip done cells); robust to a cell failing (record `error`, continue). |
| Plotter | `src/faraday/bench/plot.py` | `sweep.csv` → frontier scatter (perplexity vs peak RSS, points labeled `size-quant`, non-fitting cells annotated) + leaderboard markdown. |
| On-Pi runner | `scripts/70_quant_sweep.sh` | Ensure GNU `time` + the wikitext sample present; run `python -m faraday.bench.sweep`. |
| Results | `results/sweep/` | `sweep.csv`, `frontier.png`, `leaderboard.md`, `findings.md` (committed). |

## 6. CSV schema (the durable record)

`results/sweep/sweep.csv`, one row per cell:
```
size, quant, status, disk_bytes, peak_rss_bytes, prefill_tps, decode_tps, perplexity, notes
```
`status` ∈ {`ok`, `oom`, `download_failed`, `error`}. A non-fitting 3B cell → `status=oom`, speed/ppl blank — this is data, not failure.

## 7. Data flow & resumability

`sweep.py` loads any existing `sweep.csv` into a set of completed `(size, quant)` keys and skips them. So a crash / reboot / power-off (which happened this session) loses **one** cell, never the run; re-running resumes. Each cell is independent — no shared state beyond the append-only CSV.

## 8. Error handling

- **Download fails** (404 / network) → row `status=download_failed`, continue.
- **OOM / won't fit** (3B high-quant) → caught (non-zero exit / time -v signal), row `status=oom`, continue — a *charted* result.
- **Perplexity/bench parse fails** → row `status=error` + raw saved to `results/sweep/raw/<cell>.log`, continue.
- **Disk guard**: assert free space before each download; always `rm` the GGUF in a `finally`.
- **Throttle hygiene**: record `vcgencmd get_throttled` per cell; flag any cell whose run was throttled (untrustworthy number).

## 9. Testing

- **Unit (dev machine, no Pi):** the three parsers in `parsers.py` against captured real sample outputs (a `llama-bench` table, a `llama-perplexity` tail, a `/usr/bin/time -v` block) → assert extracted numbers; `plot.py` smoke test (sample CSV → PNG file exists, leaderboard rows correct); `sweep.py` resumability (given a CSV with N done cells, the planned worklist excludes them) with download/bench stubbed.
- **Integration (Pi):** a **1-cell smoke** (smallest model, one quant) end-to-end before the full overnight run.
- TDD throughout; ruff clean. Author on Windows → `git push pi` → `pytest` on the Pi.

## 10. File structure (delta)

```
src/faraday/bench/
  __init__.py
  parsers.py     # NEW: llama-bench / perplexity / time -v parsers
  config.py      # NEW: cell matrix + CSV schema
  sweep.py       # NEW: resumable serial orchestrator
  plot.py        # NEW: frontier chart + leaderboard
tests/
  test_bench_parsers.py   # NEW
  test_bench_sweep.py     # NEW (resumability, stubbed)
  test_bench_plot.py      # NEW (smoke)
scripts/
  70_quant_sweep.sh       # NEW
results/sweep/            # NEW: sweep.csv, frontier.png, leaderboard.md, findings.md
pyproject.toml           # +matplotlib (dev/bench extra)
```

## 11. Definition of done

- The harness runs the 18-cell sweep on the Pi, resumably, producing `sweep.csv`.
- `frontier.png` (perplexity vs peak RSS, the Pareto frontier with the knee identifiable) + `leaderboard.md` + `findings.md` are committed.
- Non-fitting cells are charted as the 4 GB wall; throttled cells flagged.
- Parser/plot/resumability unit tests green; ruff clean.

## 12. Runtime note

Perplexity dominates runtime (each chunk = a forward pass); a fixed small chunk count keeps it comparable and tractable. The full sweep is an **overnight, resumable** run when the Pi is powered back on. The harness, spec, and plan are all built off-Pi first.

## 13. Deferred to the other M4 studies

- **M4c — optimization study:** governor, safe overclock, KV-cache quant, flash-attention, threads/batch, speculative decoding; the throughput waterfall; the TTFT-vs-context and prefill-rate leads from the M2 sanity reading.
- **M4b — RAG-quality evals:** recall@k, citation accuracy, LLM-judge answer quality, abstention, ablations; GBNF grammar citations slot in here or M5.
