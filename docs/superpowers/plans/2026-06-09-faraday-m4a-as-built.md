# Faraday — M4a As-Built & Findings (harness)

**Status:** ✅ Harness complete — merged to `main` (`51a1a67`) and on GitHub. ⏳ The 18-cell
sweep is **unrun**; `results/sweep/{sweep.csv,frontier.png,leaderboard.md,findings.md}` and
the "M4a complete" sign-off follow the overnight run.
**Plan:** [2026-06-09-faraday-m4a-quant-sweep.md](./2026-06-09-faraday-m4a-quant-sweep.md)
**Spec:** [2026-06-09-faraday-m4a-quant-sweep-design.md](../specs/2026-06-09-faraday-m4a-quant-sweep-design.md)

A reproducible benchmark harness for the quality-vs-footprint frontier of Qwen2.5
{0.5B, 1.5B, 3B} × {Q8_0…Q2_K} on a 4 GB Pi 4 — built test-first, off the hardware where
possible.

## Delivered

- **`src/faraday/bench/`**: `parsers.py` (pure `llama-bench`/`perplexity`/`time -v`
  parsers), `config.py` (the 18-cell matrix + CSV schema + bartowski naming), `sweep.py`
  (resumable serial orchestrator with an **injected subprocess runner**), `plot.py`
  (Pareto frontier PNG + ranked leaderboard, headless Agg).
- **`scripts/70_quant_sweep.sh`**: on-Pi runner — ensures GNU `time` + the wikitext corpus,
  sets PATH, prints the throttle state, runs the sweep then the plot.
- **`results/sweep/README.md`**: artifact guide. **`matplotlib`** added to the `dev` extra.
- **Tests**: 20 new unit tests (parsers/config/resumable-core/runner+main/plot) + a live
  1-cell integration smoke. Full suite **57 passed, 4 deselected**, ruff clean.

## Verified

```
Unit (Pi):        57 passed, 4 deselected (integration), ruff clean on src+tests
Integration (Pi): test_one_cell_end_to_end  → 1 passed in 840s (0:14:00)
                  (real hf download → time -v llama-bench → llama-perplexity, 0.5B-Q4_K_M)
Merge:            f1896d2..51a1a67 fast-forward; tests re-verified green on main
```

## Findings worth keeping

1. **A harness is only as real as the binaries it shells out to.** The Pi's llama.cpp build
   had **only `llama-bench`/`-cli`/`-server` — no `llama-perplexity`**, the entire quality
   axis. Caught by *verifying prerequisites before launching the overnight run*, not three
   cells deep at 2 a.m. Fixed: `cmake --build ~/llama.cpp/build --target llama-perplexity
   -j3` (the libs were already built; respecting the 4 GB `-j3` rule). Now a CLAUDE.md
   gotcha.
2. **Perplexity dominates runtime → ~14 min/cell on 0.5B → ~8–10 h for the full sweep.**
   Each chunk is a forward pass, so cost scales ~`tokens/prefill_rate`: cheap on 0.5B,
   ~22 min on 1.5B, ~45+ min on 3B. This is precisely why the design chose **serial +
   resumable**; the abstract decision met a concrete number at the smoke.
3. **Injected `Runner` = a network/subprocess orchestrator that unit-tests offline.** The
   fake runner even writes a real 1 KB file on "download", so `gguf.stat()` and the
   `finally: unlink()` cleanup exercise true filesystem behavior; all four `status` paths
   (`ok`/`download_failed`/`oom`/`error`) are reachable with zero hardware. Same Protocol-DI
   seam as `Embedder`/`LLMClient`.
4. **The M3-era `.gitignore` already fit.** `*.log` + `results/raw/` mean per-cell raw logs
   are auto-ignored while curated CSV/PNG/MD commit — **no `.gitignore` change needed**.
5. **Prereq friction the runner now absorbs**: GNU `time` wasn't installed (the shell
   builtin has no `-v`); the wikitext getter writes `wikitext-2-raw/wiki.test.raw` and needs
   normalizing to the config path.

## Deliberate calls

- **Download best-practice imatrix GGUFs (bartowski), don't self-quantize** — measures the
  frontier people actually deploy; effort goes into the harness, not a quant pipeline.
- **Perplexity (wikitext) as the quality axis** — cheap, standard, comparable; shows
  quant cliffs. Task/answer quality is M4b's job, not this.
- **Download into a `models/bench/` scratch dir, delete per cell** — protects the existing
  gen model and the SD card; `*.gguf` is already git-ignored.
- **Non-fitting 3B cells are `status=oom` — charted data, not failures** (the 4 GB wall).

## Process notes

Strict per-task TDD over the Pi loop produced a clean **red→green** history (every
`test(...) (red)` precedes its `feat(...)`) — kept as evidence by fast-forward (no squash).
The plan was authored fully off-Pi; only the smoke + (pending) sweep need hardware. One
plan-vs-reality gap (finding #1) was assumed-present-but-absent — caught at the prereq
gate, the cheapest possible place.

**Next:** run `scripts/70_quant_sweep.sh` (overnight, resumable) → commit the frontier +
`findings.md` (the knee, quant cliffs, the 4 GB wall, what to actually run on a 4 GB Pi) →
sign off M4a. Then M4c (optimization waterfall: governor/overclock/KV-cache/flash-attn/
speculative; the TTFT-vs-context + prefill-rate leads) and M4b (RAG evals).
