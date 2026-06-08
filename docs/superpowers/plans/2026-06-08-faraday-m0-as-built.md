# Faraday — M0 As-Built & Findings

**Status:** ✅ M0 complete — 2026-06-08
**Plan:** [2026-06-08-faraday-m0-m1-rag-core.md](./2026-06-08-faraday-m0-m1-rag-core.md) (Milestone M0)
**Branch:** `m0-m1-rag-core`

The authoritative record of what M0 (Pi bring-up) *actually* required on real hardware — deliverables, validated measurements, findings worth keeping, and where reality diverged from the plan (and why). The plan captured intent; this captures execution.

## What M0 delivered

A running, fully-offline local LLM on a Raspberry Pi 4 (4 GB), reproducible from a bare OS via committed scripts:

- Two `llama-server` instances — generation (Qwen2.5-1.5B Q4_K_M) on `:8080`, embeddings (bge-small-en-v1.5 f16) on `:8081`, both OpenAI-compatible.
- A Windows→Pi **git-push deploy pipeline** (the Pi pinned to an exact commit).
- Reproducible runbook scripts `00`→`50`.
- A throttle-validated performance baseline.

## Validated baseline

See [results/baseline/README.md](../../../results/baseline/README.md).

| Metric | Value |
|---|---|
| Prefill (pp128) | 7.71 tok/s |
| Decode (tg64) | 3.87 tok/s |
| Throttling | **none** (`get_throttled=0x0`) |
| ARM clock under load | 1.5 GHz (full) |
| Temp under load | 38.4 °C |
| Engine | llama.cpp `715b86a`, NEON, GCC 14.2.0 |

Trustworthy — verified un-throttled and at full clock, not depressed by under-voltage.

## Findings worth keeping (seed the M4 study)

1. **The A72 has no `dotprod`/`i8mm`.** The Cortex-A72 predates the int8 matmul instructions (A76+), so quantized inference falls back to plain NEON. Prefill (matmul-heavy) suffers more than decode — hence pp128 is only ~2× tg64. We can't add the instructions; M4 gains must come from governor, overclock, KV-cache quant, flash-attention, speculative decoding.

2. **`free` "used" undercounts a model's footprint by ~3×.** llama.cpp `mmap`s the GGUF, so weights sit in file-backed page cache (`buff/cache`), not `used`. Measured side by side:
   - `free` "used": **437 MiB** (anonymous only — KV-cache/buffers)
   - gen process **RSS: 1245 MiB** (honest footprint ≈ 1.1 GB weights + ~120 MB buffers)
   - `free` "buff/cache": 2.5 GiB (where the weights actually live)

   The right metric is **peak process RSS/PSS under load** (`scripts/50_mem_report.sh`), not `free`. Honest total ≈ 1.3 GB resident, leaving ~2.4 GB headroom.

3. **Memory is "soft" vs "hard."** mmap'd weights are evictable (re-readable from the file → degrade gracefully). The KV-cache is anonymous → un-evictable → must fit or OOM. So the real 4 GB constraint is **context length** (KV-cache size), not the weights — directly informing M1's context budget and M4's longer-context experiments.

4. **Measurement hygiene is non-negotiable.** Check `get_throttled` before trusting any benchmark (an undervolted Pi silently halves numbers), and read RSS not `free` for footprint. Both guard against tools reporting convenient-but-wrong numbers.

## Deviations from the plan (and why)

| Plan assumed | As-built | Why |
|---|---|---|
| `rsync` sync (`sync.sh`) | git-push deploy (`sync.ps1`) | rsync absent on Windows; git-push also pins the Pi to an exact commit |
| `huggingface-cli download` | `hf download` | huggingface-hub v1.x removed the legacy CLI |
| `pip install --user` | project **venv** | Debian 13 is PEP 668 "externally managed" — global pip refused |
| Python 3.11 (Bookworm) | Python 3.13 (trixie) | current Pi OS Lite ships Debian 13 |
| *(unanticipated)* | NOPASSWD sudo bootstrap | non-interactive SSH automation can't type a sudo password |
| `-j$(nproc)` build | `-j3` build | leave memory headroom on 4 GB so the OOM-killer doesn't abort the compile |
| scripts `00`–`30` | added `40_smoke_test.sh`, `50_mem_report.sh` | reusable health check + honest-memory diagnostic |

## Scripts (the reproducible runbook)

| Script | Runs on | Purpose |
|---|---|---|
| `00_pi_setup.sh` | Pi | apt toolchain install |
| `10_build_llama.sh` | Pi | clone + build llama.cpp (NEON, `-j3`) |
| `20_download_models.sh` | Pi | venv + `hf download` the gen/embed GGUFs |
| `30_run_servers.sh` | Pi | launch both llama-servers (nohup) |
| `40_smoke_test.sh` | Pi | wait-for-health + gen/embed API smoke test |
| `50_mem_report.sh` | Pi | RSS/PSS vs `free` memory diagnostic |
| `sync.ps1` | Windows | `git push pi` deploy |

Reproduce M0 (Pi reachable, SSH key + NOPASSWD sudo set): run `00`→`40` on the Pi.

## M0 task completion

- [x] **Task 0** — Pi access, toolchain, git-push sync
- [x] **Task 1** — build llama.cpp (`715b86a`, NEON, `-j3`)
- [x] **Task 2** — download models (Qwen2.5-1.5B Q4_K_M, bge-small f16)
- [x] **Task 3** — run servers + smoke-test (gen `:8080`, embed `:8081`)
- [x] **Task 4** — record validated baseline

**Next:** M1 — the RAG core (`ingest → retrieve → ground → answer → verify-citations`), test-first.
