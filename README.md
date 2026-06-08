# Faraday

**A private RAG appliance on a Raspberry Pi 4 — ask questions about your own documents, get cited answers, with zero network egress.**

Faraday runs a small LLM and a vector-search engine *entirely* on a 4 GB Raspberry Pi 4. Point it at your PDFs/notes and it answers questions about them with source citations — and nothing ever leaves the device. It's both a working privacy-first appliance and an inference-engineering study of how much GenAI capability fits on ~$60 of constrained edge hardware.

> **Status:** 🟢 **M0 + M1 complete** — a local RAG appliance answering cited questions fully offline (22 tests green, proven on hardware).  🚧 **M2 next** — serving + web UI.

## Why it's interesting

- **Fully offline / private** — generation *and* embeddings run on-device; no cloud, no API calls at serve time. The privacy story *requires* the edge.
- **Engineered, not just assembled** — quantization, KV-cache, and memory are measured and optimized, not guessed. Every speed number gets paired with a measured quality number (M4).
- **Reproducible** — bare OS to running appliance via committed runbook scripts.

## Hardware & stack

- **Raspberry Pi 4 (4 GB)** · Cortex-A72 · Raspberry Pi OS Lite 64-bit (Debian 13)
- **llama.cpp** (CPU inference, NEON) serving **Qwen2.5-1.5B-Instruct** (Q4_K_M) + **bge-small-en-v1.5** embeddings
- **sqlite-vec** single-file vector store · **FastAPI** (M2) · **Python 3.13**

## Validated baseline (M0)

| | |
|---|---|
| Decode | 3.87 tok/s |
| Prefill | 7.71 tok/s |
| Throttling | none (`get_throttled=0x0`, full 1.5 GHz) |
| Honest RAM footprint | ~1.3 GB resident (gen + embed), ~2.4 GB free |

Detail: [results/baseline](results/baseline/README.md) · [M0 as-built & findings](docs/superpowers/plans/2026-06-08-faraday-m0-as-built.md)

## Reproduce M0

On a Pi 4 reachable over SSH (key auth + passwordless sudo), deploy the repo to the Pi, then on the Pi:

```bash
bash scripts/00_pi_setup.sh        # toolchain (build-essential, cmake, git, python venv)
bash scripts/10_build_llama.sh     # clone + build llama.cpp (NEON, -j3)
bash scripts/20_download_models.sh # fetch the gen + embedding GGUFs into a venv
bash scripts/30_run_servers.sh     # launch gen :8080 + embed :8081
bash scripts/40_smoke_test.sh      # verify both APIs respond
```

## Repo layout

```
docs/superpowers/specs/   design spec (the "what & why")
docs/superpowers/plans/   implementation plan + M0 as-built record
scripts/                  reproducible bring-up runbook (00 → 50)
results/                  benchmark + eval results
src/faraday/              the RAG application (M1+)
```

## Dev workflow

Code is authored on a Windows dev machine and deployed to the Pi with `git push pi` — the Pi hosts a repo configured with `receive.denyCurrentBranch=updateInstead`, so its working tree updates on every push, pinned to an exact commit. *Develop where you deploy.* See [`scripts/sync.ps1`](scripts/sync.ps1).

## Roadmap

| | Milestone | |
|---|---|---|
| **M0** | Bring-up: provisioning, llama.cpp build, local serving, validated baseline | ✅ |
| **M1** | RAG core: ingest → retrieve → ground → answer → verify-citations (CLI) | ✅ |
| **M2** | Serving: FastAPI + SSE + web UI + grammar-structured citations | 🚧 |
| **M3** | Observability: Prometheus + Grafana | |
| **M4** | The lab: quantization sweep, RAG evals, optimization study | |
| **M5** | Polish: technical report, demo, hardening | |

## Design docs

- [Design spec](docs/superpowers/specs/2026-06-08-faraday-edge-rag-appliance-design.md) — full architecture & methodology
- [Implementation plan (M0–M1)](docs/superpowers/plans/2026-06-08-faraday-m0-m1-rag-core.md)
- [M0 as-built & findings](docs/superpowers/plans/2026-06-08-faraday-m0-as-built.md)
- [M1 as-built & findings](docs/superpowers/plans/2026-06-08-faraday-m1-as-built.md)
