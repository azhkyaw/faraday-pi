# Faraday

**A private RAG appliance on a Raspberry Pi 4 — ask questions about your own documents, get cited answers, with zero network egress.**

Faraday runs a small LLM and a vector-search engine *entirely* on a 4 GB Raspberry Pi 4. Point it at your PDFs/notes and it answers questions about them with source citations — and nothing ever leaves the device. It's both a working privacy-first appliance and an inference-engineering study of how much GenAI capability fits on ~$60 of constrained edge hardware.

> **Status:** 🟢 **M0–M3 complete** — a local RAG appliance answering cited questions fully offline (CLI + token-streaming web chat), with a live Prometheus/Grafana dashboard (69 tests green, proven on hardware).  🚧 **M4 in progress** — the inference lab: the [quant-sweep frontier](results/sweep/findings.md) is measured (verdict: run **1.5B Q4_K_M** on a 4 GB Pi), the RAG-eval engine is merged, and the optimization study is designed.

## Why it's interesting

- **Fully offline / private** — generation *and* embeddings run on-device; no cloud, no API calls at serve time. The privacy story *requires* the edge.
- **Engineered, not just assembled** — quantization, KV-cache, and memory are measured and optimized, not guessed. Every speed number gets paired with a measured quality number (M4).
- **Reproducible** — bare OS to running appliance via committed runbook scripts.

## Hardware & stack

- **Raspberry Pi 4 (4 GB)** · Cortex-A72 · Raspberry Pi OS Lite 64-bit (Debian 13)
- **llama.cpp** (CPU inference, NEON) serving **Qwen2.5-1.5B-Instruct** (Q4_K_M) + **bge-small-en-v1.5** embeddings
- **sqlite-vec** single-file vector store · **FastAPI** (M2) · **Prometheus + Grafana** (M3) · **Python 3.13**

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

## Use it

Once the llama-servers are up (`scripts/30_run_servers.sh`), index documents and ask — two interfaces over the same offline RAG core:

```bash
# CLI (M1)
faraday ingest examples/corpus
faraday ask "What CPU does the Raspberry Pi 4 use?"
#   → "...a quad-core ARM Cortex-A72 (64-bit) CPU. [1]"  Sources: [1] pi-facts.md (0.855)

# Web app (M2) — token-streaming chat with live sources
bash scripts/60_run_app.sh         # serves on 0.0.0.0:8000
#   then open http://raspberrypi.local:8000 in a browser
```

The web app streams the answer token-by-token over SSE (`sources` → `token` → `done`), shows the retrieved sources with scores, and flags any hallucinated citations — all fully offline.

### Live monitoring (M3)

The Pi exposes Prometheus metrics (RAG: TTFT, decode tok/s, retrieval latency, citation validity; host: CPU temp, throttle, llama-server RSS) at `:8000/metrics`, plus the llama-servers' native metrics. Prometheus + Grafana run **off-Pi** (keeping the 4 GB budget free) via Docker Compose:

```bash
# on the dev machine, with Docker running:
docker compose -f monitoring/docker-compose.yml up -d
#   Prometheus targets → http://localhost:9090/targets   (all three UP)
#   Grafana dashboard  → http://localhost:3000           ("Faraday")
```

The dashboard renders the exact numbers Faraday previously measured by hand — TTFT, decode tok/s, Pi temp, RSS, hallucination counts — now continuous. See [`monitoring/README.md`](monitoring/README.md) (set your Pi's LAN IP in `prometheus.yml`; mDNS `.local` doesn't resolve inside containers).

## Repo layout

```
docs/superpowers/specs/   design specs (the "what & why")
docs/superpowers/plans/   implementation plans + M0–M3 as-built records
scripts/                  reproducible runbook (00 → 60: bring-up + run app)
monitoring/               off-Pi Prometheus + Grafana stack (Docker Compose)
results/                  benchmark + eval results
src/faraday/              the RAG application (engine, CLI, server, web UI, metrics)
```

## Dev workflow

Code is authored on a Windows dev machine and deployed to the Pi with `git push pi` — the Pi hosts a repo configured with `receive.denyCurrentBranch=updateInstead`, so its working tree updates on every push, pinned to an exact commit. *Develop where you deploy.* See [`scripts/sync.ps1`](scripts/sync.ps1).

## Roadmap

| | Milestone | |
|---|---|---|
| **M0** | Bring-up: provisioning, llama.cpp build, local serving, validated baseline | ✅ |
| **M1** | RAG core: ingest → retrieve → ground → answer → verify-citations (CLI) | ✅ |
| **M2** | Serving: FastAPI + SSE token streaming + web chat UI | ✅ |
| **M3** | Observability: Prometheus + Grafana (RAG + host metrics) | ✅ |
| **M4** | The lab: quantization sweep, RAG evals, optimization study, GBNF citations | 🚧 |
| **M5** | Polish: technical report, demo, hardening | |

## Design docs

- [Design spec](docs/superpowers/specs/2026-06-08-faraday-edge-rag-appliance-design.md) — full architecture & methodology
- Milestone specs: [M2 serving](docs/superpowers/specs/2026-06-09-faraday-m2-serving-design.md) · [M3 observability](docs/superpowers/specs/2026-06-09-faraday-m3-observability-design.md) · [M4a quant sweep](docs/superpowers/specs/2026-06-09-faraday-m4a-quant-sweep-design.md) · [M4b RAG evals](docs/superpowers/specs/2026-06-10-faraday-m4b-rag-evals-design.md) · [M4c optimization](docs/superpowers/specs/2026-06-10-faraday-m4c-optimization-design.md)
- Implementation plans: [M0–M1](docs/superpowers/plans/2026-06-08-faraday-m0-m1-rag-core.md) · [M2](docs/superpowers/plans/2026-06-09-faraday-m2-serving.md) · [M3](docs/superpowers/plans/2026-06-09-faraday-m3-observability.md) · [M4a](docs/superpowers/plans/2026-06-09-faraday-m4a-quant-sweep.md) · [M4b engine](docs/superpowers/plans/2026-06-10-faraday-m4b-rag-eval-engine.md) · [M4b data+run](docs/superpowers/plans/2026-06-10-faraday-m4b-rag-eval-data-and-run.md) · [M4c](docs/superpowers/plans/2026-06-10-faraday-m4c-optimization.md)
- First results: [M4a sweep findings](results/sweep/findings.md) — the quality-vs-footprint frontier + Pi-4 leaderboard
- As-built & findings: [M0](docs/superpowers/plans/2026-06-08-faraday-m0-as-built.md) · [M1](docs/superpowers/plans/2026-06-08-faraday-m1-as-built.md) · [M2](docs/superpowers/plans/2026-06-09-faraday-m2-as-built.md) · [M3](docs/superpowers/plans/2026-06-09-faraday-m3-as-built.md)
