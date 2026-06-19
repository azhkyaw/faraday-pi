# Faraday Pi

**A private RAG appliance on a Raspberry Pi 4 — ask questions about your own documents, get cited answers, with zero network egress.**

<p align="center">
  <img src="docs/assets/demo-web.gif" alt="Faraday Pi answering a document question fully offline, streaming a cited answer" width="820">
</p>

Faraday runs a small LLM and a vector-search engine *entirely* on a 4 GB Raspberry Pi 4. Point it at your PDFs/notes and it answers questions about them with source citations — and nothing ever leaves the device. It's both a working privacy-first appliance and an inference-engineering study of how much GenAI capability fits on ~$60 of constrained edge hardware.

> **Status:** 🟢 **Shipped (M0–M5 complete).** A private, always-on RAG appliance: cited answers fully offline (CLI + streaming web chat), grammar-guaranteed citations, systemd boot/crash survival, one-command bootstrap, and the full inference study behind every design choice. Read the **[technical report](docs/report.md)**.

## Results at a glance

Every number is measured on hardware (`get_throttled=0x0`) and links its raw artifact.

| Question | Answer (measured) | Study |
|---|---|---|
| **Which model?** | **1.5B Q4_K_M** — the quality/footprint knee: 11.32 ppl, 1.07 GB, 3.86 tok/s | [M4a](results/sweep/findings.md) |
| **How fast can it go?** | ~3.9 tok/s decode — a **memory-bandwidth ceiling**; no CPU lever beats it, speculative decoding is 4× *slower* on CPU | [M4c](results/optimize/findings.md) |
| **Is it any good?** | recall@4 ~0.80, faithfulness **4.22**/5, correctness **3.95**/5 (chunk 1200 / top_k 4) | [M4b](results/evals/findings.md) |
| **Can it hallucinate citations?** | No — GBNF makes an out-of-range citation **undecodable**; validity **1.000 by construction** | [GBNF](results/evals/gbnf_before_after.md) |
| **Does it survive a power cut?** | Yes — systemd auto-recovers all services (`RESTART-OK` + `BOOT-OK`), no intervention | M5 |

The one-line recommendation for *anyone* putting an LLM on a Pi 4: see the **[Pi-4 leaderboard](docs/pi4-leaderboard.md)**.

## What sets it apart

- **Fully offline / private** — generation *and* embeddings run on-device; no cloud, no API calls at serve time. The privacy story *requires* the edge.
- **Engineered, not assembled** — quantization, retrieval, throughput, and memory are *measured*, not guessed; every speed number is paired with a measured quality number.
- **Reproducible & honest** — bare OS to running appliance in one command; raw audit data for every benchmark is committed, so any result re-derives with no re-run.
- **Shipped, not just demoed** — systemd boot/crash survival, a startup memory guard, and an app container; pull the plug and it answers questions again on its own, no SSH.

## Hardware & stack

- **Raspberry Pi 4 (4 GB)** · Cortex-A72 · Raspberry Pi OS Lite 64-bit
- **llama.cpp** (CPU, NEON) serving **Qwen2.5-1.5B-Instruct Q4_K_M** (gen) + **bge-small-en-v1.5** (embeddings)
- **sqlite-vec** single-file vector store · **FastAPI** + SSE streaming · **systemd** (always-on) · **Prometheus + Grafana** (off-Pi) · **Python 3.11+**

## Quickstart

On a fresh Raspberry Pi OS (64-bit), clone the repo to `~/faraday` and run one command:

```bash
git clone https://github.com/azhkyaw/faraday-pi ~/faraday && cd ~/faraday
bash scripts/bootstrap.sh
```

That installs deps, builds llama.cpp, fetches the models, sets up the venv, installs the three **systemd** units, and smoke-tests — leaving Faraday **live on `:8000` and surviving reboots**. Then index documents and ask:

```bash
faraday ingest examples/corpus
faraday ask "What CPU does the Raspberry Pi 4 use?"
#   → "...a quad-core ARM Cortex-A72 (64-bit) CPU. [1]"   Sources: [1] pi-facts.md (0.855)
```

![Faraday answering with the Pi's internet physically severed — generation and embeddings run on-device, still citing its source](docs/assets/demo-cli.gif)

…or open the streaming web chat at `http://raspberrypi.local:8000` — it streams tokens over SSE (`sources → token → done`), shows retrieved sources with scores, and (with `FARADAY_USE_GRAMMAR=1`) constrains citations to valid sources by construction.

**Run the app off-Pi (Docker):** the FastAPI app containerizes (the llama-servers stay native on the Pi); `FARADAY_PI_HOST=<pi-ip> docker compose up --build`.

## Operating it

The servers are **systemd-managed** — they start on boot and restart on crash:

```bash
sudo systemctl status faraday-app faraday-llama-gen faraday-llama-embed
# after pushing code that changes the app:
sudo systemctl restart faraday-app
```

### Live monitoring

The Pi exposes Prometheus metrics (RAG: TTFT, decode tok/s, retrieval latency, citation validity; host: CPU temp, throttle, RSS) at `:8000/metrics`. Prometheus + Grafana run **off-Pi** (keeping the 4 GB budget free) via Docker Compose — see [`monitoring/README.md`](monitoring/README.md).

## Repo layout

```
docs/report.md            the engineering write-up (start here)
docs/pi4-leaderboard.md   what runs well on a Pi 4, measured
docs/superpowers/         design specs, plans, per-milestone as-builts
scripts/                  bootstrap + reproducible runbook (bring-up → studies)
deploy/systemd/           the three units + installer
results/                  benchmark + eval results (with raw audit data)
src/faraday/              the RAG application (engine, CLI, server, web UI, metrics, grammar)
```

## Dev workflow

Code is authored on a Windows dev machine and deployed to the Pi with `git push pi` — the Pi hosts a repo configured `receive.denyCurrentBranch=updateInstead`, so its working tree updates on every push. *Develop where you deploy*, then `sudo systemctl restart faraday-app`.

## Roadmap

| | Milestone | |
|---|---|---|
| **M0** | Bring-up: provisioning, llama.cpp build, local serving, validated baseline | ✅ |
| **M1** | RAG core: ingest → retrieve → ground → answer → verify-citations (CLI) | ✅ |
| **M2** | Serving: FastAPI + SSE token streaming + web chat UI | ✅ |
| **M3** | Observability: Prometheus + Grafana (RAG + host metrics) | ✅ |
| **M4** | The lab: quant sweep (M4a) · RAG evals (M4b) · optimization study (M4c) | ✅ |
| **M5** | Ship: GBNF citations, systemd hardening, bootstrap, Docker, report, leaderboard | ✅ |

## Design docs

- **[Technical report](docs/report.md)** — the full engineering narrative · **[Pi-4 leaderboard](docs/pi4-leaderboard.md)** — measured model menu
- [Design spec](docs/superpowers/specs/2026-06-08-faraday-edge-rag-appliance-design.md) — architecture & methodology
- Milestone specs: [M2](docs/superpowers/specs/2026-06-09-faraday-m2-serving-design.md) · [M3](docs/superpowers/specs/2026-06-09-faraday-m3-observability-design.md) · [M4a](docs/superpowers/specs/2026-06-09-faraday-m4a-quant-sweep-design.md) · [M4b](docs/superpowers/specs/2026-06-10-faraday-m4b-rag-evals-design.md) · [M4c](docs/superpowers/specs/2026-06-10-faraday-m4c-optimization-design.md) · [M5](docs/superpowers/specs/2026-06-10-faraday-m5-polish-and-ship-design.md)
- Findings: [M4a quant sweep](results/sweep/findings.md) · [M4b RAG evals](results/evals/findings.md) · [M4c optimization](results/optimize/findings.md)
