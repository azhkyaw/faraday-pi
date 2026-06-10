# Faraday M5 — Polish & Ship
## Design Document

| | |
|---|---|
| **Status** | Approved (brainstorming) — ready for implementation planning |
| **Date** | 2026-06-10 |
| **Milestone** | M5 (final) — "the narrative" + ship-hardening |
| **Builds on** | M0–M3 (the product), M4a/M4b/M4c (the rigor — results feed Phase 2) |

---

## 1. Overview

M5 finishes the project. It has **one spec, two phases**:

- **Phase 1 — Ship-hardening (code):** the appliance boots itself, survives crashes,
  installs from scratch with one script, ships the last deferred architecture feature
  (GBNF grammar-constrained citations, with a measured before/after), and gains a
  retrieval-quality regression gate. Needs the Pi; does **not** need M4 results
  (except the GBNF re-measure, which needs M4b's baseline).
- **Phase 2 — The narrative (writing):** the technical report, demo GIF, final README,
  and the Pi-4 leaderboard — packaging four milestones of evidence into the artifacts a
  reviewer actually sees. Needs the M4 results; barely needs the Pi.

Definition of "shipped": a fresh Pi can be bootstrapped to a self-starting, crash-surviving
appliance; the repo leads with a demo and a report whose every claim links to a committed
measurement.

## 2. Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Structure | **One spec, two phases** (likely two plans) | The halves converge on one goal; report needs results, hardening doesn't |
| GBNF citations | **Build + measured before/after** | Re-run M4b's citation-validity at the baseline config with grammar ON → "prompting vs constrained decoding," evidence not assertion |
| Docker scope | **systemd-first + app-only Dockerfile** | Native llama-servers keep NEON tuning + RAM headroom on a 4 GB board; the app container is the portability proof; the report documents the judgment |
| Security scope | **Lightweight**: systemd sandboxing + LAN-bind notes | Original spec's non-goals (§43/§237): no auth/multi-user/hosted hardening — that's future work |
| Eval-as-test | **Include** (original spec §188) | A real regression gate (retrieval recall threshold) — distinctive rigor artifact, cheap once M4b exists |

## 3. Goals / non-goals

**Goals**
- Start-on-boot + restart-on-crash for gen server, embed server, and app (systemd) — the
  permanent fix for the M3 stale-process class of bugs.
- Startup memory guard (refuse / fall back with a clear log when the model won't fit).
- One-shot bootstrap: fresh Pi OS → running appliance.
- App Dockerfile + compose (runs anywhere, pointed at the Pi's servers).
- GBNF citation grammar, flag-gated, + the before/after citation-validity measurement.
- Retrieval-recall regression gate (integration-marked pytest with a threshold).
- `docs/report.md`, demo GIF, final README, polished Pi-4 leaderboard, future-work section.

**Non-goals (YAGNI / future work)**
- Auth, multi-user, hosted/hardened deployment (§237). Full Pi containerization (judged
  against — documented in the report). Energy/watts study, reranker, Pi 5/NPU, fine-tuning
  (§17 — they appear in the report's "next steps", not in code).

## 4. Phase 1 — Ship-hardening

### 4.1 systemd units (`deploy/systemd/`)

Three units, installed by the bootstrap and checked into the repo:

- `faraday-llama-gen.service` — llama-server :8080 (gen model), `Restart=on-failure`.
- `faraday-llama-embed.service` — llama-server :8081 (embeddings), `Restart=on-failure`.
- `faraday-app.service` — `faraday serve` :8000, `After=`/`Wants=` both server units,
  `ExecStartPre` = the memory-guard preflight (§4.2).

All three: `WantedBy=multi-user.target` (start on boot) + lightweight sandboxing
(`NoNewPrivileges=yes`, `ProtectSystem=full`, `PrivateTmp=yes`) — this is the M5 "security"
scope. Deploy integration: restart-on-deploy is a documented step (`systemctl restart
faraday-app` after `git push pi`), closing the stale-process gotcha permanently.

### 4.2 Startup memory guard (`src/faraday/preflight.py`)

`ExecStartPre` runs `python -m faraday.preflight`: read the configured GGUF's size +
required headroom, compare to `MemAvailable`; **fit** → proceed; **no fit but a smaller
quant exists in `models/`** → log + write the chosen model path to an environment file
(`/run/faraday/model.env`) that the unit's `ExecStart` reads via `EnvironmentFile=`
(graceful fallback); **nothing fits** → exit non-zero with a clear log line (the unit
fails visibly instead of OOM-ing).
The decision logic is a pure, unit-tested function (`fits(model_bytes, available_bytes,
headroom_bytes)` + `pick_model(candidates, available)`); only the `/proc/meminfo` read is
Pi-specific.

### 4.3 One-shot bootstrap (`scripts/bootstrap.sh`)

Fresh Pi OS → clone → build llama.cpp (`-j3`, the 4 GB rule) → download models → `pip
install -e ".[dev]"` → install + enable units → appliance live. Composes the existing
`00_pi_setup` / `10_build_llama` / `20_download_models` scripts; idempotent (safe re-run);
ends with a smoke check (`40_smoke_test.sh`). This is the "recruiter-clonable" promise.

### 4.4 App container (`Dockerfile`, `docker-compose.yml`)

Containerize **the FastAPI app only** (slim Python base, `pip install .`, env-pointed at
`FARADAY_GEN_URL`/`FARADAY_EMBED_URL`); compose wires it against the Pi's native servers.
Built and run on the dev box (Docker already proven there by the monitoring stack).
llama-servers stay native on the Pi — NEON-tuned build, no container RAM tax; the report
records this trade-off explicitly.

### 4.5 GBNF citations + before/after (`src/faraday/grammar.py`)

- `build_citation_grammar(n_sources) -> str`: a GBNF grammar generated **per request** —
  free prose plus citation tokens restricted to exactly `[1]`…`[n_sources]`, so an
  out-of-range citation is impossible *by construction* (vs merely discouraged by the
  prompt).
- Wiring: `Settings.use_grammar` flag → `RagEngine`/`llm_client` pass the grammar to
  llama-server. **Plan-time verification**: the grammar parameter on the OpenAI-compat
  endpoint (fallback: the native `/completion` endpoint, which accepts `grammar`).
- **Measurement**: re-run the M4b citation-validity metric at the baseline config
  (k4_c1200_o200) with grammar ON — one bounded Pi run (~20–30 min) — and report
  prompting-only vs grammar-constrained validity side by side.

### 4.6 Eval-as-test retrieval gate (`tests/test_retrieval_gate.py`)

Integration-marked pytest: embed the golden set's questions, run **retrieval only** (no
generation — fast), assert `recall@4 ≥ threshold`. The threshold is set from M4b's measured
baseline minus a small margin (exact value fixed at plan/closeout time from the real
scorecard). Guards future chunking/embedding changes with a real quality gate.

## 5. Phase 2 — The narrative

1. **Technical report** (`docs/report.md`, in-repo markdown, blog-ready): constraint →
   architecture → M4a frontier (chart) → M4b eval scorecard (+ GBNF before/after) → M4c
   waterfall + context curve → "what to run on a Pi 4" → lessons learned (engineering
   candor: the PSU/under-voltage saga, mmap vs RSS, the stale-process bug, `pkill` self-match)
   → next steps (from §17/§237).
2. **Demo GIF** (`docs/assets/demo.gif`): the web chat answering a document question with
   the Pi's network off (LAN-only — the airplane-mode flex). Recorded from the dev box;
   embedded at the top of the README.
3. **Final README pass**: hero GIF, results summary table (frontier knee · best decode ·
   eval scores · GBNF delta), quickstart = `bootstrap.sh`, roadmap all ✅, links refreshed.
4. **Pi-4 leaderboard** (`docs/pi4-leaderboard.md`): promote the M4a quant leaderboard +
   M4c optimization leaderboard into one polished, community-citable table with method notes.
5. **Final state sync**: CLAUDE.md State → shipped; memory updated; M4a-style as-built for M5.

## 6. Sequencing rules

- **Reboot/systemd testing must never overlap a benchmark or eval run** (it would kill a
  resumable run mid-cell and perturb measurements). Natural order: M4a closeout → M4b
  verify + run → M4c run → **Phase 1** → GBNF re-measure (needs M4b baseline; bounded) →
  **Phase 2**.
- Phase 1 items 4.1–4.4 + 4.6 have no dependency on M4 results and can slot into idle-board
  gaps if convenient; 4.5's re-measure and all of Phase 2 wait for results.

## 7. Error handling / risks

- **Grammar param support** on llama-server's OpenAI endpoint is verified at plan time;
  fallback is the native `/completion` route (both documented in llama.cpp).
- **Unit misconfiguration** fails visibly: `systemctl status` + memory-guard exit codes are
  the diagnostic path; the smoke test ends the bootstrap.
- **Fallback model absent**: the guard's fallback only triggers if a smaller GGUF exists;
  otherwise it refuses loudly (no silent OOM).
- **GIF/report tooling** is dev-box-only; no Pi risk.

## 8. Testing

- **Unit (no Pi):** `build_citation_grammar` (exact grammar text per n_sources, n=0 edge),
  preflight `fits`/`pick_model` logic, any report-table generators.
- **Integration (Pi):** the retrieval gate itself (4.6); GBNF live smoke (one grammar-ON
  request returns a valid `[k]`).
- **Behavioral verification (demonstrated, not unit-tested):** kill the app → systemd
  restarts it; reboot the Pi → all three services return; bootstrap on a clean checkout
  path; Docker build+run on the dev box. Results recorded in the M5 as-built.
- TDD for the pure logic; ruff clean throughout.

## 9. File structure (delta)

```
deploy/systemd/faraday-llama-gen.service     # NEW
deploy/systemd/faraday-llama-embed.service   # NEW
deploy/systemd/faraday-app.service           # NEW
scripts/bootstrap.sh                         # NEW: one-shot fresh-Pi install
Dockerfile                                   # NEW: app-only container
docker-compose.yml                           # NEW: app vs external servers
src/faraday/preflight.py                     # NEW: memory guard (pure core + main)
src/faraday/grammar.py                       # NEW: build_citation_grammar
src/faraday/config.py                        # +use_grammar flag
src/faraday/llm_client.py / rag.py           # grammar pass-through (flag-gated)
tests/test_preflight.py, test_grammar.py     # NEW (unit)
tests/test_retrieval_gate.py                 # NEW (integration; thresholded recall)
docs/report.md                               # NEW: the technical report
docs/assets/demo.gif                         # NEW
docs/pi4-leaderboard.md                      # NEW: polished community leaderboard
README.md                                    # final pass
```

## 10. Definition of done

- Crash + reboot survival **demonstrated** (app killed → returns; Pi rebooted → all
  services up without SSH-ing in).
- `bootstrap.sh` committed + documented (fresh-OS path), ending in a green smoke test.
- App container builds and serves against the Pi from the dev box.
- GBNF before/after measured and committed; invalid citations impossible with the flag on.
- Retrieval gate green at the committed threshold.
- `report.md`, demo GIF, final README, `pi4-leaderboard.md` committed; roadmap all ✅.
- M5 as-built written; CLAUDE.md/memory at "shipped".

## 11. Future work (the report's "next steps" — not built)

Reranker ablation; energy/tokens-per-watt (USB power meter); Pi 5 / NPU (Hailo, Coral)
step-change; fine-tuning a domain model; auth + multi-user + hardened hosted deployment;
full-Pi containerization revisited on 8 GB-class hardware.
