# CLAUDE.md

Faraday — an air-gapped RAG appliance on a Raspberry Pi 4 (4GB). Product overview in
README.md; specs, plans, and per-milestone as-built records in docs/superpowers/.

## Dev/deploy loop (read first)

Code is **authored on Windows** (`C:\projects\piai`) and **runs on the Pi** — never run
the app or tests on Windows (`sqlite-vec`'s native extension won't load there).

- Deploy: `git push pi <branch>` — the Pi repo (`~/faraday`) has
  `receive.denyCurrentBranch=updateInstead`, so its working tree updates on each push.
- Test/run: `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest -q"`.
- Cycle: edit on Windows → commit → `git push pi <branch>` → `ssh pi "... pytest ..."`.

## Pi facts

- `pi@raspberrypi.local` (LAN IP `192.168.100.59`); SSH key auth + passwordless sudo set up.
- venv `~/faraday/.venv` (`pip install -e ".[dev]"`); llama.cpp at `~/llama.cpp/build/bin/`.
- Ports: gen llama-server `:8080`, embed `:8081`, FastAPI app `:8000`.
- Models in `~/faraday/models`: Qwen2.5-1.5B-Instruct Q4_K_M (gen) + bge-small-en-v1.5 f16
  (embed, 384-dim). Default model is 1.5B for the 4GB board; 3B is M4-exploratory only.
- Run: `scripts/30_run_servers.sh` (servers), `40_smoke_test.sh` (health), `60_run_app.sh`
  (web app), `70_quant_sweep.sh` (M4a quant sweep, on the Pi); `faraday ingest|ask|serve`
  (CLI); monitoring via `docker compose -f monitoring/docker-compose.yml up -d` (dev
  machine, Docker).

## Conventions

- **TDD**: test → impl → verify green on the Pi → commit per unit. Use `tests/conftest.py`
  fakes (FakeEmbedder/FakeLLM); most tests need no server.
- **Lint**: `ruff check src tests` must pass (line-length 100; imports at top — E402).
- **Integration tests**: `@pytest.mark.integration` (deselected by default); run on the Pi
  with servers up via `pytest -m integration`.
- **Commits**: conventional (`feat(m3): …`), end with
  `Co-Authored-By: Claude <noreply@anthropic.com>` (model-agnostic — the project has been
  built across model/account switches). Work on a feature branch; don't commit milestone
  work straight to `main`.
- **Architecture**: components depend on `Protocol`s (`Embedder`, `LLMClient`) — real HTTP
  impls injected by `cli.py`/`server.py`, fakes by tests. Streaming flows through a
  `Sources`→`Token`→`Done` event seam (also the metrics instrumentation point).

## Gotchas (learned the hard way)

- **Nested quotes don't survive PowerShell→ssh→bash.** Don't put inline Python/awk with
  `"`/`'`/commas in an SSH command. Instead: commit a `.sh`/`.py` and run it, use filename
  args (`python3 -m json.tool <file>`), or pipe a file to `python3 -`.
- **`pkill -f foo` matches itself over SSH** and kills the session — use `pkill -f '[f]oo'`.
- **Long-running Pi processes serve stale code** until restarted (`git push` updates files,
  not a running process's memory) — restart `faraday serve` after server changes.
- **Build llama.cpp with `-j3`, not `-j4`** — four parallel compilers OOM the 4GB board.
- **The Pi's llama.cpp build has only `llama-bench`/`-cli`/`-server`**, not
  `llama-perplexity` (M4a's quality axis needs it). Build a missing tool against the
  existing libs: `cmake --build ~/llama.cpp/build --target llama-perplexity -j3`.
- **Measurement hygiene**: check `vcgencmd get_throttled` (0x0 = healthy) before trusting
  benchmarks; read process RSS, not `free` "used" (mmap'd weights hide in buff/cache).
- **mDNS `.local` doesn't resolve inside Docker** — Prometheus scrapes the Pi by LAN IP.
- **The app/servers aren't daemons** — nothing auto-starts on boot, so after a Pi
  reboot/power-cycle re-run `scripts/30_run_servers.sh` (+ start the app). To keep the app
  alive after you close *your* SSH session, start it detached:
  `setsid nohup faraday serve --host 0.0.0.0 --port 8000 >/tmp/app.log 2>&1 </dev/null &`
  (this does NOT survive a Pi shutdown — only start-on-boot does). Start-on-boot +
  restart-on-crash via a systemd unit is the M5 hardening item.

## State

M0–M3 complete (bring-up · RAG core+CLI · streaming web chat · observability). **M4** (the
inference lab) in progress, all on `main`:

- **M4a quant sweep — ✅ COMPLETE, signed off 2026-06-10.** Final 18-cell artifacts +
  findings in `results/sweep/` (clean run, `0x0` across ~15 h). Verdict: knee = **1.5B
  Q4_K_M**; decode is bandwidth-bound (`≈3.8 GB/s ÷ model_bytes`, measured 18 ways);
  prefill is kernel-bound (Q8_0/Q4_K_M fastest); 3B fits in RAM but fails interactivity
  (1.92 tok/s) and is broken below Q4_K_M (its Q3/Q2 are dominated by 1.5B cells).
- **M4b RAG evals** — `faraday.eval` engine merged (`91a77ad`). Plan 2 code **batch-verified
  on the Pi 2026-06-12** on branch `m4b-eval-data-run` (83 tests + ruff green at `4674865`;
  audit fixes applied `2375a04`: judge now sees chunk text, ingest once per chunk-size,
  abstention judge cross-check per spec §10, script chmod; `main` merged in). **Remaining:
  the data tasks** — corpus fetch on the Pi, Claude golden-set draft + hand-curation, the
  ~3–4 h Pi run, judge scoring — which **need `ANTHROPIC_API_KEY`** (ask the user first;
  judge at baseline config only, for cost).
- **M4c optimization** — **fully designed: spec (`fbdbf51`) + plan (`35ae99a`); pending
  execution** (needs a quiet board — sequence after the M4a closeout + M4b run).
  Ablate-then-stack tuning waterfall (governor/threads/batch/KV-quant/flash-attn/overclock)
  + speculative decoding + Ollama baseline + TTFT-vs-context, on 1.5B Q4_K_M, extending
  `faraday.bench`.

Per-milestone detail (specs/plans/as-builts) in `docs/superpowers/`. **M5** (final —
"polish & ship") = technical report tying the M4 studies together + demo + README/leaderboard,
**plus** hardening (systemd auto-start/restart-on-crash — the M3 stale-process fix — Docker
packaging, security) + the GBNF citations deferred from M2. **M5 is fully designed**
(spec `2a63501` + plan `fd69961`; 15 tasks, two gated phases; reboot/systemd tests must
never overlap benchmark runs) — with M4a–c planned too, **everything remaining in the
project is execute-only**: M4b data+run → M4c run → M5. The board is free (no run in
flight); the Pi worktree is on branch `m4b-eval-data-run`.
