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
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Work on a feature branch;
  don't commit milestone work straight to `main`.
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

- **M4a quant sweep** — harness + **core run done** (12 cells, 0.5B+1.5B); the
  quality/footprint Pareto frontier is final (knee = 1.5B Q4_K_M/Q5_K_M). A clean **re-run is
  unblocked** (proper PSU now verified — the old run under-volt-throttled the speed column and
  deferred the 3B row): `rm results/sweep/sweep.csv results/sweep/raw/*` then
  `scripts/70_quant_sweep.sh` → clean 18-cell speed + 3B, then refresh `findings.md`.
- **M4b RAG evals** — the `faraday.eval` engine (dataset · metrics recall@k/MRR/citation/
  abstention · Claude-as-judge · resumable runner · scorecard+ablation report) is built+tested
  (69 tests). **Plan 2 written (`bf87672`) but unexecuted**: Apollo-era corpus fetcher +
  Claude golden-set generator + `runner.run`/`scripts/80_run_evals.sh` + `report.main`. Its
  code tasks run inline anytime; the **data tasks need `ANTHROPIC_API_KEY`** (Claude draft +
  judge) + golden-set curation + a ~3–4 h Pi run (judge at the baseline config only).
- **M4c optimization** — not yet designed.

Per-milestone detail (specs/plans/as-builts) in `docs/superpowers/`. **M5** (final —
"polish & ship") = technical report tying the M4 studies together + demo + README/leaderboard,
**plus** hardening (systemd auto-start/restart-on-crash — the M3 stale-process fix — Docker
packaging, security) + the GBNF citations deferred from M2.
