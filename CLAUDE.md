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
- Run: `scripts/30_run_servers.sh` (servers), `40_smoke_test.sh` (health), `60_run_app.sh`
  (web app); `faraday ingest|ask|serve` (CLI); monitoring via
  `docker compose -f monitoring/docker-compose.yml up -d` (dev machine, Docker).

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
- **Measurement hygiene**: check `vcgencmd get_throttled` (0x0 = healthy) before trusting
  benchmarks; read process RSS, not `free` "used" (mmap'd weights hide in buff/cache).
- **mDNS `.local` doesn't resolve inside Docker** — Prometheus scrapes the Pi by LAN IP.

## State

M0–M3 complete (bring-up · RAG core+CLI · streaming web chat · observability); M4 (the
inference lab) next. Per-milestone detail in `docs/superpowers/`.
