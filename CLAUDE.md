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
- **llama-server sends nothing until a request finishes** (embeddings, non-streaming chat) —
  so an httpx read timeout measures the server's *total compute time* for the request.
  Bound per-request work (`HttpEmbedder` batches 16 texts/POST) and size timeouts from
  measurement, not vibes (eval runner: 1800 s). *Four* M4b crashes shared the
  `ReadTimeout` symptom with four different root causes — read the *server's* log too.
- **llama-server's prompt cache (`--cache-ram`) defaults to an 8192 MiB ceiling — bigger
  than the 4 GB board.** It grows ~20 MiB per distinct prompt; over a long batch of unique
  prompts it reached 2.4 GB, evicted the *mmap'd model weights* from page cache, and
  collapsed decode to 0.07 tok/s (vs ~3.6) — a slow-burn cliff that hit at ~5 k requests
  and killed the M4b run on an *innocent small* config. `30_run_servers.sh` now launches
  gen with `--no-cache-prompt --cache-ram 512`. Symptom of weight eviction: `free` shows
  `buff/cache` below the model size + decode `tg` tanks in the gen log; a server restart
  reclaims it instantly (proof it's cache, not a leak). Not thermal (`0x0`), not OOM-kill.
- **Prefill collapses with context depth**: ~40 tok/s on short prompts → 6.75 tok/s at
  ~4.3 k tokens (1.5B Q4_K_M). Price long-context batch runs by prompt tokens, not per
  question (k8_c2400 ≈ 13 min/answer). The eval grid's biggest cell also needs
  `GEN_CTX=8192` (exported by `80_run_evals.sh`; appliance default 4096).
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
- **M4b RAG evals** — engine merged (`91a77ad`), batch-verified + audit fixes (`2375a04`).
  Data done 2026-06-12: Apollo corpus (`abd911c`, 15 articles) + curated golden set
  (`e51c082`: 41 answerable with machine-verified source spans + 6 abstention items; 3 of
  the plan's hand-authored "unanswerable" questions were answerable from the corpus and
  were replaced with grep-verified-absent ones). **Run IN FLIGHT since 2026-06-13**
  (attempt 3, code `348e278`) after three launch failures, all surfacing as
  `httpx.ReadTimeout` with distinct root causes: unbounded per-document embed batches
  (`78c7574`), gen context too small for k8_c2400 + interactive-sized LLM timeout
  (`5395037`, GEN_CTX=8192), then timeout re-sized to *measured* deep prefill (`348e278`,
  1800 s). Revised ETA **~20–27 h** — deep-context prefill is 6.75 tok/s, so k8_c2400
  ≈ 13 min/question (the plan's 3–4 h estimate assumed flat ~30 s/q). Scope choice offered
  to the user (trim deep cells?); default = full grid; resumable per (config, question).
  Then: judge scoring at the baseline config only (`ANTHROPIC_API_KEY` lives in
  `~/.faraday_env` on the Pi — source it, never echo it) → scorecard/ablations/findings →
  merge to main.
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
project is execute-only**: M4b run (in flight) → M4c run → M5. **The board is BUSY** with
the M4b eval run (started 2026-06-13, ~20–27 h, resumable — don't start other Pi work);
the Pi worktree is on branch `m4b-eval-data-run`.
