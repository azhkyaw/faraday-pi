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
- Run: **`scripts/bootstrap.sh`** (fresh Pi → running appliance in one shot; installs the
  **systemd** units so servers auto-start on boot — `systemctl status/restart faraday-app`).
  Studies: `70_quant_sweep.sh` (M4a) · `90_optimize.sh` (M4c) · `80_run_evals.sh` (M4b) ·
  `95_gbnf_measure.sh` (GBNF). Also `40_smoke_test.sh` (health), `30_run_servers.sh` (manual
  servers for a quiet-board benchmark); `faraday ingest|ask|serve` (CLI); monitoring via
  `docker compose -f monitoring/docker-compose.yml up -d` (dev machine, Docker).

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
- **Persist benchmark raw data**: commit the raw audit trail behind every benchmark (M4a's
  per-cell `*.log` with ppl ±stderr + provenance; M4b's `raw/` rows + frozen `judge/`
  verdicts; M4c's per-cell `raw/*.log` with llama-bench `±` + `time -v`) so analysis re-runs
  with no Pi run / no API re-spend — gitignore only regenerable binaries (`*.sqlite` stores,
  model GGUFs) + pre-curation drafts. The **unit is the per-cell measurement**, not the
  orchestration run-log (transient narrative — M4c's `/tmp` sweep log was 97% download-progress
  spam and held nothing the per-cell logs didn't). `.log` is globally ignored, so kept logs
  need a negation (`!results/sweep/raw/*.log`, `!results/optimize/raw/*.log`) — write it in
  `.gitignore` *before* the run so a plain `git add results/<x>/` sweeps them in (M4c baked it
  in up front; M4b had to add the raw data after the fact).

## Gotchas (learned the hard way)

- **Nested quotes don't survive PowerShell→ssh→bash.** Don't put inline Python/awk with
  `"`/`'`/commas in an SSH command. Instead: commit a `.sh`/`.py` and run it, use filename
  args (`python3 -m json.tool <file>`), or pipe a file to `python3 -`.
- **`pkill -f foo` matches itself over SSH** and kills the session — use `pkill -f '[f]oo'`.
- **Long-running Pi processes serve stale code** until restarted (`git push` updates files,
  not a running process's memory) — `sudo systemctl restart faraday-app` after app changes
  (servers are systemd-managed since M5).
- **Build llama.cpp with `-j3`, not `-j4`** — four parallel compilers OOM the 4GB board.
- **The Pi's llama.cpp build has only `llama-bench`/`-cli`/`-server`**, not
  `llama-perplexity` (M4a's quality axis needs it). Build a missing tool against the
  existing libs: `cmake --build ~/llama.cpp/build --target llama-perplexity -j3`.
- **The Pi's llama.cpp build drifts ahead of any plan's assumed CLI — verify flags against
  live `--help`/`-h`, not docs.** M4c hit three arg/parser drifts: `llama-bench` now *requires*
  an arg for `-fa` (`-fa on`; bare `-fa` is rejected and prints usage → `parse_llama_bench`
  sees no `pp/tg` rows); `llama-speculative` removed `--draft-max` (use `--spec-draft-n-max`);
  and real `llama-speculative` prints *two* `speed:` lines (an `encoded`/prefill line + a
  `decoded` line) so a naive "first `speed:`" regex records prefill as decode. A rejected flag
  fails *fast* with a tiny RSS (~11 MB, no model load) — a tidy tell. `run_cell`'s try/except
  turns each into a recorded `notes` row (not a run-killer), and the persisted `raw/` log let
  the parser bug be fixed by **re-parsing, not re-benchmarking**.
- **Measurement hygiene**: check `vcgencmd get_throttled` (0x0 = healthy) before trusting
  benchmarks; read process RSS, not `free` "used" (mmap'd weights hide in buff/cache). For a
  quiet board, stop competitors — `sudo systemctl stop faraday-* ollama`: **`ollama` runs as a
  service since M4c** and is the one persistent background competitor. The demo recorders
  (`asciinema`/`agg`, installed 2026-06-19) are *invoke-only* — dormant unless recording, in
  system Python isolated from the app venv — so they don't compete and leave no benchmark
  residue (the recorder severs the net via `ip route`, not a firewall rule, and restores it on
  exit; verified `0x0` + route restored after the demo run).
- **mDNS `.local` doesn't resolve inside Docker** — Prometheus scrapes the Pi by LAN IP.
- **llama-server sends nothing until a request finishes** (embeddings, non-streaming chat) —
  so an httpx read timeout measures the server's *total compute time* for the request.
  Bound per-request work (`HttpEmbedder` batches 16 texts/POST) and size timeouts from
  measurement, not vibes (eval runner: 1800 s). *Four* M4b crashes shared the
  `ReadTimeout` symptom with four different root causes; a *fifth* was a hard embed-server
  **HTTP 500** (not a timeout) — always read the *server's* log too (it names the cause).
- **The embed model has a hard token ceiling — llama-server *rejects* over-long input
  (HTTP 500), it does not truncate.** bge-small-en-v1.5 has 512 trained positions and embeds
  each input in one physical batch (`n_ubatch=512`), so a chunk over 512 tokens 500s the
  request (`input is too large to process`) and crashes ingest (the 5th M4b crash, at the
  c2400 cells). `HttpEmbedder` clips each input to `max_input_chars=1450` *before* the POST
  (only the embedded view; stored chunk text stays full for generation). Budget set by
  **measuring all 409 c2400 chunks** via `/tokenize` (`add_special=true`): max 718 tok, mean
  513, **52% exceed 512**; 1450 chars keeps every chunk ≤490. Finding: a chunk_size past the
  embedder's context is self-defeating (retrieval ignores the tail). Don't tune the budget to
  one sample — the run dies on the *first* breach, hiding the denser chunks behind it.
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
- **The servers are systemd-managed (M5)** — `faraday-llama-gen`/`-embed`/`-app` auto-start on
  boot and restart on crash (verified `RESTART-OK` + `BOOT-OK`). Install/repair with
  `bash deploy/systemd/install.sh`; check `systemctl status faraday-app`. **After `git push pi`
  of app code the running app serves stale code — `sudo systemctl restart faraday-app`.** (The
  pre-M5 manual launcher `scripts/30_run_servers.sh` still works for a quiet-board benchmark
  where you don't want the app competing — `sudo systemctl stop faraday-*` first.)

## State

**M0–M5 COMPLETE — all on `main`; demo GIFs shipped; repo renamed **Faraday Pi**; `v1.0` held at user's discretion** (bring-up · RAG core+CLI · streaming web chat ·
observability · the inference lab M4a/b/c · ship-hardening + GBNF + narrative M5). All on `main`:

- **M4a quant sweep — ✅ COMPLETE, signed off 2026-06-10.** Final 18-cell artifacts +
  findings + per-cell raw logs (ppl ±stderr, build hash, `time -v`) in `results/sweep/`
  (clean run, `0x0` across ~15 h). Verdict: knee = **1.5B
  Q4_K_M**; decode is bandwidth-bound (`≈3.8 GB/s ÷ model_bytes`, measured 18 ways);
  prefill is kernel-bound (Q8_0/Q4_K_M fastest); 3B fits in RAM but fails interactivity
  (1.92 tok/s) and is broken below Q4_K_M (its Q3/Q2 are dominated by 1.5B cells).
- **M4b RAG evals** — engine merged (`91a77ad`), batch-verified + audit fixes (`2375a04`).
  Data done 2026-06-12: Apollo corpus (`abd911c`, 15 articles) + curated golden set
  (`e51c082`: 41 answerable with machine-verified source spans + 6 abstention items; 3 of
  the plan's hand-authored "unanswerable" questions were answerable from the corpus and
  were replaced with grep-verified-absent ones). **✅ COMPLETE 2026-06-16 — full 9-cell grid, 423 rows banked, `0x0` throughout.** FIVE
  crashes root-caused along the way — the first four as
  `httpx.ReadTimeout`: unbounded embed batches (`78c7574`), gen ctx too small for k8_c2400 +
  interactive timeout (`5395037`, GEN_CTX=8192), timeout re-sized to *measured* deep prefill
  (`348e278`, 1800 s), prompt-cache decode-collapse (`0260b76`, `--no-cache-prompt
  --cache-ram 512`); the fifth a hard embed **HTTP 500** — c2400 chunks over bge's 512-token
  limit, clipped to a *measured* 1450 chars (`ba5879a`→`cda2f8f`; 52% of c2400 chunks exceed
  512). Scored + committed: deterministic metrics all 9 cells + Claude judge at the baseline
  (`k4_c1200_o200`). **Verdict: chunk 1200 / top_k 4** (appliance baseline validated).
  Headline: **c2400 has the worst recall at every top_k despite the biggest chunks** — 52%
  of c2400 chunks exceed bge's 512-token limit, so truncated embeddings miss tail spans (a
  chunk size past the embedder's context is self-defeating); top_k knee at k4; answer quality
  faithfulness 4.22 / correctness 3.95; abstention is the weak spot (declines only 2/6
  out-of-corpus Qs → motivates the M5 GBNF cite-or-abstain work). Artifacts in
  `results/evals/`: scorecard.md, ablations.png, findings.md + the raw rows (3 MB) + frozen
  judge caches, all committed for reproducibility (re-score via `report.py`, no Pi run / no
  API re-spend; `ANTHROPIC_API_KEY` lives in `~/.faraday_env` on the Pi — source, never echo).
- **M4c optimization — ✅ COMPLETE 2026-06-17.** Resumable ablate-then-stack harness
  (`faraday.bench.optimize*`, 18 unit tests) + 15-cell sweep on 1.5B Q4_K_M (clean, `0x0`
  throughout). **Verdict: the appliance already ships at its throughput optimum — no CPU
  lever beats baseline decode (~3.9 tok/s).** Decode is memory-bandwidth-bound (flat 3.83–3.91
  across governor/threads/batch/KV-quant/flash-attn *and* context depth — confirms M4a's
  `≈3.8 GB/s ÷ model_bytes` 15 more ways); prefill is compute-bound (collapses 7.74→6.22 over
  ctx 128→4096). **Speculative decoding is counterproductive on CPU** (0.942 tok/s, ~4× slower,
  21.6% draft accept — the GPU technique inverts on a bandwidth-bound board); Ollama −4%;
  `stacked_best` (`-t 3`) is net-negative (starves prefill). THREE live-tooling CLI/parser
  drifts caught + fixed (llama-bench needs `-fa on`; llama-speculative `--draft-max`→
  `--spec-draft-n-max`; `parse_speculative` must read the `decoded` line, not the `encoded`
  prefill line — re-derived from the persisted raw log, **no re-benchmark**). Artifacts in
  `results/optimize/`: optimize.csv, leaderboard.md, 3 plots, findings.md + per-cell raw logs
  (re-plot via `optimize_plot`, no Pi run). Overclock left as a separate reboot-gated manual
  step (not run — keeps the shipped clock as baseline of record).

- **M5 polish & ship — ✅ COMPLETE 2026-06-17 (merged to `main`; `v1.0` tag held pending the demo GIF).** Phase 1 (hardening): **GBNF
  citations** (per-request grammar → `LLMClient`/`RagEngine` DI → `FARADAY_USE_GRAMMAR` flag;
  live-proven the OAI endpoint honors `grammar`; before/after = validity **1.000 by
  construction**, recall 0.805 / abstention 0.872 unchanged); **startup memory guard**
  (`preflight.py`); **systemd units** (gen→embed→app, sandboxed) — live-verified `RESTART-OK`
  (SIGKILL→restart) + `BOOT-OK` (reboot→all 3 auto-recover), the M3 stale-process fix;
  **one-shot `bootstrap.sh`**; **app Docker image** (+ package-data fix); **retrieval recall
  gate** (≥0.70, eval-as-test, ~15 min). Phase 2: **`docs/report.md`** (engineering write-up),
  **`docs/pi4-leaderboard.md`**, **final README**. **Demo GIFs shipped post-M5** (2026-06-19,
  `6ec68c9`): airplane-mode **CLI** (asciinema+agg) + streaming **web UI** (Playwright→ffmpeg),
  both reproducible from committed recorders (`scripts/97_record_demo.sh`, `scripts/web-demo/`);
  README web-hero + CLI-at-quickstart; web UI `<h1>`/`<title>` → "Faraday Pi". 118 unit tests
  pass, `0x0` throughout. As-built:
  `docs/superpowers/plans/2026-06-10-faraday-m5-as-built.md`.

Per-milestone detail (specs/plans/as-builts) in `docs/superpowers/`. **The project is merged to
`main`, functionally complete, demo GIFs shipped; the `v1.0` tag is now held only at the user's
discretion** (nothing blocks it — to cut: `git tag -a v1.0 && git push origin v1.0`). The GitHub
repo is renamed **`faraday-pi`** (front-door + README + app `<h1>` = "Faraday Pi"; package/CLI/env/systemd
stay `faraday`); the Pi is on `main`, **systemd-managed** (gen/embed/app auto-start on boot, restart on
crash) and has `ollama` + the 0.5B draft + a built `llama-speculative` from the M4c run. Open
follow-ups: cut `v1.0` whenever; the optional reboot-gated overclock study; and the
"next steps" in `docs/report.md` §9 (reranker, energy axis, Pi 5/NPU). M4/M5 verdict: the
appliance is *already throughput-optimal* (the decode ceiling is physics).
