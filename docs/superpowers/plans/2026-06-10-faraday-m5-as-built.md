# Faraday M5 — Polish & Ship — As-Built

**Status:** ✅ **COMPLETE — shipped (v1.0).** Phase 1 (hardening) fully delivered and
live-verified; Phase 2 (narrative) delivered, with the airplane-mode demo GIF deferred to a
user recording (recipe below). Branch `m5-polish-and-ship`, merged to `main`.

Plan: [2026-06-10-faraday-m5-polish-and-ship.md](./2026-06-10-faraday-m5-polish-and-ship.md) ·
Spec: [../specs/2026-06-10-faraday-m5-polish-and-ship-design.md](../specs/2026-06-10-faraday-m5-polish-and-ship-design.md)

## Delivered

**Phase 1 — ship-hardening (all live-verified on the Pi):**
- **GBNF grammar-constrained citations** (the M2-deferred feature): a pure per-request grammar
  builder (`grammar.py`), threaded through the `LLMClient` Protocol + `RagEngine` via DI, behind
  a `FARADAY_USE_GRAMMAR` flag wired at both edges (`cli.py`/`server.py`). 8 unit tests.
- **Startup memory guard** (`preflight.py`): pure `fits`/`pick_model` + a `/proc/meminfo` read
  that writes `/run/faraday/model.env` or exits 1 loudly. Gen systemd unit runs it as
  `ExecStartPre`.
- **systemd units** (`deploy/systemd/`): gen → embed → app with dependency ordering, sandboxing
  (`NoNewPrivileges`/`PrivateTmp`/`ProtectSystem=full`), `Restart=on-failure`, + an idempotent
  installer.
- **One-shot bootstrap** (`scripts/bootstrap.sh`): fresh Pi → running, reboot-proof appliance.
- **App Docker image** (`Dockerfile` + compose + `.dockerignore` + a pyproject package-data fix):
  the app containerizes; the llama-servers stay native.
- **Retrieval regression gate** (`tests/test_retrieval_gate.py`): eval-as-test, recall@4 ≥ 0.70.

**Phase 2 — narrative:**
- **Technical report** (`docs/report.md`, ~1850 words, 9 sections, every claim linked to an
  artifact); **Pi-4 leaderboard** (`docs/pi4-leaderboard.md`); **final README** (shipped state).
- **Demo GIF** — deferred (recipe below); README hero is a documented placeholder.

## Verified (transcripts)

- **GBNF live smoke** — `1 passed`: one grammar-ON request to llama-server's OAI endpoint;
  every `[...]` in the output was a valid citation. **The §7 contingency (switch to
  `/completion`) was not needed** — the OAI chat endpoint honors the `grammar` param.
- **GBNF before/after** ([gbnf_before_after.md](../../../results/evals/gbnf_before_after.md)) —
  citation validity **1.000 → 1.000** (now structural/guaranteed, not emergent); recall@k 0.805
  and abstention 0.872 **unchanged** → the grammar constrains without distorting.
- **systemd crash survival** — `RESTART-OK`: `systemctl kill -s SIGKILL faraday-app` → systemd
  restarted it; `/healthz` green ~9 s later.
- **systemd boot survival** — `BOOT-OK`: full `sudo reboot` → watcher saw the Pi drop
  (`saw_down=1`) then all three services answer their own health checks ~25 s after boot, **no
  SSH intervention**. This is the demonstrated fix for the M3 stale-process class.
- **bootstrap** — idempotent re-run on the live install: stages skipped/passed, smoke green
  (gen `reply: Ready`, embed `384 dims`, `RAM 442Mi/3.7Gi`).
- **Docker** — image builds; container serves `{"status":"ok"}` on `/healthz` and the full chat
  HTML on `/` (proving the package-data fix bundled `static/index.html`).
- **Retrieval gate** — `1 passed`, recall@4 ≥ 0.70 (took ~15 min — corpus ingest + 41 query
  embeds, slower than the plan's ~2–3 min estimate; fine for an on-demand release gate).
- **Full suite** — 118 unit tests pass, 7 integration deselected, `ruff` clean, `0x0` throughout.

## Findings & deliberate calls

- **The GBNF result is "guaranteed, not fixed."** M4b already measured citation_validity = 1.0,
  so the number didn't move — but its *meaning* did: emergent (prompt-dependent) → structural
  (an out-of-range citation is undecodable). The live smoke is the qualitative proof; the
  measurement confirms no quality cost.
- **Five plan deviations, each flagged:** (1) the gen systemd unit adds `--no-cache-prompt
  --cache-ram 512` — the M4b prompt-cache fix the plan predates, critical for an *always-on*
  box; (2) `parse_speculative`/CLI fixes carried from M4c; (3) the inline `FakeHttpLLM` in
  `test_eval_runner` needed the new `grammar` kwarg (a Protocol change ripples to *implementers*,
  not just callers — defaults only save callers); (4) the retrieval gate threshold tightened to
  0.70 (M4b closed, so the rule applies) instead of the 0.50 sanity floor; (5) `results/evals/
  raw_grammar/` is **kept** (the after-data behind the before/after), per the data-persistence
  convention, not ignored.
- **The airplane-mode `iptables` rule needed a loopback exemption.** The plan's
  `-A OUTPUT ! -d <lan> -j REJECT` would also block `127.0.0.1`, killing the app↔llama-server
  calls. Corrected with `-A OUTPUT -o lo -j ACCEPT` first.

## Demo GIF recipe (for the deferred recording)

1. Index docs: `cd ~/faraday && . .venv/bin/activate && faraday ingest examples/corpus`.
2. Sever the internet, keep LAN + loopback:
   `sudo iptables -A OUTPUT -o lo -j ACCEPT && sudo iptables -A OUTPUT ! -d 192.168.100.0/24 -j REJECT`.
3. Film: on the Pi `curl --max-time 5 https://example.com` **fails**; the browser at
   `http://192.168.100.59:8000` still streams a cited answer (e.g. "What CPU does the Raspberry
   Pi 4 use?" → "...Cortex-A72... [1]").
4. Record ≤15 s / ≤5 MB → `docs/assets/demo.gif` (ScreenToGif, or
   `ffmpeg -i in.mp4 -vf "fps=10,scale=800:-1" docs/assets/demo.gif`).
5. Restore: `sudo iptables -D OUTPUT ! -d 192.168.100.0/24 -j REJECT && sudo iptables -D OUTPUT -o lo -j ACCEPT`.

## Operational change

The appliance is now **systemd-managed** — gen/embed/app auto-start on boot and restart on
crash. The M3-era "nothing auto-starts; re-run `30_run_servers.sh`" gotcha is retired; the new
rule after deploying app code is `sudo systemctl restart faraday-app`.
