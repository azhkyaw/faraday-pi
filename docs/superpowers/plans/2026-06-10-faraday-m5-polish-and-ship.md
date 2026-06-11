# Faraday M5 — Polish & Ship — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Faraday — systemd-managed always-on appliance (boot/crash survival + memory guard + one-shot bootstrap + app container), GBNF grammar-constrained citations with a measured before/after, a retrieval regression gate, and the narrative artifacts (technical report, airplane-mode demo GIF, final README, Pi-4 leaderboard).

**Architecture:** Phase 1 (Tasks 1–10) is code: a pure grammar builder wired through the `LLMClient` Protocol behind a `Settings.use_grammar` flag; a preflight memory guard whose pure core is unit-tested and whose env-file output feeds the gen unit; three systemd units + bootstrap; an app-only Docker image; an integration recall gate; and the GBNF re-measure via the M4b harness. Phase 2 (Tasks 11–15) is the narrative, written from committed artifacts. Reboot/systemd testing never overlaps benchmark/eval runs.

**Tech Stack:** Python 3.11+, systemd, Docker (dev box), llama.cpp GBNF grammars, the merged `faraday.eval` harness, pytest, ruff.

**Spec:** [../specs/2026-06-10-faraday-m5-polish-and-ship-design.md](../specs/2026-06-10-faraday-m5-polish-and-ship-design.md)

**Verified interfaces:** `RagEngine(retriever, llm, top_k, max_tokens)` built at `cli.py:31` and `server.py:28` (`make_engine`); `LLMClient` Protocol = `complete(messages, max_tokens=512) -> str` / `stream(messages, max_tokens=512) -> Iterator[str]`; `HttpLLMClient` posts `{"messages", "max_tokens", "stream"}` to `/v1/chat/completions`; conftest `FakeLLM` implements both; `Settings` frozen dataclass with `from_env` (gen_url/embed_url/db_path only); web UI = single `src/faraday/static/index.html`; `/healthz` already exists (`server.py:60` — spec §180 item pre-satisfied). llama-server: `/completion` documents `grammar`; the OAI chat endpoint supports `/completion`-specific extension params (grammar passed there; integration smoke proves it; **contingency** = switch the grammar path to native `/completion`).

**Sequencing gates:**
- Tasks 1–9 need the Pi for pytest only (any quiet moment); **systemd live verification (Task 5 step 4) and reboot tests must NOT overlap a benchmark/eval run.**
- Task 10 (GBNF re-measure) requires the `m4b-eval-data-run` branch **merged** and the M4b baseline run complete.
- Tasks 11–15 (narrative) require M4a/M4b/M4c results committed.

---

## Phase 1 — Ship-hardening

### Task 1: `grammar.py` — citation grammar builder (TDD)

**Files:**
- Create: `src/faraday/grammar.py`
- Test: `tests/test_grammar.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_grammar.py`:

```python
from faraday.grammar import build_citation_grammar


def test_grammar_enumerates_exactly_the_retrieved_indices():
    g = build_citation_grammar(4)
    assert '"[1]"' in g and '"[4]"' in g
    assert '"[5]"' not in g
    assert "root ::=" in g and "cite ::=" in g


def test_grammar_text_rule_blocks_free_brackets():
    # prose may be anything except '[', so the only way to emit '[' is a valid cite
    assert "[^\\[]" in build_citation_grammar(2)


def test_grammar_zero_sources_allows_no_citations():
    g = build_citation_grammar(0)
    assert "cite" not in g
    assert "root ::=" in g
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: faraday.grammar`):

`ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_grammar.py -q"`

- [ ] **Step 3: Implement.** Create `src/faraday/grammar.py`:

```python
"""GBNF grammar for citation-constrained decoding (the M2-deferred feature).

The grammar is generated PER REQUEST from the number of retrieved sources, so a
citation token can only ever be one of [1]..[n_sources] — an out-of-range citation
becomes impossible by construction (vs merely discouraged by the prompt). Prose is
unconstrained except that a bare '[' cannot appear outside a valid citation.
"""
from __future__ import annotations


def build_citation_grammar(n_sources: int) -> str:
    if n_sources <= 0:
        return "root ::= [^\\[]*\n"
    cites = " | ".join(f'"[{i}]"' for i in range(1, n_sources + 1))
    return (
        "root ::= ( text | cite )*\n"
        "text ::= [^\\[]+\n"
        f"cite ::= {cites}\n"
    )
```

- [ ] **Step 4: Run — expect PASS** (3 passed). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_grammar.py -q && ruff check src/faraday/grammar.py tests/test_grammar.py"
git add src/faraday/grammar.py tests/test_grammar.py
git commit -m "feat(m5): per-request GBNF citation grammar builder"
```

---

### Task 2: grammar pass-through — `LLMClient` + `RagEngine` (TDD)

**Files:**
- Modify: `src/faraday/llm_client.py` (Protocol + HttpLLMClient)
- Modify: `src/faraday/rag.py` (grammar_builder DI)
- Modify: `tests/conftest.py` (FakeLLM accepts/records grammar)
- Test: `tests/test_rag_grammar.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_rag_grammar.py`:

```python
from faraday.grammar import build_citation_grammar
from faraday.models import Chunk, RetrievedChunk
from faraday.rag import RagEngine


class FakeRetriever:
    def search(self, query, k=4):
        mk = lambda i: RetrievedChunk(  # noqa: E731
            chunk=Chunk(doc_id="d", ord=i, text=f"t{i}", source="s.txt"), score=0.9)
        return [mk(0), mk(1)]


def test_engine_builds_grammar_from_n_sources(fake_llm):
    eng = RagEngine(FakeRetriever(), fake_llm, top_k=2,
                    grammar_builder=build_citation_grammar)
    eng.answer("q")
    assert fake_llm.last_grammar is not None
    assert '"[2]"' in fake_llm.last_grammar and '"[3]"' not in fake_llm.last_grammar


def test_engine_passes_none_without_builder(fake_llm):
    eng = RagEngine(FakeRetriever(), fake_llm, top_k=2)
    eng.answer("q")
    assert fake_llm.last_grammar is None


def test_stream_also_carries_grammar(fake_llm):
    eng = RagEngine(FakeRetriever(), fake_llm, top_k=2,
                    grammar_builder=build_citation_grammar)
    list(eng.answer_stream("q"))
    assert fake_llm.last_grammar is not None
```

- [ ] **Step 2: Run — expect FAIL** (`TypeError: unexpected keyword 'grammar_builder'`).

- [ ] **Step 3: Implement (three edits).**

`src/faraday/llm_client.py` — Protocol + client gain an optional `grammar` kwarg:

```python
@runtime_checkable
class LLMClient(Protocol):
    def complete(self, messages: list[dict], max_tokens: int = 512,
                 grammar: str | None = None) -> str: ...
    def stream(self, messages: list[dict], max_tokens: int = 512,
               grammar: str | None = None) -> Iterator[str]: ...
```

`HttpLLMClient.complete` / `.stream` — build the payload once, add grammar only when set
(llama-server accepts `/completion`-style extension params on the OAI endpoint):

```python
    def complete(self, messages: list[dict], max_tokens: int = 512,
                 grammar: str | None = None) -> str:
        payload = {"messages": messages, "max_tokens": max_tokens, "stream": False}
        if grammar:
            payload["grammar"] = grammar
        resp = self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def stream(self, messages: list[dict], max_tokens: int = 512,
               grammar: str | None = None) -> Iterator[str]:
        payload = {"messages": messages, "max_tokens": max_tokens, "stream": True}
        if grammar:
            payload["grammar"] = grammar
        with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            yield from _tokens_from_sse(resp.iter_lines())
```

`src/faraday/rag.py` — DI a builder, call it with the retrieved count:

```python
from typing import Callable, Iterator


class RagEngine:
    def __init__(self, retriever: Retriever, llm: LLMClient, top_k: int = 4,
                 max_tokens: int = 512,
                 grammar_builder: Callable[[int], str] | None = None):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.grammar_builder = grammar_builder

    def answer(self, query: str) -> Answer:
        sources = self.retriever.search(query, k=self.top_k)
        messages = build_messages(query, sources)
        grammar = self.grammar_builder(len(sources)) if self.grammar_builder else None
        text = self.llm.complete(messages, max_tokens=self.max_tokens, grammar=grammar)
        valid, invalid = classify_citations(text, n_sources=len(sources))
        return Answer(text=text, sources=sources,
                      cited_indices=valid, invalid_citations=invalid)

    def answer_stream(self, query: str) -> Iterator[Event]:
        sources = self.retriever.search(query, k=self.top_k)
        yield SourcesEvent(sources)
        messages = build_messages(query, sources)
        grammar = self.grammar_builder(len(sources)) if self.grammar_builder else None
        parts: list[str] = []
        for token in self.llm.stream(messages, max_tokens=self.max_tokens, grammar=grammar):
            parts.append(token)
            yield TokenEvent(token)
        valid, invalid = classify_citations("".join(parts), n_sources=len(sources))
        yield DoneEvent(cited_indices=valid, invalid_citations=invalid)
```

`tests/conftest.py` — FakeLLM records the grammar (existing callers unaffected):

```python
class FakeLLM:
    def __init__(self, reply: str = "Answer [1]."):
        self.reply = reply
        self.last_messages = None
        self.last_grammar = None

    def complete(self, messages: list[dict], max_tokens: int = 512,
                 grammar: str | None = None) -> str:
        self.last_messages = messages
        self.last_grammar = grammar
        return self.reply

    def stream(self, messages: list[dict], max_tokens: int = 512,
               grammar: str | None = None):
        self.last_messages = messages
        self.last_grammar = grammar
        mid = len(self.reply) // 2          # two chunks that rejoin to the exact reply
        yield self.reply[:mid]
        yield self.reply[mid:]
```

- [ ] **Step 4: Run new + full suite — expect PASS** (no regressions; grammar defaults keep old call sites green):

`ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_rag_grammar.py -q && pytest -q"`

- [ ] **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && ruff check src tests"
git add src/faraday/llm_client.py src/faraday/rag.py tests/conftest.py tests/test_rag_grammar.py
git commit -m "feat(m5): grammar pass-through (LLMClient kwarg + RagEngine grammar_builder DI)"
```

---

### Task 3: `Settings.use_grammar` flag + edge wiring (TDD)

**Files:**
- Modify: `src/faraday/config.py`, `src/faraday/cli.py:31-32`, `src/faraday/server.py:25-30`
- Test: `tests/test_grammar_wiring.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_grammar_wiring.py`:

```python
from faraday.config import Settings
from faraday.grammar import build_citation_grammar
from faraday.server import make_engine


def test_settings_reads_use_grammar_env(monkeypatch):
    monkeypatch.setenv("FARADAY_USE_GRAMMAR", "1")
    assert Settings.from_env().use_grammar is True
    monkeypatch.delenv("FARADAY_USE_GRAMMAR")
    assert Settings.from_env().use_grammar is False


def test_make_engine_wires_grammar_builder(tmp_path):
    on = Settings(db_path=str(tmp_path / "a.sqlite"), use_grammar=True)
    engine, store = make_engine(on)
    assert engine.grammar_builder is build_citation_grammar
    store.close()
    off = Settings(db_path=str(tmp_path / "b.sqlite"))
    engine, store = make_engine(off)
    assert engine.grammar_builder is None
    store.close()
```

- [ ] **Step 2: Run — expect FAIL** (`TypeError: unexpected keyword 'use_grammar'`).

- [ ] **Step 3: Implement.** `config.py` — add the field + env read:

```python
    use_grammar: bool = False    # GBNF citation-constrained decoding (M5)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            gen_url=os.environ.get("FARADAY_GEN_URL", cls.gen_url),
            embed_url=os.environ.get("FARADAY_EMBED_URL", cls.embed_url),
            db_path=os.environ.get("FARADAY_DB", cls.db_path),
            use_grammar=os.environ.get("FARADAY_USE_GRAMMAR", "").lower() in ("1", "true"),
        )
```

`server.py make_engine` (and the identical pattern in `cli.py ask`):

```python
def make_engine(settings: Settings):
    """Build a per-request engine + store (caller closes the store)."""
    from faraday.grammar import build_citation_grammar
    store = SqliteVecStore(settings.db_path, dim=settings.embed_dim)
    gb = build_citation_grammar if settings.use_grammar else None
    engine = RagEngine(Retriever(HttpEmbedder(settings), store), HttpLLMClient(settings),
                       top_k=settings.top_k, max_tokens=settings.max_tokens,
                       grammar_builder=gb)
    return engine, store
```

In `cli.py` `ask`, replace the `RagEngine(...)` construction with:

```python
    from faraday.grammar import build_citation_grammar
    gb = build_citation_grammar if s.use_grammar else None
    engine = RagEngine(Retriever(HttpEmbedder(s), store), HttpLLMClient(s),
                       top_k=s.top_k, max_tokens=s.max_tokens, grammar_builder=gb)
```

- [ ] **Step 4: Run — expect PASS** (2 passed; full suite still green). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_grammar_wiring.py -q && pytest -q && ruff check src tests"
git add src/faraday/config.py src/faraday/server.py src/faraday/cli.py tests/test_grammar_wiring.py
git commit -m "feat(m5): FARADAY_USE_GRAMMAR flag wired through cli + server"
```

---

### Task 4: `preflight.py` — startup memory guard (TDD)

**Files:**
- Create: `src/faraday/preflight.py`
- Test: `tests/test_preflight.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_preflight.py`:

```python
from pathlib import Path

from faraday.preflight import HEADROOM_BYTES, fits, pick_model

MB = 1024 * 1024


def test_fits_requires_model_plus_headroom():
    assert fits(1000 * MB, available_bytes=1000 * MB + HEADROOM_BYTES) is True
    assert fits(1000 * MB, available_bytes=1000 * MB + HEADROOM_BYTES - 1) is False


def test_pick_model_prefers_largest_that_fits():
    cands = [(Path("big.gguf"), 3000 * MB), (Path("mid.gguf"), 1000 * MB),
             (Path("small.gguf"), 500 * MB)]
    assert pick_model(cands, available_bytes=1200 * MB + HEADROOM_BYTES).name == "mid.gguf"


def test_pick_model_none_when_nothing_fits():
    assert pick_model([(Path("big.gguf"), 3000 * MB)], available_bytes=1000 * MB) is None
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement.** Create `src/faraday/preflight.py`:

```python
"""Startup memory guard (systemd ExecStartPre on the gen unit).

Checks the configured GGUF + headroom against MemAvailable. Fit -> write the model
path to /run/faraday/model.env (read by the unit's ExecStart via EnvironmentFile=)
and exit 0. No fit but a smaller gen GGUF exists -> fall back to it (logged). Nothing
fits -> exit 1 loudly, so the unit fails visibly instead of OOM-ing the board.

Pure decision logic (fits/pick_model) is unit-tested; only the /proc/meminfo read
and the env-file write are Pi-specific.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# KV cache at -c 4096 + compute buffers + slack, sized from M4a peak-RSS-vs-file-size
# deltas. Conservative on purpose: refusing too early beats OOM-ing the 4 GB board.
HEADROOM_BYTES = 700 * 1024 * 1024

MODELS_DIR = Path.home() / "faraday" / "models"
ENV_FILE = Path("/run/faraday/model.env")


def fits(model_bytes: int, available_bytes: int,
         headroom_bytes: int = HEADROOM_BYTES) -> bool:
    return model_bytes + headroom_bytes <= available_bytes


def pick_model(candidates: list[tuple[Path, int]], available_bytes: int,
               headroom_bytes: int = HEADROOM_BYTES) -> Path | None:
    """Largest candidate that fits (candidates need not be sorted)."""
    fitting = [(p, s) for p, s in candidates if fits(s, available_bytes, headroom_bytes)]
    if not fitting:
        return None
    return max(fitting, key=lambda t: t[1])[0]


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def _gen_candidates() -> list[tuple[Path, int]]:
    # All gen-model GGUFs (exclude the bge embedding model), any quant present.
    return [(p, p.stat().st_size) for p in MODELS_DIR.glob("*.gguf")
            if "bge" not in p.name.lower()]


def main() -> int:
    configured = os.environ.get("FARADAY_GEN_MODEL", "")
    avail = mem_available_bytes()
    if configured and Path(configured).exists() \
            and fits(Path(configured).stat().st_size, avail):
        chosen = Path(configured)
    else:
        chosen = pick_model(_gen_candidates(), avail)
        if chosen is None:
            print(f"preflight: NO model fits (MemAvailable={avail // 2**20} MiB, "
                  f"headroom={HEADROOM_BYTES // 2**20} MiB) — refusing to start",
                  file=sys.stderr)
            return 1
        if configured:
            print(f"preflight: '{configured}' does not fit — falling back to {chosen}")
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(f"FARADAY_GEN_MODEL={chosen}\n")
    print(f"preflight: ok — {chosen.name} (MemAvailable={avail // 2**20} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run — expect PASS** (3 passed). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_preflight.py -q && ruff check src/faraday/preflight.py tests/test_preflight.py"
git add src/faraday/preflight.py tests/test_preflight.py
git commit -m "feat(m5): startup memory guard (fits/pick_model + env-file fallback)"
```

---

### Task 5: systemd units + installer

**Files:**
- Create: `deploy/systemd/faraday-llama-gen.service`, `deploy/systemd/faraday-llama-embed.service`, `deploy/systemd/faraday-app.service`, `deploy/systemd/install.sh`

No unit test (verified live in Step 4 — **only on a quiet board**, never during a run).

- [ ] **Step 1: Write the three units.**

`deploy/systemd/faraday-llama-gen.service`:

```ini
[Unit]
Description=Faraday generation llama-server (:8080)
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/faraday
RuntimeDirectory=faraday
Environment=FARADAY_GEN_MODEL=/home/pi/faraday/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
ExecStartPre=/home/pi/faraday/.venv/bin/python -m faraday.preflight
EnvironmentFile=-/run/faraday/model.env
ExecStart=/home/pi/llama.cpp/build/bin/llama-server -m ${FARADAY_GEN_MODEL} -c 4096 -t 4 --metrics --host 0.0.0.0 --port 8080
Restart=on-failure
RestartSec=5
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=full

[Install]
WantedBy=multi-user.target
```

`deploy/systemd/faraday-llama-embed.service`:

```ini
[Unit]
Description=Faraday embedding llama-server (:8081)
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/faraday
ExecStart=/home/pi/llama.cpp/build/bin/llama-server -m /home/pi/faraday/models/bge-small-en-v1.5-f16.gguf --embeddings --metrics -t 4 --host 0.0.0.0 --port 8081
Restart=on-failure
RestartSec=5
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=full

[Install]
WantedBy=multi-user.target
```

`deploy/systemd/faraday-app.service`:

```ini
[Unit]
Description=Faraday RAG web app (:8000)
After=faraday-llama-gen.service faraday-llama-embed.service
Wants=faraday-llama-gen.service faraday-llama-embed.service

[Service]
User=pi
WorkingDirectory=/home/pi/faraday
ExecStart=/home/pi/faraday/.venv/bin/faraday serve --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=full

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Write the installer.** `deploy/systemd/install.sh`:

```bash
#!/usr/bin/env bash
# Run ON the Pi. Installs + enables the three Faraday units (idempotent).
set -euo pipefail
cd "$(dirname "$0")"
sudo cp faraday-llama-gen.service faraday-llama-embed.service faraday-app.service \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now faraday-llama-gen faraday-llama-embed faraday-app
systemctl --no-pager --type=service --state=running | grep faraday || true
echo "Installed. After 'git push pi', restart with: sudo systemctl restart faraday-app"
```

- [ ] **Step 3: Commit (units inert until installed).**

```bash
git update-index --add --chmod=+x deploy/systemd/install.sh
git add deploy/systemd/
git commit -m "feat(m5): systemd units (boot/crash survival + sandboxing) + installer"
```

- [ ] **Step 4: Live verification — QUIET BOARD ONLY.** Run on the Pi:

```bash
ssh pi@raspberrypi.local "cd ~/faraday && bash deploy/systemd/install.sh"
# crash survival: kill the app, watch systemd bring it back
ssh pi@raspberrypi.local "pkill -f '[f]araday serve'; sleep 8; curl -sf http://localhost:8000/healthz && echo RESTART-OK"
# boot survival: reboot, then everything returns with no SSH intervention
ssh pi@raspberrypi.local "sudo reboot"   # wait ~90s
ssh pi@raspberrypi.local "curl -sf http://localhost:8000/healthz && curl -sf http://localhost:8080/health && curl -sf http://localhost:8081/health && echo BOOT-OK"
```
Expected: `RESTART-OK`, then `BOOT-OK`. Record both in the M5 as-built (this is the demonstrated fix for the M3 stale-process class).

---

### Task 6: `scripts/bootstrap.sh` — one-shot install

**Files:** Create `scripts/bootstrap.sh`

- [ ] **Step 1: Write the script** (composes the existing 00/10/20 scripts; idempotent):

```bash
#!/usr/bin/env bash
# Run ON a fresh Raspberry Pi OS (64-bit) after cloning this repo to ~/faraday.
# One shot: deps -> build llama.cpp (-j3, 4GB-safe) -> models -> venv -> systemd units
# -> smoke test. Idempotent: every stage skips itself if already done.
set -euo pipefail
cd "$HOME/faraday"

echo "[1/5] System deps + Pi setup"
bash scripts/00_pi_setup.sh

echo "[2/5] llama.cpp (skip if built)"
if [[ ! -x "$HOME/llama.cpp/build/bin/llama-server" ]]; then
  bash scripts/10_build_llama.sh
fi

echo "[3/5] Models (skip if present)"
if ! ls "$HOME"/faraday/models/*q4_k_m.gguf >/dev/null 2>&1; then
  bash scripts/20_download_models.sh
fi

echo "[4/5] Python venv + package"
if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -e ".[dev]"

echo "[5/5] systemd units + smoke"
bash deploy/systemd/install.sh
sleep 20   # model load
bash scripts/40_smoke_test.sh
echo "Bootstrap complete — Faraday is live on :8000 and survives reboots."
```

- [ ] **Step 2: Commit + verify the happy path on the Pi** (idempotent re-run on the existing install is the practical test — a true fresh-OS run is documented as performed-if-possible in the as-built):

```bash
git update-index --add --chmod=+x scripts/bootstrap.sh
git add scripts/bootstrap.sh
git commit -m "feat(m5): one-shot bootstrap (fresh Pi -> running appliance)"
ssh pi@raspberrypi.local "cd ~/faraday && bash scripts/bootstrap.sh"   # quiet board only
```
Expected: all five stages skip-or-pass; smoke test green.

---

### Task 7: app Dockerfile + compose (+ package-data fix)

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- Modify: `pyproject.toml` (package-data — without it a non-editable install drops `static/index.html`)

- [ ] **Step 1: pyproject package-data.** Add after the `[tool.setuptools.packages.find]` block:

```toml
[tool.setuptools.package-data]
faraday = ["static/*"]
```

- [ ] **Step 2: Write the Docker files.**

`Dockerfile`:

```dockerfile
# Faraday app container (portability artifact). The llama-servers stay NATIVE on the
# Pi (NEON-tuned build, no container RAM tax on a 4 GB board) — this image runs the
# FastAPI app anywhere, pointed at those servers. See docs/report.md for the trade-off.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV FARADAY_GEN_URL=http://host.docker.internal:8080 \
    FARADAY_EMBED_URL=http://host.docker.internal:8081 \
    FARADAY_DB=/app/data/faraday.sqlite
EXPOSE 8000
CMD ["faraday", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`:

```yaml
# Run the Faraday app off-Pi against the Pi's llama-servers:
#   FARADAY_PI_HOST=192.168.100.59 docker compose up --build
services:
  faraday-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      FARADAY_GEN_URL: "http://${FARADAY_PI_HOST:-192.168.100.59}:8080"
      FARADAY_EMBED_URL: "http://${FARADAY_PI_HOST:-192.168.100.59}:8081"
      FARADAY_DB: /app/data/faraday.sqlite
    volumes:
      - ./data:/app/data
```

`.dockerignore`:

```
.venv
results
models
data
docs
monitoring
tests
.git
```

- [ ] **Step 3: Build + verify on the dev box** (no Pi dependency for the health check):

```powershell
docker build -t faraday-app .
docker run --rm -d -p 8000:8000 --name faraday-app-test faraday-app
curl http://localhost:8000/healthz   # expect {"status":"ok"}
curl http://localhost:8000/          # expect the chat HTML (proves package-data fix)
docker rm -f faraday-app-test
```

- [ ] **Step 4: Commit.**

```bash
git add Dockerfile docker-compose.yml .dockerignore pyproject.toml
git commit -m "feat(m5): app-only Docker image + compose (native servers documented)"
```

---

### Task 8: retrieval regression gate (integration)

**Files:** Test: `tests/test_retrieval_gate.py`

- [ ] **Step 1: Write the gate.** Create `tests/test_retrieval_gate.py`:

```python
"""Eval-as-test (design spec §188): retrieval-only recall gate over the golden set.
Integration-marked: needs the embed server + examples/eval_corpus + golden.jsonl.
Run: pytest -m integration tests/test_retrieval_gate.py -q   (on the Pi, servers up)

THRESHOLD rule: M4b baseline recall@4 minus 0.10 margin, rounded down to 0.05.
Committed at 0.50 (sanity floor); TIGHTENED to the rule's value in the M4b closeout
once the measured scorecard exists.
"""
import pytest

from faraday.config import Settings
from faraday.embedder import HttpEmbedder
from faraday.eval import config as eval_config
from faraday.eval.dataset import load_golden
from faraday.eval.metrics import chunk_is_relevant
from faraday.index_store import SqliteVecStore
from faraday.ingest import ingest
from faraday.retriever import Retriever

THRESHOLD = 0.50  # sanity floor; tightened from the M4b scorecard (see module docstring)


@pytest.mark.integration
def test_recall_at_4_meets_threshold(tmp_path):
    s = Settings.from_env()
    store = SqliteVecStore(str(tmp_path / "gate.sqlite"), dim=s.embed_dim)
    embedder = HttpEmbedder(s)
    ingest(eval_config.CORPUS_DIR, store, embedder, chunk_size=1200, chunk_overlap=200)
    retriever = Retriever(embedder, store)

    items = [i for i in load_golden(eval_config.GOLDEN_PATH) if i.answerable]
    hits = 0
    for item in items:
        retrieved = retriever.search(item.question, k=4)
        if any(chunk_is_relevant(rc.chunk.source, rc.chunk.ord, item, 1200, 200)
               for rc in retrieved):
            hits += 1
    store.close()
    recall = hits / len(items)
    assert recall >= THRESHOLD, f"recall@4={recall:.2f} below gate {THRESHOLD}"
```

- [ ] **Step 2: Commit (inert until run with `-m integration`; requires the merged M4b corpus + golden set):**

```bash
git add tests/test_retrieval_gate.py
git commit -m "test(m5): retrieval-recall regression gate (eval-as-test)"
```

- [ ] **Step 3: First live run** (after M4b merge + corpus + golden set; quiet board):

`ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest -m integration tests/test_retrieval_gate.py -q"` → expect 1 passed (~2–3 min, embed-only).

---

### Task 9: GBNF live smoke + full suite

**Files:** Test: `tests/test_grammar_integration.py`

- [ ] **Step 1: Write the live smoke** (proves the grammar param actually constrains on the OAI endpoint — the plan's contingency check). Create `tests/test_grammar_integration.py`:

```python
"""Live GBNF smoke: one grammar-ON request must yield text whose '[' usage is only
valid citations. If this FAILS while plain requests succeed, the OAI endpoint is
ignoring `grammar` — switch HttpLLMClient's grammar path to the native /completion
endpoint (documented contingency in the M5 plan)."""
import re

import pytest

from faraday.config import Settings
from faraday.grammar import build_citation_grammar
from faraday.llm_client import HttpLLMClient


@pytest.mark.integration
def test_grammar_constrains_live_output():
    llm = HttpLLMClient(Settings.from_env())
    g = build_citation_grammar(2)
    text = llm.complete(
        [{"role": "user", "content": "Say something brief and cite source one as [1]."}],
        max_tokens=64, grammar=g)
    assert text  # got output at all
    for m in re.finditer(r"\[([^\]]*)\]", text):
        assert m.group(1) in ("1", "2"), f"grammar leaked invalid citation: {m.group(0)!r}"
```

- [ ] **Step 2: Commit; run live when servers are up (quiet board).** Then full suite + lint:

```bash
git add tests/test_grammar_integration.py
git commit -m "test(m5): live GBNF constraint smoke (OAI-endpoint contingency check)"
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest -q && ruff check src tests"
```
Expected: all unit tests green, ruff clean. (The integration smoke runs with `-m integration` when servers are up.)

---

### Task 10: GBNF before/after measurement — **gated on M4b merge + baseline run**

**Files:**
- Modify: `src/faraday/eval/runner.py` (env overrides; grammar wiring where `run()` builds each `RagEngine`)
- Create: `src/faraday/eval/gbnf_compare.py`, `scripts/95_gbnf_measure.sh`
- Test: `tests/test_gbnf_compare.py`

- [ ] **Step 1: Failing test for the pure comparison.** Create `tests/test_gbnf_compare.py`:

```python
from faraday.eval.dataset import EvalItem
from faraday.eval.gbnf_compare import compare


def _row(qid, cited, invalid):
    return {"qid": qid, "retrieved": [{"source": "s.txt", "ord": 0}], "answer": "a [1].",
            "cited": cited, "invalid": invalid, "abstained": False}


def test_compare_reports_citation_validity_both_ways():
    items = {"q1": EvalItem("q1", "?", True, "s.txt", (0, 10), "r")}
    before = [_row("q1", [1], [3])]   # 50% valid
    after = [_row("q1", [1], [])]     # 100% valid
    out = compare(before, after, items, size=1200, overlap=200)
    assert out["before"]["citation_validity"] == 0.5
    assert out["after"]["citation_validity"] == 1.0
```

- [ ] **Step 2: Run — expect FAIL.** Then implement. `src/faraday/eval/gbnf_compare.py`:

```python
"""Compare citation validity prompting-only vs grammar-constrained (M5 §4.5).
Reads two raw dirs (the M4b baseline run and the grammar re-run), aggregates the
deterministic metrics for the baseline config, writes a small markdown summary."""
from __future__ import annotations

import json
from pathlib import Path

from faraday.eval import config
from faraday.eval.dataset import EvalItem, load_golden
from faraday.eval.metrics import aggregate

GRAMMAR_RAW_DIR = config.EVAL_DIR / "raw_grammar"


def _rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def compare(before_rows: list[dict], after_rows: list[dict],
            items_by_id: dict[str, EvalItem], size: int, overlap: int) -> dict:
    return {"before": aggregate(before_rows, items_by_id, size, overlap),
            "after": aggregate(after_rows, items_by_id, size, overlap)}


def main() -> None:
    slug = config.BASELINE.slug
    items = {i.id: i for i in load_golden(config.GOLDEN_PATH)}
    out = compare(_rows(config.RAW_DIR / f"{slug}.jsonl"),
                  _rows(GRAMMAR_RAW_DIR / f"{slug}.jsonl"),
                  items, config.BASELINE.chunk_size, config.BASELINE.chunk_overlap)
    b, a = out["before"], out["after"]
    md = (
        "# GBNF citations — before/after (baseline config)\n\n"
        "| | prompting only | grammar-constrained |\n|---|---|---|\n"
        f"| citation validity | {b['citation_validity']:.3f} | {a['citation_validity']:.3f} |\n"
        f"| recall@k (sanity) | {b['recall_at_k']:.3f} | {a['recall_at_k']:.3f} |\n"
        f"| abstention acc (sanity) | {b['abstention_accuracy']:.3f} | {a['abstention_accuracy']:.3f} |\n"
    )
    (config.EVAL_DIR / "gbnf_before_after.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
```

`src/faraday/eval/runner.py` — two env overrides in `run()` (+ `import os` at top), and grammar
wiring where `run()` builds each `RagEngine` (since the M4b audit fixes, ingest happens once per
chunk-size via `build_retriever` and `run()` constructs one engine per top_k):

```python
# in run(), where the engine is built:
    from faraday.grammar import build_citation_grammar
    gb = build_citation_grammar if settings.use_grammar else None
    engine = RagEngine(retriever, llm, top_k=top_k, grammar_builder=gb)

# in run(), at the top:
    only = {s for s in os.environ.get("FARADAY_EVAL_CONFIGS", "").split(",") if s}
    raw_base = Path(os.environ.get("FARADAY_EVAL_RAW_DIR", str(config.RAW_DIR)))
# and inside the inner loop, replace `_raw_path(cfg)` / add the filter:
            if only and cfg.slug not in only:
                continue
            made = run_config(cfg, engine, items, raw_base / f"{cfg.slug}.jsonl")
```
(Also skip a chunk-size's outer iteration entirely when none of its configs are selected, so
filtered sizes don't ingest — the ingest now lives in the outer per-size loop.)

`scripts/95_gbnf_measure.sh`:

```bash
#!/usr/bin/env bash
# Run ON the Pi AFTER the M4b baseline run. Re-runs ONLY the baseline config with
# grammar-constrained decoding into a separate raw dir, then writes the before/after.
set -euo pipefail
cd "$HOME/faraday"
# shellcheck disable=SC1091
source .venv/bin/activate
export FARADAY_USE_GRAMMAR=1
export FARADAY_EVAL_CONFIGS=k4_c1200_o200
export FARADAY_EVAL_RAW_DIR="$HOME/faraday/results/evals/raw_grammar"
python -m faraday.eval.runner
python -m faraday.eval.gbnf_compare
echo "Done — commit results/evals/gbnf_before_after.md"
```

- [ ] **Step 3: Run unit test green + ruff; commit; execute the script after the M4b baseline run** (~20–30 min, quiet board). Add `results/evals/raw_grammar/` to `.gitignore` (same block as `raw/`).

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_gbnf_compare.py -q && ruff check src tests"
git update-index --add --chmod=+x scripts/95_gbnf_measure.sh
git add src/faraday/eval/gbnf_compare.py src/faraday/eval/runner.py scripts/95_gbnf_measure.sh tests/test_gbnf_compare.py .gitignore
git commit -m "feat(m5): GBNF before/after measurement (runner overrides + compare)"
```

---

## Phase 2 — The narrative (after all M4 results are committed)

### Task 11: technical report `docs/report.md`

Procedure task — written from committed artifacts. **Skeleton (the report must follow it):**

```markdown
# Engineering a private RAG appliance on a 4 GB Raspberry Pi
1. The constraint (why 4 GB is the interesting case; the privacy story needs the edge)
2. Architecture (RAG pipeline, Protocol-DI, the event seam; diagram)
3. Choosing the model — the quality/footprint frontier (M4a: frontier.png, the knee,
   dominated cells, the 3B "speed wall not memory wall" finding)
4. Is it actually good? — RAG quality (M4b: scorecard, ablations.png, abstention;
   GBNF before/after table)
5. Making it fast (M4c: lever_gains.png, waterfall.png, speculative verdict,
   Ollama-vs-tuned, context_curve.png)
6. Shipping it (systemd survival demo, bootstrap, memory guard, the Docker judgment)
7. What to run on a Pi 4 (the one-paragraph answer + leaderboard link)
8. Lessons learned (PSU/under-voltage saga; mmap vs RSS; the stale-process bug;
   pkill self-match; measurement hygiene as a discipline)
9. Next steps (reranker, energy/watts, Pi 5/NPU, fine-tuning, hardened deployment)
```

**Acceptance:** every numeric claim links a committed artifact (CSV/PNG/md); lessons section names real incidents; ≤ ~2500 words excluding figures. Commit: `docs(m5): technical report`.

### Task 12: demo GIF (airplane-mode flex)

Procedure (dev box + Pi, after systemd is live):

1. Block the Pi's internet but keep LAN/SSH: `ssh pi@raspberrypi.local "sudo iptables -A OUTPUT ! -d 192.168.100.0/24 -j REJECT"`.
2. Prove it on camera: `curl --max-time 5 https://example.com` fails on the Pi; the browser at `http://192.168.100.59:8000` still answers a document question with sources.
3. Record the browser interaction (Windows: ScreenToGif; or ffmpeg screen capture → `ffmpeg -i in.mp4 -vf "fps=10,scale=800:-1" docs/assets/demo.gif`), ≤ 15 s, ≤ 5 MB.
4. Revert: `ssh pi@raspberrypi.local "sudo iptables -D OUTPUT ! -d 192.168.100.0/24 -j REJECT"`.
5. Save `docs/assets/demo.gif`; commit `docs(m5): airplane-mode demo gif`.

### Task 13: README final pass

Checklist (every item required):
- [ ] Hero: demo GIF embedded directly under the title; status line → 🟢 all milestones complete.
- [ ] Results table: frontier knee (1.5B Q4_K_M + its clean ppl/RSS), best-tuned decode tok/s (+% over baseline), recall@4 / faithfulness / abstention at baseline, GBNF before→after validity.
- [ ] Quickstart = `git clone` → `bash scripts/bootstrap.sh` (+ Docker option for the app).
- [ ] Roadmap table: M0–M5 all ✅. Design-docs links: + M5 spec/plan, report, leaderboard.
- [ ] Commit: `docs(m5): final README`.

### Task 14: `docs/pi4-leaderboard.md`

Structure: intro ("what runs well on a Pi 4, measured"); method note (llama-bench rules, throttle hygiene, PSU lesson, exact hardware); **Table 1** = the M4a 18-cell quant leaderboard (from `results/sweep/leaderboard.md`, clean re-run numbers); **Table 2** = the M4c optimization leaderboard (best-tuned/speculative/Ollama); caveats (single board, wikitext ppl, CC-BY-SA corpus); how to reproduce (the three runner scripts). Commit: `docs(m5): Pi-4 leaderboard`.

### Task 15: M5 as-built + final state sync

- Write `docs/superpowers/plans/2026-06-10-faraday-m5-as-built.md` (M3/M4a as-built format: delivered, verified — RESTART-OK/BOOT-OK transcripts, bootstrap run, Docker build, GBNF delta — findings, deliberate calls).
- CLAUDE.md: State → "M0–M5 complete — shipped"; Run list += `bootstrap.sh`, systemd note replaces the "not daemons" gotcha (rewrite that gotcha: *now* systemd-managed; `sudo systemctl restart faraday-app` after deploy).
- Memory: `faraday-project.md` → shipped; retire stale trigger memories.
- Commit: `docs(m5): as-built + final state sync`. Tag: `git tag -a v1.0 -m "Faraday v1.0 - shipped" && git push origin v1.0`.

---

## Self-Review

**1. Spec coverage:**

| Spec § | Task |
|---|---|
| 4.1 systemd units + sandboxing + restart-on-deploy doc | 5 |
| 4.2 memory guard (pure core, env-file fallback, loud refusal) | 4 (+ gen unit wiring in 5) |
| 4.3 bootstrap | 6 |
| 4.4 app container + judgment doc | 7 (+ report §6 in 11) |
| 4.5 GBNF: builder, flag, pass-through, before/after | 1, 2, 3, 9 (live proof), 10 (measure) |
| 4.6 eval-as-test gate | 8 |
| 5.1–5.5 report / GIF / README / leaderboard / sync | 11–15 |
| §6 sequencing rules | header gates + Task 5/6/8/10 "quiet board" notes |
| §7 grammar-param risk + contingency | verified via server README; Task 9 smoke is the proof; contingency named in the test docstring |
| §8 testing matrix | unit (1,2,3,4,10) / integration (8,9) / behavioral (5,6,7) |
| §180 `/healthz` | already exists (`server.py:60`) — no task needed, noted |

**2. Placeholder scan:** none. The two execution-time values are explicit *rules*, not TBDs: the gate THRESHOLD (committed sanity floor 0.50 + the measured−0.10 tightening rule at M4b closeout) and report numbers (pulled from committed artifacts at writing time). Phase-2 tasks are procedures with concrete skeletons/checklists, per the M4b-Plan-2 precedent.

**3. Type consistency:** `build_citation_grammar(n_sources:int)->str` consistent (1,2,3,9,10). `LLMClient.complete/stream(..., grammar: str|None=None)` consistent across Protocol, HttpLLMClient, FakeLLM, RagEngine call sites. `RagEngine(..., grammar_builder: Callable[[int],str]|None)` consistent (2,3,10). `fits(model_bytes, available_bytes, headroom_bytes)` / `pick_model(candidates, available_bytes, headroom_bytes)` match tests. `compare(before_rows, after_rows, items_by_id, size, overlap)` matches its test; reuses verified `aggregate(rows, items_by_id, size, overlap)` and `chunk_is_relevant(source, ord, item, size, overlap)` from the merged engine. Unit/env names consistent: `FARADAY_GEN_MODEL`, `/run/faraday/model.env`, `FARADAY_USE_GRAMMAR`, `FARADAY_EVAL_CONFIGS`, `FARADAY_EVAL_RAW_DIR`.

**Verdict:** complete, spec-covering, placeholder-free. With this plan, every remaining unit of project work is execute-only.
