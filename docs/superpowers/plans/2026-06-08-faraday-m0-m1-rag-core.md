# Faraday — Plan 1: Bring-up + RAG Core (M0–M1) Implementation Plan

> **✅ STATUS: COMPLETE — merged to `main`.** Both milestones shipped; see the as-built records: [M0](./2026-06-08-faraday-m0-as-built.md) · [M1](./2026-06-08-faraday-m1-as-built.md). 22 tests green, appliance proven offline on hardware.
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a Raspberry Pi 4 (4GB) running local llama.cpp inference, then build a fully-offline CLI that answers natural-language questions about a user's documents with verifiable citations.

**Architecture:** Two local `llama-server` instances (generation + embeddings) expose OpenAI-compatible APIs on the Pi. A small Python package ingests documents → chunks → embeddings stored in a single-file `sqlite-vec` DB; at query time it retrieves top-k chunks, grounds a prompt, calls the local LLM, and verifies the returned citations point to real retrieved chunks. No network egress at serve time.

**Tech Stack:** Raspberry Pi OS Lite 64-bit · llama.cpp (`llama-server`, `llama-bench`) · Qwen2.5-1.5B-Instruct (Q4_K_M GGUF) · bge-small-en-v1.5 (GGUF, 384-dim) · Python 3.13 · httpx · sqlite-vec · pypdf · typer · pytest · ruff

---

## Scope of this plan (and what's deferred)

- **In scope:** M0 (Pi bring-up, llama.cpp build, models, baseline benchmark) + M1 (the ingest → retrieve → ground → answer → verify-citations RAG core, exposed as a CLI).
- **Deferred to Plan 2 (M2):** FastAPI serving layer, SSE streaming, OpenAI-compatible passthrough, web UI, GBNF-grammar structured citations, systemd/docker packaging, and incremental/idempotent re-ingest (content-hash skip — M1 ingests into a fresh DB; delete `data/faraday.sqlite` to re-index).
- **Deferred to later plans (M3–M5):** Prometheus/Grafana observability, the benchmark/eval lab, the written report. These are re-planned after M0–M1 produce real numbers.
- **Definition of done for Plan 1:** On the Pi, `faraday ingest ./corpus` then `faraday ask "..."` returns a grounded answer with a citation report, fully offline. All unit tests green on the dev machine; the integration test green on the Pi.

## Development Environment & Prerequisites

**As-built setup (M0 verified on real hardware — see the [M0 as-built doc](./2026-06-08-faraday-m0-as-built.md)):**
- A Raspberry Pi 4 **(4GB)** with **Raspberry Pi OS Lite (64-bit, Debian 13 "trixie", Python 3.13)** flashed, on your network, **SSH key auth + passwordless sudo** enabled, reachable as `pi@raspberrypi.local`.
- This repo (`C:\projects\piai`, git-initialized) is the **source of truth**, edited on the Windows dev machine.
- **Dev loop (develop-against-the-Pi):** author code on Windows → `git push pi` deploys it to the Pi's working tree → run `pytest`/servers on the Pi. Chosen because Windows + Python 3.13 can't cleanly load the `sqlite-vec` native extension, and the Pi is the real target anyway.
- **Test split:**
  - **Pure-Python units** (fakes, no llama-server) — fast; run in the Pi venv.
  - **`[on Pi]`** — `sqlite-vec`-backed tests, the integration test (needs `llama-server`), and all hardware/benchmark steps.
- Code reaches the Pi via **`scripts/sync.ps1`** (`git push pi`, a repo with `receive.denyCurrentBranch=updateInstead`). No GitHub remote required.

**Conventions in this plan:** Each step is tagged `[on dev machine]` or `[on Pi]`. Commits always happen in the repo on the dev machine. Shell commands are bash (use Git Bash / WSL on Windows, or the Pi's shell).

## File Structure

```
scripts/
  00_pi_setup.sh        # apt toolchain install (run on Pi)
  10_build_llama.sh     # clone + cmake-build llama.cpp (NEON, -j3)
  20_download_models.sh # fetch GGUF gen + embed models (venv + hf CLI)
  30_run_servers.sh     # launch the two llama-server instances (nohup)
  40_smoke_test.sh      # health-check + gen/embed API smoke test
  50_mem_report.sh      # honest RSS/PSS memory diagnostic (vs free)
  sync.ps1              # git-push repo → Pi (Windows/PowerShell)
pyproject.toml          # package + deps + pytest/ruff config
src/faraday/
  __init__.py
  config.py             # Settings: server URLs, dims, chunk params, top-k
  models.py             # dataclasses: Document, Chunk, RetrievedChunk, Answer
  parsers.py            # load_document(path) -> Document
  chunker.py            # chunk_document(doc, ...) -> list[Chunk]
  embedder.py           # Embedder protocol + HttpEmbedder
  index_store.py        # SqliteVecStore: add_chunks, search
  retriever.py          # Retriever.search(query) -> list[RetrievedChunk]
  prompt.py             # build_messages(query, chunks) -> messages
  citations.py          # extract + verify citations
  llm_client.py         # LLMClient protocol + HttpLLMClient
  ingest.py             # ingest(source_dir, ...) -> IngestStats
  rag.py                # RagEngine.answer(query) -> Answer
  cli.py                # typer app: ingest, ask
tests/
  conftest.py           # fakes + fixtures
  test_chunker.py  test_parsers.py  test_index_store.py
  test_retriever.py  test_prompt.py  test_citations.py
  test_ingest.py  test_rag.py
  test_integration_pi.py   # @pytest.mark.integration (run on Pi)
results/baseline/         # committed baseline benchmark numbers
examples/corpus/          # shipped demo docs (committed)
corpus/                   # your own private docs (gitignored)
```

---

# Milestone M0 — Pi Bring-up

> **✅ COMPLETED 2026-06-08.** Executed on real hardware. The authoritative record of *what was actually built* (commands, deviations, findings, validated numbers) is in **[2026-06-08-faraday-m0-as-built.md](./2026-06-08-faraday-m0-as-built.md)**. The task steps below are the original plan, preserved for provenance; where reality diverged (git-push sync, `hf` CLI, venv, NOPASSWD sudo, `-j3` build, added `40_`/`50_` scripts), the as-built doc is the source of truth.

> M0 produces committed, re-runnable scripts (`00`→`50`) plus a recorded, throttle-validated baseline. The "test" for each ops task is a verification command with expected output.

### Task 0: Establish Pi access and system prep

**Files:**
- Create: `scripts/00_pi_setup.sh`
- Create: `scripts/sync.sh`

- [ ] **Step 1: Write `scripts/sync.sh`** `[on dev machine]`

```bash
#!/usr/bin/env bash
# Sync the repo to the Pi (excludes git + local model/data dirs).
set -euo pipefail
PI="${PI_HOST:-pi@raspberrypi.local}"
DEST="${PI_DEST:-~/faraday}"
rsync -av --delete \
  --exclude '.git' --exclude 'models' --exclude 'data' \
  --exclude '*.sqlite*' --exclude '.venv' --exclude '__pycache__' \
  ./ "$PI:$DEST/"
echo "Synced to $PI:$DEST"
```

- [ ] **Step 2: Write `scripts/00_pi_setup.sh`** `[on Pi]`

```bash
#!/usr/bin/env bash
# Run ON the Pi. Installs build/runtime deps and tunes memory for a 4GB board.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 python3-venv python3-pip \
  libcurl4-openssl-dev pkg-config rsync
# zram (compressed RAM swap) helps survive memory spikes on 4GB.
sudo apt-get install -y zram-tools
echo -e "ALGO=zstd\nPERCENT=50" | sudo tee /etc/default/zramswap
sudo systemctl restart zramswap || true
# Report the environment so we know what we're working with.
echo "cores: $(nproc)"; free -h; vcgencmd measure_temp
```

- [ ] **Step 3: Verify access + run prep** `[on dev machine]` then `[on Pi]`

Run: `bash scripts/sync.sh` then `ssh pi@raspberrypi.local 'cd ~/faraday && bash scripts/00_pi_setup.sh'`
Expected: ends with `cores: 4`, a `free -h` table showing ~3.7Gi total + a `Swap` row, and `temp=NN.N'C`.

- [ ] **Step 4: Commit** `[on dev machine]`

```bash
git add scripts/00_pi_setup.sh scripts/sync.sh
git commit -m "feat(m0): pi system-prep and repo sync scripts"
```

### Task 1: Build llama.cpp on the Pi

**Files:**
- Create: `scripts/10_build_llama.sh`

- [ ] **Step 1: Write `scripts/10_build_llama.sh`** `[on Pi]`

```bash
#!/usr/bin/env bash
# Run ON the Pi. Builds llama.cpp (server + bench) with native NEON optimization.
set -euo pipefail
cd "$HOME"
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && git pull --ff-only
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=ON
cmake --build build --config Release -j"$(nproc)" --target llama-server llama-bench llama-cli
ls -lh build/bin/llama-server build/bin/llama-bench
```

- [ ] **Step 2: Run the build** `[on Pi]`

Run: `ssh pi@raspberrypi.local 'cd ~/faraday && bash scripts/10_build_llama.sh'`
Expected: build completes (several minutes); final `ls` lists `llama-server` and `llama-bench` binaries.

- [ ] **Step 3: Smoke-check the binary** `[on Pi]`

Run: `ssh pi@raspberrypi.local '~/llama.cpp/build/bin/llama-server --version'`
Expected: prints a version/build line with no error.

- [ ] **Step 4: Commit** `[on dev machine]`

```bash
git add scripts/10_build_llama.sh
git commit -m "feat(m0): llama.cpp build script (NEON, server+bench)"
```

### Task 2: Download the models

**Files:**
- Create: `scripts/20_download_models.sh`

- [ ] **Step 1: Write `scripts/20_download_models.sh`** `[on Pi]`

```bash
#!/usr/bin/env bash
# Run ON the Pi. Downloads the generation + embedding GGUF models.
set -euo pipefail
cd "$HOME/faraday"
python3 -m pip install --user --quiet "huggingface-hub[cli]"
mkdir -p models
# Generation model (~1.1GB). Verify the exact filename if the download 404s:
#   huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF --include "*.gguf" --local-dir /tmp/list
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models
# Embedding model (384-dim, ~130MB):
huggingface-cli download CompendiumLabs/bge-small-en-v1.5-gguf \
  bge-small-en-v1.5-f16.gguf --local-dir models
ls -lh models
```

- [ ] **Step 2: Run the download** `[on Pi]`

Run: `ssh pi@raspberrypi.local 'cd ~/faraday && bash scripts/20_download_models.sh'`
Expected: `models/` lists `qwen2.5-1.5b-instruct-q4_k_m.gguf` (~1.0–1.1G) and `bge-small-en-v1.5-f16.gguf` (~130M).

- [ ] **Step 3: Commit** `[on dev machine]`

```bash
git add scripts/20_download_models.sh
git commit -m "feat(m0): model download script (Qwen2.5-1.5B + bge-small)"
```

### Task 3: Run the two servers and smoke-test the APIs

**Files:**
- Create: `scripts/30_run_servers.sh`

- [ ] **Step 1: Write `scripts/30_run_servers.sh`** `[on Pi]`

```bash
#!/usr/bin/env bash
# Run ON the Pi. Launches generation (:8080) and embedding (:8081) servers.
set -euo pipefail
BIN="$HOME/llama.cpp/build/bin/llama-server"
M="$HOME/faraday/models"
THREADS="$(nproc)"
"$BIN" -m "$M/qwen2.5-1.5b-instruct-q4_k_m.gguf" -c 4096 -t "$THREADS" \
  --host 0.0.0.0 --port 8080 >/tmp/gen.log 2>&1 &
echo "gen server pid $!"
"$BIN" -m "$M/bge-small-en-v1.5-f16.gguf" --embeddings -t "$THREADS" \
  --host 0.0.0.0 --port 8081 >/tmp/embed.log 2>&1 &
echo "embed server pid $!"
echo "logs: /tmp/gen.log /tmp/embed.log"
```

- [ ] **Step 2: Start servers and wait for readiness** `[on Pi]`

Run: `ssh pi@raspberrypi.local 'cd ~/faraday && bash scripts/30_run_servers.sh && sleep 20 && curl -s localhost:8080/health && echo && curl -s localhost:8081/health'`
Expected: two `{"status":"ok"}` responses (model load can take 10–20s).

- [ ] **Step 3: Smoke-test generation** `[on Pi]`

Run:
```bash
ssh pi@raspberrypi.local 'curl -s localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ready\"}],\"max_tokens\":8}"'
```
Expected: JSON with `choices[0].message.content` containing `ready`.

- [ ] **Step 4: Smoke-test embeddings (confirm 384 dims)** `[on Pi]`

Run:
```bash
ssh pi@raspberrypi.local 'curl -s localhost:8081/v1/embeddings -H "Content-Type: application/json" -d "{\"input\":\"hello world\"}" | python3 -c "import sys,json;d=json.load(sys.stdin);print(len(d[\"data\"][0][\"embedding\"]))"'
```
Expected: prints `384`.

- [ ] **Step 5: Commit** `[on dev machine]`

```bash
git add scripts/30_run_servers.sh
git commit -m "feat(m0): server launch script + API smoke tests"
```

### Task 4: Record the baseline benchmark

**Files:**
- Create: `results/baseline/README.md`

- [ ] **Step 1: Run `llama-bench`** `[on Pi]`

Run:
```bash
ssh pi@raspberrypi.local '~/llama.cpp/build/bin/llama-bench -m ~/faraday/models/qwen2.5-1.5b-instruct-q4_k_m.gguf -t $(nproc) -p 128 -n 64'
```
Expected: a table with `pp` (prefill) and `tg` (decode) tok/s rows. **Note the numbers.**

- [ ] **Step 2: Record results in `results/baseline/README.md`** `[on dev machine]`

```markdown
# Baseline — Qwen2.5-1.5B-Instruct Q4_K_M on Pi 4 (4GB)

| Metric | Value |
|---|---|
| Prefill tok/s (pp128) | <fill from llama-bench> |
| Decode tok/s (tg64) | <fill from llama-bench> |
| Threads | 4 |
| Context | 4096 |
| Date | 2026-06-08 |

Captured via `llama-bench`. This is the pre-optimization reference for the M4 study.
```
Replace `<fill ...>` with the measured numbers from Step 1.

- [ ] **Step 3: Commit** `[on dev machine]`

```bash
git add results/baseline/README.md
git commit -m "docs(m0): record baseline llama-bench numbers"
```

---

# Milestone M1 — RAG Core (CLI)

> TDD from here. Unit tests run `[on dev machine]` unless tagged otherwise. Use fakes for the embedder and LLM so units need no server.

### Task 5: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/faraday/__init__.py`, `src/faraday/config.py`, `tests/conftest.py`

- [ ] **Step 1: Write `pyproject.toml`** `[on dev machine]`

```toml
[project]
name = "faraday"
version = "0.1.0"
description = "Air-gapped personal RAG appliance for the Raspberry Pi"
requires-python = ">=3.11"
dependencies = [
  "httpx>=0.27",
  "sqlite-vec>=0.1.3",
  "pypdf>=4.0",
  "typer>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5"]

[project.scripts]
faraday = "faraday.cli:app"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
markers = ["integration: requires a running llama-server (run on the Pi)"]
addopts = "-m 'not integration'"

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Create the package + config** `[on dev machine]`

`src/faraday/__init__.py`:
```python
__all__ = ["__version__"]
__version__ = "0.1.0"
```

`src/faraday/config.py`:
```python
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    gen_url: str = "http://localhost:8080"
    embed_url: str = "http://localhost:8081"
    embed_dim: int = 384
    db_path: str = "data/faraday.sqlite"
    chunk_size: int = 1200       # characters
    chunk_overlap: int = 200     # characters
    top_k: int = 4
    max_tokens: int = 512

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            gen_url=os.environ.get("FARADAY_GEN_URL", cls.gen_url),
            embed_url=os.environ.get("FARADAY_EMBED_URL", cls.embed_url),
            db_path=os.environ.get("FARADAY_DB", cls.db_path),
        )
```

- [ ] **Step 3: Install editable + verify** `[on dev machine]`

Run: `python -m pip install -e ".[dev]"` then `python -c "import faraday; print(faraday.__version__)"`
Expected: prints `0.1.0`.

- [ ] **Step 4: Commit** `[on dev machine]`

```bash
git add pyproject.toml src/faraday/__init__.py src/faraday/config.py
git commit -m "chore(m1): project scaffold, deps, settings"
```

### Task 6: Domain models

**Files:**
- Create: `src/faraday/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_models.py`:
```python
from faraday.models import Chunk, RetrievedChunk


def test_retrieved_chunk_exposes_source():
    c = Chunk(doc_id="doc1", ord=0, text="hello", source="a.txt")
    rc = RetrievedChunk(chunk=c, score=0.9)
    assert rc.chunk.source == "a.txt"
    assert rc.score == 0.9
```

- [ ] **Step 2: Run it to confirm failure**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'faraday.models'`.

- [ ] **Step 3: Implement `src/faraday/models.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    source: str          # filename / path
    text: str


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    ord: int             # position within the document
    text: str
    source: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/faraday/models.py tests/test_models.py
git commit -m "feat(m1): domain dataclasses"
```

### Task 7: Document parsers

**Files:**
- Create: `src/faraday/parsers.py`
- Test: `tests/test_parsers.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_parsers.py`:
```python
from pathlib import Path
from faraday.parsers import load_document


def test_loads_text_file(tmp_path: Path):
    p = tmp_path / "note.txt"
    p.write_text("Hello Faraday", encoding="utf-8")
    doc = load_document(p)
    assert doc.source == "note.txt"
    assert doc.text == "Hello Faraday"


def test_loads_markdown_file(tmp_path: Path):
    p = tmp_path / "note.md"
    p.write_text("# Title\n\nBody", encoding="utf-8")
    doc = load_document(p)
    assert "Body" in doc.text


def test_rejects_unknown_extension(tmp_path: Path):
    p = tmp_path / "image.png"
    p.write_bytes(b"\x89PNG")
    try:
        load_document(p)
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_parsers.py -v`
Expected: FAIL — module/function not found.

- [ ] **Step 3: Implement `src/faraday/parsers.py`**

```python
from __future__ import annotations
from pathlib import Path
from faraday.models import Document

TEXT_EXTS = {".txt", ".md", ".markdown"}


def load_document(path: str | Path) -> Document:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif ext == ".pdf":
        text = _load_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return Document(source=path.name, text=text)


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(parts)
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_parsers.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/faraday/parsers.py tests/test_parsers.py
git commit -m "feat(m1): txt/md/pdf document parsers"
```

### Task 8: Chunker

**Files:**
- Create: `src/faraday/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_chunker.py`:
```python
from faraday.models import Document
from faraday.chunker import chunk_document


def test_short_doc_is_single_chunk():
    doc = Document(source="a.txt", text="one two three")
    chunks = chunk_document(doc, size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].ord == 0
    assert chunks[0].source == "a.txt"
    assert chunks[0].text == "one two three"


def test_long_doc_splits_with_overlap():
    text = "x" * 250
    doc = Document(source="a.txt", text=text)
    chunks = chunk_document(doc, size=100, overlap=20)
    assert len(chunks) == 3            # 0-100, 80-180, 160-250
    assert [c.ord for c in chunks] == [0, 1, 2]
    # overlap: chunk 1 starts 20 chars before chunk 0 ended
    assert chunks[1].text[:20] == chunks[0].text[-20:]


def test_doc_ids_are_stable_for_same_source():
    doc = Document(source="a.txt", text="hello")
    a = chunk_document(doc, size=100, overlap=10)
    b = chunk_document(doc, size=100, overlap=10)
    assert a[0].doc_id == b[0].doc_id
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL — module/function not found.

- [ ] **Step 3: Implement `src/faraday/chunker.py`**

```python
from __future__ import annotations
import hashlib
from faraday.models import Chunk, Document


def _doc_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def chunk_document(doc: Document, size: int = 1200, overlap: int = 200) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap >= size:
        raise ValueError("overlap must be < size")
    text = doc.text
    doc_id = _doc_id(doc.source)
    chunks: list[Chunk] = []
    start, ord_ = 0, 0
    step = size - overlap
    while start < len(text):
        piece = text[start : start + size]
        chunks.append(Chunk(doc_id=doc_id, ord=ord_, text=piece, source=doc.source))
        ord_ += 1
        start += step
    if not chunks:  # empty document
        chunks.append(Chunk(doc_id=doc_id, ord=0, text="", source=doc.source))
    return chunks
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_chunker.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/faraday/chunker.py tests/test_chunker.py
git commit -m "feat(m1): character chunker with overlap"
```

### Task 9: Embedder (protocol + HTTP client + fake)

**Files:**
- Create: `src/faraday/embedder.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_embedder.py`

- [ ] **Step 1: Write `tests/conftest.py` with a deterministic fake** `[on dev machine]`

```python
import pytest


class FakeEmbedder:
    """Deterministic vocab-based embeddings: each distinct word gets a stable slot,
    so the same instance embeds corpus and query consistently (no hash randomization)."""
    dim = 32

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def _slot(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = len(self._vocab) % self.dim
        return self._vocab[word]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                v[self._slot(w)] += 1.0
            out.append(v)
        return out


class FakeLLM:
    def __init__(self, reply: str = "Answer [1]."):
        self.reply = reply
        self.last_messages = None

    def complete(self, messages: list[dict], max_tokens: int = 512) -> str:
        self.last_messages = messages
        return self.reply


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def make_llm():
    """Factory for a FakeLLM with a custom canned reply."""
    return lambda reply: FakeLLM(reply)
```

- [ ] **Step 2: Write the failing test**

`tests/test_embedder.py`:
```python
from faraday.embedder import Embedder


def test_fake_embedder_satisfies_protocol(fake_embedder):
    assert isinstance(fake_embedder, Embedder)          # runtime_checkable
    vecs = fake_embedder.embed(["hello world", "hello"])
    assert len(vecs) == 2
    assert len(vecs[0]) == fake_embedder.dim
```

- [ ] **Step 3: Run to confirm failure**

Run: `pytest tests/test_embedder.py -v`
Expected: FAIL — cannot import `Embedder`.

- [ ] **Step 4: Implement `src/faraday/embedder.py`**

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable
import httpx
from faraday.config import Settings


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HttpEmbedder:
    """Calls a llama-server (started with --embeddings) OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings | None = None, timeout: float = 60.0):
        self.settings = settings or Settings()
        self._client = httpx.Client(base_url=self.settings.embed_url, timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post("/v1/embeddings", json={"input": texts})
        resp.raise_for_status()
        data = resp.json()["data"]
        # Preserve input order (OpenAI returns an "index" per item).
        ordered = sorted(data, key=lambda d: d["index"])
        return [d["embedding"] for d in ordered]

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 5: Run to confirm pass**

Run: `pytest tests/test_embedder.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/faraday/embedder.py tests/conftest.py tests/test_embedder.py
git commit -m "feat(m1): embedder protocol + http client + test fakes"
```

### Task 10: Vector store (sqlite-vec)

**Files:**
- Create: `src/faraday/index_store.py`
- Test: `tests/test_index_store.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_index_store.py`:
```python
from faraday.models import Chunk
from faraday.index_store import SqliteVecStore


def _chunk(i, text):
    return Chunk(doc_id="d1", ord=i, text=text, source="s.txt")


def test_add_and_search_returns_nearest(tmp_path):
    store = SqliteVecStore(str(tmp_path / "t.sqlite"), dim=3)
    store.add_chunks(
        [_chunk(0, "red"), _chunk(1, "blue")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    results = store.search([0.9, 0.1, 0.0], k=1)
    assert len(results) == 1
    assert results[0].chunk.text == "red"
    assert results[0].score >= 0.0
    store.close()


def test_search_respects_k(tmp_path):
    store = SqliteVecStore(str(tmp_path / "t.sqlite"), dim=3)
    store.add_chunks(
        [_chunk(0, "a"), _chunk(1, "b"), _chunk(2, "c")],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert len(store.search([1, 0, 0], k=2)) == 2
    store.close()
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_index_store.py -v`
Expected: FAIL — module/class not found.

- [ ] **Step 3: Implement `src/faraday/index_store.py`**

```python
from __future__ import annotations
import os
import sqlite3
import sqlite_vec
from faraday.models import Chunk, RetrievedChunk


class SqliteVecStore:
    """Single-file vector store. Chunk text/metadata in `chunks`; vectors in a vec0 table."""

    def __init__(self, path: str, dim: int):
        self.dim = dim
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, ord INTEGER, "
            "text TEXT, source TEXT)"
        )
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding float[{self.dim}] distance_metric=cosine)"
        )
        self.db.commit()

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        for chunk, vec in zip(chunks, vectors):
            cur = self.db.execute(
                "INSERT INTO chunks(doc_id, ord, text, source) VALUES (?,?,?,?)",
                (chunk.doc_id, chunk.ord, chunk.text, chunk.source),
            )
            self.db.execute(
                "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                (cur.lastrowid, sqlite_vec.serialize_float32(vec)),
            )
        self.db.commit()

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        rows = self.db.execute(
            "SELECT c.doc_id, c.ord, c.text, c.source, v.distance "
            "FROM vec_chunks v JOIN chunks c ON c.id = v.chunk_id "
            "WHERE v.embedding MATCH ? ORDER BY v.distance LIMIT ?",
            (sqlite_vec.serialize_float32(vector), k),
        ).fetchall()
        out = []
        for doc_id, ord_, text, source, distance in rows:
            chunk = Chunk(doc_id=doc_id, ord=ord_, text=text, source=source)
            out.append(RetrievedChunk(chunk=chunk, score=1.0 - float(distance)))
        return out

    def close(self) -> None:
        self.db.close()
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_index_store.py -v`
Expected: PASS (2 tests). *(If `sqlite_vec.load` errors on your dev OS, run this test on the Pi; the store is the one component whose native extension is platform-sensitive.)*

- [ ] **Step 5: Commit**

```bash
git add src/faraday/index_store.py tests/test_index_store.py
git commit -m "feat(m1): sqlite-vec single-file vector store"
```

### Task 11: Retriever

**Files:**
- Create: `src/faraday/retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_retriever.py`:
```python
from faraday.index_store import SqliteVecStore
from faraday.retriever import Retriever
from faraday.models import Chunk


def test_retriever_embeds_query_then_searches(tmp_path, fake_embedder):
    store = SqliteVecStore(str(tmp_path / "r.sqlite"), dim=fake_embedder.dim)
    chunks = [Chunk(doc_id="d", ord=0, text="alpha beta", source="s.txt"),
              Chunk(doc_id="d", ord=1, text="gamma delta", source="s.txt")]
    store.add_chunks(chunks, fake_embedder.embed([c.text for c in chunks]))
    retriever = Retriever(embedder=fake_embedder, store=store)
    results = retriever.search("alpha", k=1)
    assert results[0].chunk.text == "alpha beta"
    store.close()
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL — module/class not found.

- [ ] **Step 3: Implement `src/faraday/retriever.py`**

```python
from __future__ import annotations
from faraday.embedder import Embedder
from faraday.index_store import SqliteVecStore
from faraday.models import RetrievedChunk


class Retriever:
    def __init__(self, embedder: Embedder, store: SqliteVecStore):
        self.embedder = embedder
        self.store = store

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        query_vec = self.embedder.embed([query])[0]
        return self.store.search(query_vec, k=k)
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_retriever.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/faraday/retriever.py tests/test_retriever.py
git commit -m "feat(m1): retriever (embed query -> vector search)"
```

### Task 12: Prompt builder

**Files:**
- Create: `src/faraday/prompt.py`
- Test: `tests/test_prompt.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_prompt.py`:
```python
from faraday.prompt import build_messages
from faraday.models import RetrievedChunk, Chunk


def _rc(i, text):
    return RetrievedChunk(chunk=Chunk(doc_id="d", ord=i, text=text, source="s.txt"), score=1.0)


def test_messages_number_sources_and_instruct_citations():
    msgs = build_messages("What RAM?", [_rc(0, "4GB RAM"), _rc(1, "ARM CPU")])
    assert msgs[0]["role"] == "system"
    user = msgs[-1]["content"]
    assert "[1]" in user and "[2]" in user      # sources numbered from 1
    assert "4GB RAM" in user
    assert "What RAM?" in user


def test_abstention_instruction_present():
    msgs = build_messages("q", [_rc(0, "ctx")])
    assert "don't know" in msgs[0]["content"].lower()
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_prompt.py -v`
Expected: FAIL — module/function not found.

- [ ] **Step 3: Implement `src/faraday/prompt.py`**

```python
from __future__ import annotations
from faraday.models import RetrievedChunk

SYSTEM = (
    "You answer strictly from the provided sources. "
    "Cite every claim with bracketed source numbers like [1] or [2]. "
    "If the answer is not in the sources, say you don't know. "
    "Do not use outside knowledge."
)


def build_messages(query: str, chunks: list[RetrievedChunk]) -> list[dict]:
    blocks = []
    for i, rc in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {rc.chunk.source})\n{rc.chunk.text}")
    context = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    user = f"Sources:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_prompt.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/faraday/prompt.py tests/test_prompt.py
git commit -m "feat(m1): grounded prompt builder with numbered citations"
```

### Task 13: Citation verifier

**Files:**
- Create: `src/faraday/citations.py`
- Test: `tests/test_citations.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_citations.py`:
```python
from faraday.citations import extract_citations, classify_citations


def test_extract_unique_sorted_indices():
    assert extract_citations("Foo [2] bar [1][1].") == [1, 2]


def test_classify_valid_and_invalid():
    valid, invalid = classify_citations("Uses [1] and [3].", n_sources=2)
    assert valid == [1]          # [1] is in range (1..2)
    assert invalid == [3]        # [3] is hallucinated
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_citations.py -v`
Expected: FAIL — module/functions not found.

- [ ] **Step 3: Implement `src/faraday/citations.py`**

```python
from __future__ import annotations
import re

_CITE = re.compile(r"\[(\d+)\]")


def extract_citations(text: str) -> list[int]:
    return sorted({int(m) for m in _CITE.findall(text)})


def classify_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Split cited indices into (valid in 1..n_sources, invalid/hallucinated)."""
    valid, invalid = [], []
    for i in extract_citations(text):
        (valid if 1 <= i <= n_sources else invalid).append(i)
    return valid, invalid
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_citations.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/faraday/citations.py tests/test_citations.py
git commit -m "feat(m1): citation extraction + hallucination check"
```

### Task 14: LLM client (protocol + HTTP)

**Files:**
- Create: `src/faraday/llm_client.py`
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_llm_client.py`:
```python
from faraday.llm_client import LLMClient


def test_fake_llm_satisfies_protocol(fake_llm):
    assert isinstance(fake_llm, LLMClient)        # runtime_checkable
    out = fake_llm.complete([{"role": "user", "content": "hi"}])
    assert isinstance(out, str)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL — cannot import `LLMClient`.

- [ ] **Step 3: Implement `src/faraday/llm_client.py`**

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable
import httpx
from faraday.config import Settings


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, messages: list[dict], max_tokens: int = 512) -> str: ...


class HttpLLMClient:
    """Non-streaming chat completion against a local llama-server (streaming lands in M2)."""

    def __init__(self, settings: Settings | None = None, timeout: float = 120.0):
        self.settings = settings or Settings()
        self._client = httpx.Client(base_url=self.settings.gen_url, timeout=timeout)

    def complete(self, messages: list[dict], max_tokens: int = 512) -> str:
        resp = self._client.post(
            "/v1/chat/completions",
            json={"messages": messages, "max_tokens": max_tokens, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_llm_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/faraday/llm_client.py tests/test_llm_client.py
git commit -m "feat(m1): llm client protocol + http (non-streaming)"
```

### Task 15: Ingest pipeline

**Files:**
- Create: `src/faraday/ingest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_ingest.py`:
```python
from pathlib import Path
from faraday.index_store import SqliteVecStore
from faraday.ingest import ingest


def test_ingest_indexes_documents(tmp_path: Path, fake_embedder):
    (tmp_path / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
    (tmp_path / "b.md").write_text("# H\n\ndelta epsilon", encoding="utf-8")
    store = SqliteVecStore(str(tmp_path / "i.sqlite"), dim=fake_embedder.dim)
    stats = ingest(tmp_path, store=store, embedder=fake_embedder,
                   chunk_size=100, chunk_overlap=10)
    assert stats.documents == 2
    assert stats.chunks >= 2
    assert len(store.search(fake_embedder.embed(["alpha"])[0], k=1)) == 1
    store.close()


def test_ingest_skips_unsupported_files(tmp_path: Path, fake_embedder):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "img.png").write_bytes(b"\x89PNG")
    store = SqliteVecStore(str(tmp_path / "i.sqlite"), dim=fake_embedder.dim)
    stats = ingest(tmp_path, store=store, embedder=fake_embedder)
    assert stats.documents == 1
    assert stats.skipped == 1
    store.close()
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL — module/function not found.

- [ ] **Step 3: Implement `src/faraday/ingest.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from faraday.chunker import chunk_document
from faraday.embedder import Embedder
from faraday.index_store import SqliteVecStore
from faraday.parsers import load_document, TEXT_EXTS

SUPPORTED = TEXT_EXTS | {".pdf"}


@dataclass
class IngestStats:
    documents: int = 0
    chunks: int = 0
    skipped: int = 0


def ingest(source_dir, store: SqliteVecStore, embedder: Embedder,
           chunk_size: int = 1200, chunk_overlap: int = 200) -> IngestStats:
    source_dir = Path(source_dir)
    stats = IngestStats()
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED:
            stats.skipped += 1
            continue
        doc = load_document(path)
        chunks = chunk_document(doc, size=chunk_size, overlap=chunk_overlap)
        vectors = embedder.embed([c.text for c in chunks])
        store.add_chunks(chunks, vectors)
        stats.documents += 1
        stats.chunks += len(chunks)
    return stats
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/faraday/ingest.py tests/test_ingest.py
git commit -m "feat(m1): ingest pipeline (parse->chunk->embed->store)"
```

### Task 16: RAG engine

**Files:**
- Create: `src/faraday/rag.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_rag.py`:
```python
from faraday.rag import RagEngine
from faraday.retriever import Retriever
from faraday.index_store import SqliteVecStore
from faraday.models import Chunk


def _store(tmp_path, embedder):
    store = SqliteVecStore(str(tmp_path / "rag.sqlite"), dim=embedder.dim)
    chunks = [Chunk(doc_id="d", ord=0, text="The Pi 4 has 4GB RAM.", source="pi.txt"),
              Chunk(doc_id="d", ord=1, text="It uses an ARM CPU.", source="pi.txt")]
    store.add_chunks(chunks, embedder.embed([c.text for c in chunks]))
    return store


def test_answer_assembles_sources_and_valid_citations(tmp_path, fake_embedder, fake_llm):
    store = _store(tmp_path, fake_embedder)
    engine = RagEngine(Retriever(fake_embedder, store), fake_llm, top_k=2)
    ans = engine.answer("How much RAM?")
    assert ans.text == "Answer [1]."
    assert len(ans.sources) == 2
    assert ans.cited_indices == [1]
    assert ans.invalid_citations == []
    store.close()


def test_answer_flags_hallucinated_citation(tmp_path, fake_embedder, make_llm):
    store = _store(tmp_path, fake_embedder)
    engine = RagEngine(Retriever(fake_embedder, store), make_llm("Nope [9]."), top_k=2)
    ans = engine.answer("q")
    assert ans.invalid_citations == [9]
    store.close()
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_rag.py -v`
Expected: FAIL — module/class not found.

- [ ] **Step 3: Implement `src/faraday/rag.py`**

```python
from __future__ import annotations
from faraday.citations import classify_citations
from faraday.llm_client import LLMClient
from faraday.models import Answer
from faraday.prompt import build_messages
from faraday.retriever import Retriever


class RagEngine:
    def __init__(self, retriever: Retriever, llm: LLMClient, top_k: int = 4,
                 max_tokens: int = 512):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_tokens = max_tokens

    def answer(self, query: str) -> Answer:
        sources = self.retriever.search(query, k=self.top_k)
        messages = build_messages(query, sources)
        text = self.llm.complete(messages, max_tokens=self.max_tokens)
        valid, invalid = classify_citations(text, n_sources=len(sources))
        return Answer(text=text, sources=sources,
                      cited_indices=valid, invalid_citations=invalid)
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_rag.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/faraday/rag.py tests/test_rag.py
git commit -m "feat(m1): rag engine (retrieve->ground->generate->verify)"
```

### Task 17: CLI

**Files:**
- Create: `src/faraday/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test** `[on dev machine]`

`tests/test_cli.py`:
```python
from typer.testing import CliRunner
from faraday.cli import app

runner = CliRunner()


def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "ask" in result.output
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — cannot import `app`.

- [ ] **Step 3: Implement `src/faraday/cli.py`**

```python
from __future__ import annotations
import typer
from faraday.config import Settings
from faraday.embedder import HttpEmbedder
from faraday.index_store import SqliteVecStore
from faraday.ingest import ingest as run_ingest
from faraday.llm_client import HttpLLMClient
from faraday.rag import RagEngine
from faraday.retriever import Retriever

app = typer.Typer(help="Faraday — air-gapped personal RAG appliance")


@app.command()
def ingest(source: str, db: str = Settings().db_path):
    """Index a folder of documents into the local vector store."""
    s = Settings.from_env()
    store = SqliteVecStore(db, dim=s.embed_dim)
    stats = run_ingest(source, store=store, embedder=HttpEmbedder(s),
                       chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    store.close()
    typer.echo(f"Indexed {stats.documents} docs, {stats.chunks} chunks "
               f"({stats.skipped} skipped).")


@app.command()
def ask(question: str, db: str = Settings().db_path):
    """Answer a question from the indexed documents (fully offline)."""
    s = Settings.from_env()
    store = SqliteVecStore(db, dim=s.embed_dim)
    engine = RagEngine(Retriever(HttpEmbedder(s), store), HttpLLMClient(s),
                       top_k=s.top_k, max_tokens=s.max_tokens)
    ans = engine.answer(question)
    store.close()
    typer.echo("\n" + ans.text + "\n")
    typer.echo("Sources:")
    for i, rc in enumerate(ans.sources, start=1):
        typer.echo(f"  [{i}] {rc.chunk.source} (score {rc.score:.3f})")
    if ans.invalid_citations:
        typer.secho(f"  ! hallucinated citations: {ans.invalid_citations}",
                    fg=typer.colors.RED)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run to confirm pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full unit suite + lint**

Run: `pytest && ruff check src tests`
Expected: all tests pass (integration test deselected); ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/faraday/cli.py tests/test_cli.py
git commit -m "feat(m1): typer CLI (ingest, ask)"
```

### Task 18: End-to-end integration test on the Pi

**Files:**
- Create: `tests/test_integration_pi.py`
- Create: `examples/corpus/pi-facts.md`

- [ ] **Step 1: Create a tiny demo corpus** `[on dev machine]`

`examples/corpus/pi-facts.md`:
```markdown
# Raspberry Pi 4 facts

The Raspberry Pi 4 Model B is available with 1GB, 2GB, 4GB, or 8GB of RAM.
It uses a Broadcom BCM2711 with a quad-core ARM Cortex-A72 (64-bit) CPU.
It has no discrete GPU suitable for CUDA; LLM inference runs on the CPU.
```

- [ ] **Step 2: Write the integration test** `[on dev machine]`

`tests/test_integration_pi.py`:
```python
"""Runs only on the Pi with both llama-servers up: pytest -m integration"""
import pytest
from faraday.config import Settings
from faraday.embedder import HttpEmbedder
from faraday.index_store import SqliteVecStore
from faraday.ingest import ingest
from faraday.llm_client import HttpLLMClient
from faraday.rag import RagEngine
from faraday.retriever import Retriever


@pytest.mark.integration
def test_end_to_end_offline_answer(tmp_path):
    s = Settings()
    store = SqliteVecStore(str(tmp_path / "e2e.sqlite"), dim=s.embed_dim)
    stats = ingest("examples/corpus", store=store, embedder=HttpEmbedder(s))
    assert stats.documents >= 1

    engine = RagEngine(Retriever(HttpEmbedder(s), store), HttpLLMClient(s), top_k=s.top_k)
    ans = engine.answer("How much RAM can a Raspberry Pi 4 have?")
    assert "8gb" in ans.text.lower() or "8 gb" in ans.text.lower()
    assert ans.sources                      # retrieved something
    assert ans.invalid_citations == []      # no hallucinated sources
    store.close()
```

- [ ] **Step 3: Sync to the Pi and run it there** `[on dev machine]` → `[on Pi]`

Run:
```bash
bash scripts/sync.sh
ssh pi@raspberrypi.local 'cd ~/faraday && python3 -m venv .venv && . .venv/bin/activate \
  && pip install -e ".[dev]" && bash scripts/30_run_servers.sh && sleep 20 \
  && pytest -m integration -v'
```
Expected: `test_end_to_end_offline_answer PASSED`. (If the RAM assertion fails because the 1.5B model phrased the answer differently, read the printed answer and loosen the assertion to the fact actually stated — note this as the first real quality observation for M4.)

- [ ] **Step 4: Manual demo (the offline flex)** `[on Pi]`

Run:
```bash
ssh pi@raspberrypi.local 'cd ~/faraday && . .venv/bin/activate \
  && faraday ingest examples/corpus && faraday ask "What CPU does the Pi 4 use?"'
```
Expected: an answer mentioning ARM Cortex-A72 with a `Sources:` list. **This is the M0–M1 done state.**

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration_pi.py examples/corpus/pi-facts.md
git commit -m "test(m1): end-to-end offline integration test + demo corpus"
```

---

## Plan 1 done criteria

- [ ] M0: all four scripts committed; servers respond to `/health`, `/v1/chat/completions`, `/v1/embeddings`; baseline tok/s recorded in `results/baseline/`.
- [ ] M1: full unit suite green on the dev machine; ruff clean.
- [ ] Integration test green on the Pi; `faraday ask` returns a grounded, cited answer fully offline.
- [ ] First empirical observations (real tok/s; whether 1.5B answers the demo question well) noted — these seed Plan 3 (the M4 lab).

## Next plan

Plan 2 (M2): wrap `RagEngine` in a FastAPI service (`/chat` SSE + OpenAI-compatible `/v1/chat/completions`, both grounded), a minimal HTMX web UI showing streamed tokens + sources, GBNF-grammar structured citations, and systemd/docker packaging.
