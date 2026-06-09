# Faraday M4b — RAG Eval: Data + Run — Implementation Plan (Plan 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed the (already-built, merged) `faraday.eval` engine real data and run it: fetch a themed Apollo-era Wikipedia corpus, draft-then-curate a golden eval set with Claude, run the real `RagEngine` across the ablation grid on the Pi, score it (deterministic metrics + Claude judge), and commit the scorecard + ablation plot + findings.

**Architecture:** Record-then-judge (the M4a/Plan-1 pattern). Dev-time builds the corpus + golden set (Claude draft → human curate → commit). The **Pi** ingests the corpus per chunk-size, runs `RagEngine.answer` per `(config, question)`, and records raw JSONL (resumable). Scoring (deterministic metrics over all 9 configs + Claude-judge answer quality at the baseline config) runs where `anthropic` + an API key + the raw files are — the Pi — and the curated artifacts are committed from the dev box.

**Tech Stack:** Python 3.11+, `httpx` (already a dep — Wikipedia action API), `anthropic` (generator + judge, `claude-opus-4-8`), the merged `faraday.eval` engine, `faraday.ingest`/`retriever`/`rag` (real RAG on the Pi), matplotlib, pytest, ruff.

**Spec:** [../specs/2026-06-10-faraday-m4b-rag-evals-design.md](../specs/2026-06-10-faraday-m4b-rag-evals-design.md) · **Builds on:** Plan 1 engine ([2026-06-10-faraday-m4b-rag-eval-engine.md](./2026-06-10-faraday-m4b-rag-eval-engine.md), merged `91a77ad`).

**Prerequisites (execution-time):**
- `ANTHROPIC_API_KEY` exported in the Pi SSH session for the generate + judge steps (the eval harness is dev-time tooling; the Pi has LAN internet — this does **not** touch the appliance's runtime air-gap).
- The gen + embed llama-servers up (`scripts/30_run_servers.sh`) for the run step.

**Engine facts (from Plan 1, verified):** `AblationConfig(top_k, chunk_size, chunk_overlap).slug`; `EvalItem(id, question, answerable, relevant_doc, relevant_span, reference_answer)`; `load_golden(path)`; `aggregate(rows, items_by_id, size, overlap)`; `record_from_answer(cfg, qid, answer)`, `append_record(path, rec)`, `done_keys(path)`; `judge_rows(rows, items_by_id, judge)`, `make_scorecard(per_config)`, `render_ablation(per_config, out, metric)`; `JudgeVerdict(faithfulness, correctness, rationale)`, `AnthropicJudge`. Config paths: `CORPUS_DIR=examples/eval_corpus`, `EVAL_DIR=results/evals`, `GOLDEN_PATH`, `RAW_DIR=results/evals/raw`, `JUDGE_DIR=results/evals/judge`.

---

### Task 1: Config baseline + corpus fetcher (pure helpers TDD)

**Files:**
- Modify: `src/faraday/eval/config.py`
- Create: `src/faraday/eval/corpus.py`
- Test: `tests/test_eval_corpus.py`

- [ ] **Step 1: Add the baseline + article list to config.** Append to `src/faraday/eval/config.py` (after `TOP_KS`):

```python
# Judge answer-quality only at this baseline config (cost control); deterministic
# metrics still cover the full grid.
BASELINE = AblationConfig(top_k=4, chunk_size=1200, chunk_overlap=200)

# The themed corpus: Apollo-era crewed spaceflight (dense, overlapping facts so
# retrieval must be precise). ~15 articles.
CORPUS_TITLES = (
    "Apollo 11", "Apollo 13", "Apollo program", "Saturn V", "Neil Armstrong",
    "Buzz Aldrin", "Michael Collins (astronaut)", "Apollo Lunar Module",
    "Project Gemini", "Apollo 1", "Apollo 17", "Space Race",
    "Apollo command and service module", "Lunar Roving Vehicle", "Apollo 8",
)
```

> `BASELINE` references `AblationConfig`, defined above it in the same file — fine.

- [ ] **Step 2: Write the failing test.** Create `tests/test_eval_corpus.py`:

```python
from faraday.eval.corpus import parse_pages, slugify


def test_slugify_makes_safe_filenames():
    assert slugify("Apollo 11") == "apollo_11"
    assert slugify("Michael Collins (astronaut)") == "michael_collins_astronaut"


def test_parse_pages_extracts_text_and_url():
    payload = {"query": {"pages": {"123": {
        "title": "Apollo 11", "extract": "Apollo 11 was a spaceflight.",
        "fullurl": "https://en.wikipedia.org/wiki/Apollo_11"}}}}
    text, url = parse_pages(payload)
    assert text == "Apollo 11 was a spaceflight."
    assert url == "https://en.wikipedia.org/wiki/Apollo_11"


def test_parse_pages_raises_on_missing_extract():
    import pytest
    with pytest.raises(ValueError):
        parse_pages({"query": {"pages": {"-1": {"title": "Nope", "missing": ""}}}})
```

- [ ] **Step 3: Run — expect FAIL** (`ModuleNotFoundError: faraday.eval.corpus`):

`ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_eval_corpus.py -q"`

- [ ] **Step 4: Implement.** Create `src/faraday/eval/corpus.py`:

```python
"""Fetch the themed eval corpus from Wikipedia (dev-time, network) into committed
plain-text files + a SOURCES.md attribution. Uses the MediaWiki action API's
plaintext extracts (no HTML/markup cleaning needed). Pure helpers are unit-tested;
fetch_all() is a one-shot live script.
"""
from __future__ import annotations

import re
from pathlib import Path

import httpx

from faraday.eval import config

API = "https://en.wikipedia.org/w/api.php"


def slugify(title: str) -> str:
    s = title.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", s)


def parse_pages(payload: dict) -> tuple[str, str]:
    """Pull (plaintext extract, canonical url) out of an action-API response."""
    pages = payload["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page or not page["extract"].strip():
        raise ValueError(f"no extract for page: {page.get('title')!r}")
    return page["extract"], page.get("fullurl", "")


def fetch_extract(client: httpx.Client, title: str) -> tuple[str, str]:
    resp = client.get(API, params={
        "action": "query", "format": "json", "prop": "extracts|info",
        "titles": title, "explaintext": 1, "redirects": 1, "inprop": "url",
    })
    resp.raise_for_status()
    return parse_pages(resp.json())


def fetch_all(out_dir: Path | None = None) -> Path:
    """Fetch every CORPUS_TITLES article -> out_dir/<slug>.txt + SOURCES.md."""
    out_dir = out_dir or config.CORPUS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = ["# Eval corpus sources",
               "", "Wikipedia articles, text licensed CC BY-SA 4.0. "
               "Fetched via the MediaWiki action API (plaintext extracts).", ""]
    with httpx.Client(timeout=30.0, headers={"User-Agent": "faraday-eval/0.1"}) as client:
        for title in config.CORPUS_TITLES:
            text, url = fetch_extract(client, title)
            (out_dir / f"{slugify(title)}.txt").write_text(text, encoding="utf-8")
            sources.append(f"- **{title}** — {url}")
            print(f"  {slugify(title)}.txt  ({len(text)} chars)", flush=True)
    (out_dir / "SOURCES.md").write_text("\n".join(sources) + "\n", encoding="utf-8")
    return out_dir


if __name__ == "__main__":
    fetch_all()
```

- [ ] **Step 5: Run — expect PASS** (3 passed). **Step 6: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_eval_corpus.py -q && ruff check src/faraday/eval/corpus.py src/faraday/eval/config.py tests/test_eval_corpus.py"
git add src/faraday/eval/config.py src/faraday/eval/corpus.py tests/test_eval_corpus.py
git commit -m "feat(m4b): corpus fetcher (Wikipedia action API) + baseline config"
```

---

### Task 2: Fetch the corpus on the Pi, commit it

**Files:** Create `examples/eval_corpus/*.txt` + `examples/eval_corpus/SOURCES.md` (generated)

No unit test (one-shot live fetch); verified by inspecting committed output.

- [ ] **Step 1: Run the fetch on the Pi** (has LAN internet):

```bash
git push pi <branch>
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && python -m faraday.eval.corpus && ls -la examples/eval_corpus/"
```
Expected: 15 `*.txt` files (each tens-of-KB) + `SOURCES.md`.

- [ ] **Step 2: Bring the corpus to the dev box to commit** (`scp`, the M4a reverse path):

```bash
scp -r pi@raspberrypi.local:faraday/examples/eval_corpus "C:\projects\piai\examples\eval_corpus"
```

- [ ] **Step 3: Sanity-check + commit.** Confirm a couple files look like clean prose (e.g. open `apollo_11.txt`), then:

```bash
git add examples/eval_corpus
git commit -m "data(m4b): Apollo-era Wikipedia eval corpus (15 articles + SOURCES)"
```

> The corpus is committed plain text (air-gap-reproducible). `examples/corpus/` is git-ignored by the M1 `/corpus/` rule, but `examples/eval_corpus/` is **not** matched (the rule is anchored to repo-root `/corpus/`), so this commits cleanly.

---

### Task 3: `generate.py` — Claude drafts the golden set (pure helpers TDD)

**Files:**
- Create: `src/faraday/eval/generate.py`
- Test: `tests/test_eval_generate.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_eval_generate.py`:

```python
from faraday.eval.generate import build_item_prompt, locate_span


def test_locate_span_finds_quote_offsets():
    text = "Apollo 11 was the spaceflight that first landed humans on the Moon."
    span = locate_span(text, "first landed humans on the Moon")
    assert span is not None
    assert text[span[0]:span[1]] == "first landed humans on the Moon"


def test_locate_span_returns_none_when_absent():
    assert locate_span("some text", "not present here") is None


def test_build_item_prompt_includes_article_and_count():
    p = build_item_prompt("Apollo 11", "Apollo 11 landed in 1969.", n=5)
    assert "Apollo 11" in p
    assert "Apollo 11 landed in 1969." in p
    assert "5" in p
    assert "verbatim" in p.lower()  # instructs an exact supporting quote
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement.** Create `src/faraday/eval/generate.py`:

```python
"""Draft candidate golden-set items from the corpus with Claude (dev-time). The
model returns a verbatim supporting quote per item; we locate it in the source to
get reproducible char-offset spans. Output is a DRAFT for human curation, never
committed as-is. Pure helpers (prompt, span location) are unit-tested.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from faraday.eval import config


def locate_span(text: str, quote: str) -> tuple[int, int] | None:
    """Find quote in text -> (start, end). Exact first, then whitespace-tolerant."""
    i = text.find(quote)
    if i >= 0:
        return (i, i + len(quote))
    pattern = re.escape(quote.strip()).replace(r"\ ", r"\s+")
    m = re.search(pattern, text)
    return (m.start(), m.end()) if m else None


def build_item_prompt(title: str, article_text: str, n: int) -> str:
    return (
        f"From the Wikipedia article below, write {n} factual question/answer pairs "
        "for evaluating a retrieval system. Each must be answerable SOLELY from this "
        "article, with a short factual reference answer and a VERBATIM supporting "
        "quote copied exactly from the text.\n\n"
        f"Article: {title}\n\"\"\"\n{article_text}\n\"\"\"\n\n"
        "Return a JSON list; each element: "
        '{"question": str, "reference_answer": str, "supporting_quote": str}.'
    )


class AnthropicGenerator:
    """Live generator. Requires `anthropic` + ANTHROPIC_API_KEY. Not unit-tested."""

    def __init__(self, client=None, model: str = config.JUDGE_MODEL):
        import anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model

    def draft_for_article(self, title: str, text: str, n: int = 4) -> list[dict]:
        from pydantic import BaseModel

        class _Item(BaseModel):
            question: str
            reference_answer: str
            supporting_quote: str

        class _Items(BaseModel):
            items: list[_Item]

        resp = self.client.messages.parse(
            model=self.model, max_tokens=4096,
            messages=[{"role": "user", "content": build_item_prompt(title, text, n)}],
            output_format=_Items,
        )
        return [i.model_dump() for i in resp.parsed_output.items]


# Out-of-corpus questions for the abstention axis (hand-authored: plausible
# spaceflight questions NOT answerable from the Apollo-era corpus).
UNANSWERABLE = (
    "Which Space Shuttle was the first to reach orbit?",
    "Who was the first person to walk in space?",
    "What year did the International Space Station launch its first module?",
    "How many people have walked on Mars?",
    "What was the name of the first SpaceX crewed mission?",
    "Which country launched the Sputnik 1 satellite?",
)


def draft_golden(corpus_dir: Path, out_path: Path, gen: AnthropicGenerator,
                 per_article: int = 3) -> int:
    """Draft answerable items from each corpus file + append unanswerable ones."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for txt in sorted(Path(corpus_dir).glob("*.txt")):
            text = txt.read_text(encoding="utf-8")
            for it in gen.draft_for_article(txt.stem, text, per_article):
                span = locate_span(text, it["supporting_quote"])
                rec = {
                    "id": f"{txt.stem}_{n:03d}", "question": it["question"],
                    "answerable": True, "relevant_doc": txt.name,
                    "relevant_span": list(span) if span else None,
                    "reference_answer": it["reference_answer"],
                    "_quote": it["supporting_quote"],  # kept for human review; strip on curate
                }
                f.write(json.dumps(rec) + "\n")
                n += 1
        for i, q in enumerate(UNANSWERABLE):
            f.write(json.dumps({
                "id": f"unanswerable_{i:03d}", "question": q, "answerable": False,
                "relevant_doc": "", "relevant_span": None, "reference_answer": "",
            }) + "\n")
            n += 1
    return n


def main() -> None:
    out = config.EVAL_DIR / "golden_draft.jsonl"
    count = draft_golden(config.CORPUS_DIR, out, AnthropicGenerator())
    print(f"wrote {count} draft items to {out} (review -> {config.GOLDEN_PATH})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run — expect PASS** (3 passed). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_eval_generate.py -q && ruff check src/faraday/eval/generate.py tests/test_eval_generate.py"
git add src/faraday/eval/generate.py tests/test_eval_generate.py
git commit -m "feat(m4b): Claude golden-set generator (draft + span location)"
```

---

### Task 4: Generate the draft, curate it, commit `golden.jsonl`

**Files:** Create `results/evals/golden.jsonl` (curated)

This is the hybrid generate-then-curate step. No unit test; the golden set IS the trusted artifact.

- [ ] **Step 1: Draft with Claude on the Pi** (needs the key + corpus):

```bash
git push pi <branch>
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && export ANTHROPIC_API_KEY=<key> && python -m faraday.eval.generate"
scp pi@raspberrypi.local:faraday/results/evals/golden_draft.jsonl "C:\projects\piai\results\evals\golden_draft.jsonl"
```
Expected: ~51 draft items (15 articles × 3 + 6 unanswerable).

- [ ] **Step 2: Curate by hand → `results/evals/golden.jsonl`.** Review every line; for each:
  - **Keep** only items whose `relevant_span` is non-null and whose `text[span]` truly supports the answer (drop or fix nulls — a null means the quote wasn't located).
  - **Verify** the reference answer is correct and concise; fix wording.
  - **Drop** near-duplicates and any question answerable from multiple articles (ambiguous relevance).
  - **Strip** the `_quote` helper field.
  - Aim for **~30–40 answerable + the 6 unanswerable**, balanced across articles.
  Write the cleaned lines to `results/evals/golden.jsonl`.

- [ ] **Step 3: Validate the curated set loads + is well-formed.** Run (on the Pi after `git push pi`):

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && python -c 'from faraday.eval.dataset import load_golden; from faraday.eval import config; xs=load_golden(config.GOLDEN_PATH); print(len(xs),\"items;\", sum(x.answerable for x in xs),\"answerable\")'"
```
Expected: prints the counts; no exception. (Single-quote-safe: no nested doubles in the Python.)

- [ ] **Step 4: Commit the golden set** (not the draft):

```bash
git add results/evals/golden.jsonl
git commit -m "data(m4b): curated golden eval set (~40 items, incl. abstention)"
```

---

### Task 5: `runner.run()` — drive the real RagEngine per config (loop TDD)

**Files:**
- Modify: `src/faraday/eval/runner.py` (extend)
- Test: `tests/test_eval_runner.py` (extend)

The pure per-config loop (`run_config`) is tested with a fake engine; `build_engine`/`run` (Pi-only: sqlite-vec + servers) are exercised by the live run (Task 9).

- [ ] **Step 1: Write the failing test.** Append to `tests/test_eval_runner.py`:

```python
from faraday.eval.dataset import EvalItem  # noqa: E402
from faraday.eval.runner import run_config  # noqa: E402


class _FakeEngine:
    """Stands in for RagEngine: returns a canned Answer, records the question."""
    def __init__(self):
        self.asked = []

    def answer(self, query):
        self.asked.append(query)
        rc = RetrievedChunk(chunk=Chunk(doc_id="d", ord=0, text="x", source="moon.txt"),
                            score=0.9)
        return Answer(text="ans [1].", sources=[rc], cited_indices=[1], invalid_citations=[])


def _items():
    return [EvalItem("q1", "Q1?", True, "moon.txt", (0, 10), "a"),
            EvalItem("q2", "Q2?", True, "moon.txt", (0, 10), "b")]


def test_run_config_records_each_question(tmp_path):
    cfg = AblationConfig(top_k=4, chunk_size=1200, chunk_overlap=200)
    raw = tmp_path / "raw.jsonl"
    eng = _FakeEngine()
    n = run_config(cfg, eng, _items(), raw)
    assert n == 2 and eng.asked == ["Q1?", "Q2?"]
    assert done_keys(raw) == {(cfg.slug, "q1"), (cfg.slug, "q2")}


def test_run_config_is_resumable(tmp_path):
    cfg = AblationConfig(top_k=4, chunk_size=1200, chunk_overlap=200)
    raw = tmp_path / "raw.jsonl"
    append_record(raw, record_from_answer(cfg, "q1", _FakeEngine().answer("Q1?")))
    eng = _FakeEngine()
    n = run_config(cfg, eng, _items(), raw)
    assert n == 1 and eng.asked == ["Q2?"]   # q1 skipped (already done)
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: cannot import name 'run_config'`).

- [ ] **Step 3: Implement.** Edit `src/faraday/eval/runner.py`. Replace the import block at the top with:

```python
from __future__ import annotations

import json
from pathlib import Path

from faraday.config import Settings
from faraday.eval import config
from faraday.eval.config import AblationConfig
from faraday.eval.dataset import EvalItem, load_golden
from faraday.eval.metrics import is_abstention
from faraday.models import Answer
```

Then append, after `done_keys`:

```python
def _raw_path(cfg: AblationConfig) -> Path:
    return config.RAW_DIR / f"{cfg.slug}.jsonl"


def run_config(cfg: AblationConfig, engine, items: list[EvalItem], raw_path: Path) -> int:
    """Ask each not-yet-done question through `engine` and record raw. Returns #new."""
    done = done_keys(raw_path)
    n = 0
    for item in items:
        if (cfg.slug, item.id) in done:
            continue
        answer = engine.answer(item.question)
        append_record(raw_path, record_from_answer(cfg, item.id, answer))
        n += 1
    return n


def build_engine(chunk_size: int, overlap: int, top_k: int, settings: Settings):
    """Pi-only: ingest the corpus at this chunk-size into a fresh store, wire a real
    RagEngine at this top_k. Needs sqlite-vec + the embed/gen servers up."""
    from faraday.embedder import HttpEmbedder
    from faraday.index_store import SqliteVecStore
    from faraday.ingest import ingest
    from faraday.llm_client import HttpLLMClient
    from faraday.rag import RagEngine
    from faraday.retriever import Retriever

    db = config.EVAL_DIR / f"store_c{chunk_size}.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()  # fresh store (CREATE TABLE IF NOT EXISTS would otherwise dup)
    embedder = HttpEmbedder(settings)
    store = SqliteVecStore(str(db), dim=settings.embed_dim)
    ingest(config.CORPUS_DIR, store, embedder, chunk_size=chunk_size, chunk_overlap=overlap)
    retriever = Retriever(embedder, store)
    return RagEngine(retriever, HttpLLMClient(settings), top_k=top_k)


def run() -> None:
    """Full grid on the Pi: ingest once per chunk-size, loop top_k, record raw."""
    settings = Settings.from_env()
    items = load_golden(config.GOLDEN_PATH)
    by_size: dict[int, int] = {}
    for cfg in config.configs():
        by_size[cfg.chunk_size] = cfg.chunk_overlap
    for size, overlap in sorted(by_size.items()):
        for top_k in config.TOP_KS:
            cfg = AblationConfig(top_k=top_k, chunk_size=size, chunk_overlap=overlap)
            print(f"--- {cfg.slug} ---", flush=True)
            engine = build_engine(size, overlap, top_k, settings)
            made = run_config(cfg, engine, items, _raw_path(cfg))
            print(f"    recorded {made} new rows", flush=True)


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run — expect PASS** (4 passed: the 2 from Plan 1 + 2 new). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_eval_runner.py -q && ruff check src/faraday/eval/runner.py tests/test_eval_runner.py"
git add src/faraday/eval/runner.py tests/test_eval_runner.py
git commit -m "feat(m4b): runner.run_config (tested loop) + build_engine/run (Pi wiring)"
```

---

### Task 6: On-Pi runner script `scripts/80_run_evals.sh`

**Files:** Create `scripts/80_run_evals.sh`

- [ ] **Step 1: Write the script.** Create `scripts/80_run_evals.sh`:

```bash
#!/usr/bin/env bash
# Run ON the Raspberry Pi. Drives the M4b RAG eval RUN phase:
#   for each (top_k x chunk_size) config -> ingest corpus at that chunk-size ->
#   RagEngine.answer each golden question -> record results/evals/raw/<slug>.jsonl.
# Resumable (re-running skips done (config, question)). Needs the gen + embed
# servers up. Scoring (judge + scorecard) is a separate dev/Pi step (report.py).
set -euo pipefail
cd "$HOME/faraday"
# shellcheck disable=SC1091
source .venv/bin/activate

# Servers must be up (embed :8081 for ingest/retrieval, gen :8080 for answers).
if ! curl -sf http://localhost:8081/health >/dev/null 2>&1; then
  echo "Embed/gen servers not healthy — starting them..."
  bash scripts/30_run_servers.sh
  echo "Waiting ~20s for models to load..." && sleep 20
fi

echo "throttle (0x0 = healthy): $(vcgencmd get_throttled)"
python -m faraday.eval.runner
echo "Done. Raw rows in results/evals/raw/. Next: score with report.py (needs ANTHROPIC_API_KEY)."
```

- [ ] **Step 2: Make executable + commit.**

```bash
git update-index --add --chmod=+x scripts/80_run_evals.sh
git add scripts/80_run_evals.sh
git commit -m "feat(m4b): on-Pi eval run script (servers + resumable runner)"
```

---

### Task 7: `report.py` — freeze judge + wire `main()` (TDD on the freeze)

**Files:**
- Modify: `src/faraday/eval/report.py` (extend)
- Test: `tests/test_eval_report.py` (extend)

`load_or_score` caches judge verdicts to disk so re-scoring never re-calls the API (spec §10). `main()` is thin glue (load → metrics over all configs → judge at baseline → scorecard + plot), exercised at run time.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_eval_report.py`:

```python
from faraday.eval.report import load_or_score  # noqa: E402


class BoomJudge:
    def score(self, **kwargs):
        raise AssertionError("should not be called when cache exists")


def test_load_or_score_writes_then_reads_cache(tmp_path):
    items = {"q1": _item("q1", True)}
    rows = _rows("k4_c1200_o200")
    cache = tmp_path / "judge_k4.jsonl"
    first = load_or_score(rows, items, FakeJudge(), cache)   # scores + writes cache
    assert first["q1"].faithfulness == 5
    assert cache.exists()
    again = load_or_score(rows, items, BoomJudge(), cache)   # loads cache, no judge call
    assert again["q1"].correctness == 4
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: cannot import name 'load_or_score'`).

- [ ] **Step 3: Implement.** Edit `src/faraday/eval/report.py`. Add `import json` and `from faraday.eval.config ...` is not needed; add these imports under the existing `from pathlib import Path`:

```python
import json
```

Then append (after `judge_rows`):

```python
def load_or_score(rows: list[dict], items_by_id: dict[str, EvalItem],
                  judge: Judge, cache_path: Path) -> dict[str, JudgeVerdict]:
    """Judge answered rows, freezing verdicts to cache_path. If the cache exists,
    load it and skip the API entirely (re-score without re-calling Claude)."""
    if Path(cache_path).exists():
        out: dict[str, JudgeVerdict] = {}
        for line in Path(cache_path).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["qid"]] = JudgeVerdict(d["faithfulness"], d["correctness"],
                                             d["rationale"])
        return out
    verdicts = judge_rows(rows, items_by_id, judge)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(cache_path).open("w") as f:
        for qid, v in verdicts.items():
            f.write(json.dumps({"qid": qid, "faithfulness": v.faithfulness,
                                "correctness": v.correctness, "rationale": v.rationale}) + "\n")
    return verdicts
```

Then append a `main()` at the end:

```python
def main() -> None:
    """Score the recorded run: deterministic metrics over ALL configs + judge answer
    quality at the BASELINE config only (cost control). Writes scorecard + plot."""
    from faraday.eval import config
    from faraday.eval.dataset import load_golden
    from faraday.eval.judge import AnthropicJudge
    from faraday.eval.metrics import aggregate
    from faraday.eval.runner import done_keys  # noqa: F401  (kept for symmetry)

    items = load_golden(config.GOLDEN_PATH)
    by_id = {it.id: it for it in items}
    per_config: dict[str, dict] = {}
    for cfg in config.configs():
        raw = config.RAW_DIR / f"{cfg.slug}.jsonl"
        if not raw.exists():
            continue
        rows = [json.loads(ln) for ln in raw.read_text().splitlines() if ln.strip()]
        m = aggregate(rows, by_id, size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        if cfg.slug == config.BASELINE.slug:  # judge answer quality at baseline only
            cache = config.JUDGE_DIR / f"{cfg.slug}.jsonl"
            verdicts = load_or_score(rows, by_id, AnthropicJudge(), cache)
            if verdicts:
                m["faithfulness"] = sum(v.faithfulness for v in verdicts.values()) / len(verdicts)
                m["correctness"] = sum(v.correctness for v in verdicts.values()) / len(verdicts)
        per_config[cfg.slug] = m

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (config.EVAL_DIR / "scorecard.md").write_text(make_scorecard(per_config))
    render_ablation(per_config, config.EVAL_DIR / "ablations.png", metric="recall_at_k")
    print(f"wrote scorecard.md + ablations.png for {len(per_config)} configs")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run — expect PASS** (4 passed: 3 from Plan 1 + 1 new). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_eval_report.py -q && ruff check src/faraday/eval/report.py tests/test_eval_report.py"
git add src/faraday/eval/report.py tests/test_eval_report.py
git commit -m "feat(m4b): judge freezing (load_or_score) + report.main wiring"
```

---

### Task 8: `.gitignore` + full-suite regression + lint

**Files:** Modify `.gitignore`

- [ ] **Step 1: Ignore raw/judge JSONL + the draft + scratch stores; keep curated outputs.** Add to `.gitignore` under the benchmark section:

```
# M4b eval: ignore raw run rows, frozen judge caches, the draft, and scratch stores;
# keep golden.jsonl, scorecard.md, ablations.png, findings.md, SOURCES.md.
results/evals/raw/
results/evals/judge/
results/evals/golden_draft.jsonl
results/evals/store_*.sqlite
```

- [ ] **Step 2: Full suite on the Pi.** Run:

`ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest -q"`
Expected: all prior + the new corpus/generate/runner/report tests pass (≈79 total); integration deselected.

- [ ] **Step 3: Full lint.** Run:

`ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && ruff check src tests"`
Expected: `All checks passed!`

- [ ] **Step 4: Commit the ignore rule.**

```bash
git add .gitignore
git commit -m "chore(m4b): gitignore eval raw/judge/draft/scratch, keep curated outputs"
```

---

### Task 9: Live run, score, findings

**Files:** Create `results/evals/README.md`, `results/evals/findings.md`; commit `scorecard.md`, `ablations.png`

- [ ] **Step 1: Run the eval on the Pi** (servers + ~3–4 h, resumable; detached):

```bash
git push pi <branch>
ssh pi@raspberrypi.local "cd ~/faraday && setsid nohup bash scripts/80_run_evals.sh >/tmp/evals.log 2>&1 </dev/null & echo started"
# monitor: ssh pi@raspberrypi.local "tail -f /tmp/evals.log"; rows accrue in results/evals/raw/
```

- [ ] **Step 2: Score it** (deterministic metrics + judge at baseline; needs the key):

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && export ANTHROPIC_API_KEY=<key> && python -m faraday.eval.report && cat results/evals/scorecard.md"
```

- [ ] **Step 3: Bring artifacts to the dev box.**

```bash
scp pi@raspberrypi.local:faraday/results/evals/scorecard.md "C:\projects\piai\results\evals\scorecard.md"
scp pi@raspberrypi.local:faraday/results/evals/ablations.png "C:\projects\piai\results\evals\ablations.png"
```

- [ ] **Step 4: Write `results/evals/README.md`** (artifact guide):

```markdown
# M4b — RAG Evaluation Results

RAG quality of Faraday (Qwen2.5-1.5B Q4_K_M gen + bge-small embed) over an
Apollo-era Wikipedia corpus, across a top_k × chunk-size ablation grid.

| File | What |
|---|---|
| `golden.jsonl` | The curated eval set (~40 items incl. abstention); source-span relevance labels. |
| `scorecard.md` | Per-config metrics: recall@k, MRR, citation validity, abstention; + judge faithfulness/correctness at the baseline config. |
| `ablations.png` | recall@k across the grid — what top_k/chunk-size actually move retrieval. |
| `findings.md` | The narrative: best config, what the ablations show, judge results, abstention behavior. |
| `SOURCES.md` | Wikipedia attribution (CC BY-SA). |
| `raw/`, `judge/` | Per-config raw rows + frozen judge verdicts (git-ignored). |

## Reproduce
On the Pi (servers up): `bash scripts/80_run_evals.sh` then
`ANTHROPIC_API_KEY=… python -m faraday.eval.report`.
```

- [ ] **Step 5: Write `results/evals/findings.md`** from the actual scorecard — the best config, how recall@k moves with top_k/chunk-size, the judge faithfulness/correctness at baseline, abstention accuracy, and any surprises (engineering candor). Commit:

```bash
git add results/evals/README.md results/evals/findings.md results/evals/scorecard.md results/evals/ablations.png
git commit -m "docs(m4b): RAG eval results + findings (scorecard, ablations, narrative)"
```

---

## Self-Review

**1. Spec coverage** (data + run portion):

| Spec section | Task |
|---|---|
| §2/§5 themed Wikipedia corpus (committed + SOURCES) | Tasks 1–2 (`corpus.py`, fetch, commit) |
| §6 hybrid generate-then-curate golden set; source-span labels | Tasks 3–4 (`generate.py` draft + `locate_span` → spans; manual curate) |
| §4/§8 on-Pi run: ingest-per-chunk-size → RagEngine.answer → record raw; resumable | Task 5 (`run_config`/`build_engine`/`run`), Task 6 (`80_run_evals.sh`) |
| §7 metrics over the grid; §7 judge faithfulness+correctness | Task 7 (`report.main` aggregate-all + judge-baseline) |
| §9 record-then-judge; §10 judge frozen to disk | Task 7 (`load_or_score` cache) |
| §10 raw/judge git-ignored, curated committed | Task 8 (`.gitignore`) |
| §11 testing (corpus/generate/runner/report pure cores) | Tasks 1/3/5/7 with fakes |
| §13 definition of done: scorecard + ablation plot + findings committed | Task 9 |

**Decisions recorded:** (a) Judge runs only at `BASELINE` (k4_c1200_o200), not all 9 configs — deterministic retrieval metrics are the ablation signal; judging every config would be ~9× the API cost for little added signal. (b) Corpus fetched via the MediaWiki action API's `explaintext` (plain text — no markup cleaning). (c) Everything runs on the Pi (it has network, `anthropic`, matplotlib, sqlite-vec, servers, corpus); curated artifacts `scp` back to commit (the M4a reverse path).

**2. Placeholder scan:** No TBD/TODO/"handle edge cases". `<key>`/`<branch>` are explicit user-supplied execution values, not code placeholders. Manual-curation (Task 4) and findings-authoring (Task 9) are inherently human steps, specified with concrete acceptance criteria.

**3. Type consistency:** `AblationConfig`/`.slug`, `EvalItem`, `JudgeVerdict`, the raw-row dict keys, and `aggregate(...,size,overlap)` all match Plan 1's definitions (verified against the merged engine). `run_config(cfg, engine, items, raw_path)`, `build_engine(chunk_size, overlap, top_k, settings)`, `load_or_score(rows, items_by_id, judge, cache_path)` are self-consistent across their tasks and call sites. `config.BASELINE.slug` (k4_c1200_o200) is the key linking `report.main`'s judge gate to a real grid config.

**Verdict:** data+run plan is complete, spec-covering, and placeholder-free.
