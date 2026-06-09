# Faraday M4b — RAG Eval Engine — Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reusable RAG-evaluation engine — config, dataset loader, deterministic metrics, the Claude-as-judge, the on-Pi record runner, and the report aggregator — as pure, unit-tested code with an injected (fake-in-tests) judge and RAG engine.

**Architecture:** Record-then-judge (mirrors M4a). The Pi runs the real `RagEngine` per ablation config and records raw outputs as JSONL; dev-side computes deterministic metrics (recall@k/MRR/citation/abstention) + Claude-judge answer scores; `report.py` aggregates into a scorecard + ablation table + plot. Everything is built behind Protocols so the pure core tests with fakes (no Pi, no network).

**Tech Stack:** Python 3.11+, `anthropic` SDK (`messages.parse` + Pydantic, model `claude-opus-4-8`), matplotlib, pytest, ruff. Reuses `faraday.rag.RagEngine`, `faraday.ingest`, `faraday.retriever`, `faraday.citations`.

**Spec:** [../specs/2026-06-10-faraday-m4b-rag-evals-design.md](../specs/2026-06-10-faraday-m4b-rag-evals-design.md)

**Scope note:** This plan builds the **engine**. Plan 2 (separate) builds the **data + run**: the Wikipedia corpus, the Claude-drafted-then-curated `golden.jsonl`, `scripts/80_run_evals.sh`, the live Pi run, and `findings.md`.

**Key codebase facts (verified):** `RagEngine.answer(query) -> Answer{text, sources:list[RetrievedChunk], cited_indices, invalid_citations}`; `RetrievedChunk.chunk` has `(doc_id, ord, text, source)` — **no char offsets**, so recall spans are recomputed from `ord` + `(chunk_size, chunk_overlap)` (chunker: `step = size - overlap`, `start = ord*step`). `classify_citations(text, n) -> (valid, invalid)`. `ingest(dir, store, embedder, chunk_size, chunk_overlap)`. Prompt already instructs abstention ("if not in the sources, say you don't know").

---

### Task 1: Scaffold `faraday/eval/` + config + deps + fixtures

**Files:**
- Create: `src/faraday/eval/__init__.py`, `src/faraday/eval/config.py`
- Modify: `pyproject.toml:17`
- Create: `tests/eval_samples.py`

- [ ] **Step 1: Package marker + dep.** Create `src/faraday/eval/__init__.py`:

```python
"""M4b RAG evaluation engine: dataset, metrics, judge, runner, report."""
```

In `pyproject.toml`, change line 17 from `dev = ["pytest>=8.0", "ruff>=0.5", "matplotlib>=3.7"]` to:

```toml
dev = ["pytest>=8.0", "ruff>=0.5", "matplotlib>=3.7", "anthropic>=0.69"]
```

- [ ] **Step 2: Write `config.py`.** Create `src/faraday/eval/config.py`:

```python
"""M4b eval config: the ablation grid, paths, and the judge model. Pure data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Judge model. Default per the claude-api skill; swap to "claude-sonnet-4-6" to cut cost.
JUDGE_MODEL = "claude-opus-4-8"

CORPUS_DIR = Path("examples/eval_corpus")
EVAL_DIR = Path("results/evals")
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"
RAW_DIR = EVAL_DIR / "raw"      # raw_<slug>.jsonl (git-ignored via *.jsonl? no — see Plan 2)
JUDGE_DIR = EVAL_DIR / "judge"  # judge_<slug>.jsonl (frozen judge outputs)

# Ablation grid: chunk-size brackets the real default (1200 chars); overlap = size//6.
CHUNK_SIZES = (600, 1200, 2400)
TOP_KS = (2, 4, 8)


@dataclass(frozen=True)
class AblationConfig:
    top_k: int
    chunk_size: int
    chunk_overlap: int

    @property
    def slug(self) -> str:
        return f"k{self.top_k}_c{self.chunk_size}_o{self.chunk_overlap}"


def configs() -> list[AblationConfig]:
    """The core grid: 3 chunk-sizes x 3 top_k = 9 configs (overlap = size//6)."""
    return [
        AblationConfig(top_k=k, chunk_size=cs, chunk_overlap=cs // 6)
        for cs in CHUNK_SIZES
        for k in TOP_KS
    ]
```

- [ ] **Step 3: Shared test fixtures.** Create `tests/eval_samples.py`:

```python
"""Shared fixtures for the M4b eval-engine tests (no Pi, no network)."""

# A tiny golden set: one answerable (with a labeled span), one unanswerable.
GOLDEN_JSONL = (
    '{"id":"q1","question":"When did Apollo 11 land?","answerable":true,'
    '"relevant_doc":"moon.txt","relevant_span":[100,260],'
    '"reference_answer":"July 1969."}\n'
    '{"id":"q2","question":"What is the capital of Mars?","answerable":false,'
    '"relevant_doc":"","relevant_span":[0,0],"reference_answer":""}\n'
)
```

- [ ] **Step 4: Commit.**

```bash
git add src/faraday/eval/__init__.py src/faraday/eval/config.py pyproject.toml tests/eval_samples.py
git commit -m "feat(m4b): scaffold eval engine package + config + anthropic dep"
```

> **Pi-side once:** `ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pip install -e '.[dev]'"` to pull in `anthropic` before running tests there.

---

### Task 2: `dataset.py` — golden-set schema + loader (TDD)

**Files:**
- Create: `src/faraday/eval/dataset.py`
- Test: `tests/test_eval_dataset.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_eval_dataset.py`:

```python
from faraday.eval.dataset import EvalItem, load_golden

from eval_samples import GOLDEN_JSONL


def test_load_golden_parses_items(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(GOLDEN_JSONL)
    items = load_golden(path)
    assert [i.id for i in items] == ["q1", "q2"]
    a = items[0]
    assert isinstance(a, EvalItem)
    assert a.answerable is True
    assert a.relevant_doc == "moon.txt"
    assert a.relevant_span == (100, 260)
    assert items[1].answerable is False


def test_load_golden_skips_blank_lines(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text("\n" + GOLDEN_JSONL + "\n")
    assert len(load_golden(path)) == 2
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: faraday.eval.dataset`):

`ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_eval_dataset.py -q"`

- [ ] **Step 3: Implement.** Create `src/faraday/eval/dataset.py`:

```python
"""The golden eval set: the EvalItem schema + a JSONL loader. Pure."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    answerable: bool
    relevant_doc: str           # source filename, e.g. "moon.txt" ("" if unanswerable)
    relevant_span: tuple[int, int]  # [start, end) char offsets in the source ((0,0) if N/A)
    reference_answer: str


def load_golden(path: Path) -> list[EvalItem]:
    """Parse golden.jsonl (one JSON object per non-blank line) into EvalItems."""
    items: list[EvalItem] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        span = d.get("relevant_span") or [0, 0]
        items.append(EvalItem(
            id=d["id"], question=d["question"], answerable=bool(d["answerable"]),
            relevant_doc=d.get("relevant_doc", ""), relevant_span=(span[0], span[1]),
            reference_answer=d.get("reference_answer", ""),
        ))
    return items
```

- [ ] **Step 4: Run — expect PASS** (2 passed). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_eval_dataset.py -q && ruff check src/faraday/eval/dataset.py tests/test_eval_dataset.py"
git add src/faraday/eval/dataset.py tests/test_eval_dataset.py
git commit -m "feat(m4b): EvalItem schema + golden.jsonl loader"
```

---

### Task 3: `metrics.py` — the deterministic core (TDD)

**Files:**
- Create: `src/faraday/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

This is the unit-tested heart: span-overlap recall (recomputing chunk spans from `ord` + config), MRR, citation validity, abstention.

- [ ] **Step 1: Write the failing test.** Create `tests/test_eval_metrics.py`:

```python
from faraday.eval.dataset import EvalItem
from faraday.eval.metrics import (
    aggregate,
    chunk_is_relevant,
    is_abstention,
)


def _item(qid, answerable, doc="moon.txt", span=(100, 260)):
    return EvalItem(qid, "?", answerable, doc if answerable else "",
                    span if answerable else (0, 0), "ref")


def test_chunk_span_overlap_uses_ord_and_config():
    item = _item("q1", True, "moon.txt", (100, 260))
    # size=200, overlap=0 -> step=200. ord=0 covers [0,200): overlaps [100,260). ord=2 -> [400,600): no.
    assert chunk_is_relevant("moon.txt", 0, item, size=200, overlap=0) is True
    assert chunk_is_relevant("moon.txt", 2, item, size=200, overlap=0) is False
    assert chunk_is_relevant("other.txt", 0, item, size=200, overlap=0) is False  # wrong doc


def test_is_abstention_detects_dont_know():
    assert is_abstention("I don't know based on the sources.") is True
    assert is_abstention("Apollo 11 landed in July 1969 [1].") is False


def test_aggregate_computes_recall_mrr_citation_abstention():
    items = {"q1": _item("q1", True), "q2": _item("q2", False)}
    # q1 (answerable): top-2 retrieved; ord=0 is relevant (rank 1). Cites [1] valid.
    # q2 (unanswerable): correctly abstains.
    rows = [
        {"qid": "q1", "retrieved": [{"source": "moon.txt", "ord": 0},
                                    {"source": "moon.txt", "ord": 5}],
         "answer": "July 1969 [1].", "cited": [1], "invalid": [2], "abstained": False},
        {"qid": "q2", "retrieved": [{"source": "x.txt", "ord": 0}],
         "answer": "I don't know.", "cited": [], "invalid": [], "abstained": True},
    ]
    m = aggregate(rows, items, size=200, overlap=0)
    assert m["recall_at_k"] == 1.0          # q1 found a relevant chunk
    assert m["mrr"] == 1.0                   # relevant at rank 1
    assert m["citation_validity"] == 0.5     # 1 valid / (1 valid + 1 invalid)
    assert m["abstention_accuracy"] == 1.0   # q1 answered (correct), q2 abstained (correct)
    assert m["n_answerable"] == 1
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement.** Create `src/faraday/eval/metrics.py`:

```python
"""Deterministic RAG metrics over recorded raw rows. Pure — no Pi, no network.

Retrieved chunks carry only (source, ord), so a chunk's char span is recomputed
from the chunker's geometry: step = size - overlap, start = ord*step, end = start+size.
Relevance = the chunk's span overlaps the labeled relevant_span in the same source doc.
"""
from __future__ import annotations

from faraday.eval.dataset import EvalItem

_ABSTAIN_PHRASES = (
    "don't know", "do not know", "not in the source", "not contain",
    "no information", "cannot answer", "can't answer", "couldn't find",
    "could not find", "unable to answer",
)


def is_abstention(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _ABSTAIN_PHRASES)


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def chunk_is_relevant(source: str, ord_: int, item: EvalItem, size: int, overlap: int) -> bool:
    if source != item.relevant_doc:
        return False
    step = size - overlap
    start = ord_ * step
    return _overlaps((start, start + size), item.relevant_span)


def _first_relevant_rank(row: dict, item: EvalItem, size: int, overlap: int) -> int | None:
    for rank, c in enumerate(row["retrieved"], start=1):
        if chunk_is_relevant(c["source"], c["ord"], item, size, overlap):
            return rank
    return None


def aggregate(rows: list[dict], items_by_id: dict[str, EvalItem],
              size: int, overlap: int) -> dict:
    """Compute all deterministic metrics for one config's rows."""
    answerable = [r for r in rows if items_by_id[r["qid"]].answerable]
    ranks = [_first_relevant_rank(r, items_by_id[r["qid"]], size, overlap) for r in answerable]
    hits = [rk for rk in ranks if rk is not None]
    recall = len(hits) / len(answerable) if answerable else 0.0
    mrr = (sum(1.0 / rk for rk in hits) / len(answerable)) if answerable else 0.0

    valid = sum(len(r["cited"]) for r in rows)
    invalid = sum(len(r["invalid"]) for r in rows)
    citation_validity = valid / (valid + invalid) if (valid + invalid) else 1.0

    correct_abstain = sum(
        1 for r in rows
        if r["abstained"] == (not items_by_id[r["qid"]].answerable)
    )
    abstention_accuracy = correct_abstain / len(rows) if rows else 0.0

    return {
        "recall_at_k": recall, "mrr": mrr, "citation_validity": citation_validity,
        "abstention_accuracy": abstention_accuracy,
        "n_answerable": len(answerable), "n_total": len(rows),
    }
```

- [ ] **Step 4: Run — expect PASS** (3 passed). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_eval_metrics.py -q && ruff check src/faraday/eval/metrics.py tests/test_eval_metrics.py"
git add src/faraday/eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat(m4b): deterministic metrics (recall@k/MRR/citation/abstention)"
```

---

### Task 4: `judge.py` — Claude-as-judge behind a Protocol (TDD)

**Files:**
- Create: `src/faraday/eval/judge.py`
- Test: `tests/test_eval_judge.py`

The judge is injected (Protocol) so tests use a fake. The real `AnthropicJudge` uses `messages.parse` + Pydantic per the claude-api skill; its only unit-tested logic is the pure prompt builder.

- [ ] **Step 1: Write the failing test.** Create `tests/test_eval_judge.py`:

```python
from faraday.eval.judge import JudgeVerdict, build_judge_prompt


def test_build_judge_prompt_includes_all_parts():
    p = build_judge_prompt(
        question="When did Apollo 11 land?",
        reference_answer="July 1969.",
        context="[1] Apollo 11 landed on 20 July 1969.",
        answer="It landed in July 1969 [1].",
    )
    assert "When did Apollo 11 land?" in p
    assert "July 1969." in p
    assert "Apollo 11 landed on 20 July 1969" in p
    assert "It landed in July 1969 [1]." in p
    assert "faithfulness" in p.lower() and "correctness" in p.lower()


def test_judge_verdict_holds_scores():
    v = JudgeVerdict(faithfulness=5, correctness=4, rationale="grounded; minor omission")
    assert v.faithfulness == 5 and v.correctness == 4
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement.** Create `src/faraday/eval/judge.py`:

```python
"""Claude-as-judge for answer faithfulness + correctness. Dev-time only (never on
the Pi). The judge is a Protocol; tests inject a fake. The real AnthropicJudge uses
the anthropic SDK's messages.parse (structured output) with model claude-opus-4-8.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from faraday.eval import config


@dataclass(frozen=True)
class JudgeVerdict:
    faithfulness: int   # 1-5: is the answer grounded in the retrieved context (no hallucination)?
    correctness: int    # 1-5: does it match the reference answer?
    rationale: str


class Judge(Protocol):
    def score(self, *, question: str, reference_answer: str,
              context: str, answer: str) -> JudgeVerdict: ...


def build_judge_prompt(*, question: str, reference_answer: str,
                       context: str, answer: str) -> str:
    return (
        "You are grading a retrieval-augmented answer. Score two axes from 1-5.\n"
        "- faithfulness: is EVERY claim supported by the Retrieved context? "
        "(5 = fully grounded, 1 = hallucinated)\n"
        "- correctness: does the Answer match the Reference answer? "
        "(5 = fully correct, 1 = wrong)\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Answer to grade:\n{answer}\n\n"
        "Return faithfulness, correctness, and a one-sentence rationale."
    )


class AnthropicJudge:
    """Real judge. Requires `anthropic` + ANTHROPIC_API_KEY. Not unit-tested (live)."""

    def __init__(self, client=None, model: str = config.JUDGE_MODEL):
        import anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model

    def score(self, *, question: str, reference_answer: str,
              context: str, answer: str) -> JudgeVerdict:
        from pydantic import BaseModel  # provided transitively by anthropic

        class _Scores(BaseModel):
            faithfulness: int
            correctness: int
            rationale: str

        prompt = build_judge_prompt(question=question, reference_answer=reference_answer,
                                    context=context, answer=answer)
        resp = self.client.messages.parse(
            model=self.model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=_Scores,
        )
        s = resp.parsed_output
        return JudgeVerdict(faithfulness=int(s.faithfulness),
                            correctness=int(s.correctness), rationale=s.rationale)
```

- [ ] **Step 4: Run — expect PASS** (2 passed). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_eval_judge.py -q && ruff check src/faraday/eval/judge.py tests/test_eval_judge.py"
git add src/faraday/eval/judge.py tests/test_eval_judge.py
git commit -m "feat(m4b): Claude-as-judge (Protocol + AnthropicJudge + prompt builder)"
```

---

### Task 5: `runner.py` — record raw outputs, resumable (TDD)

**Files:**
- Create: `src/faraday/eval/runner.py`
- Test: `tests/test_eval_runner.py`

Pure/testable parts: building a raw record from an `Answer`, JSONL append, and resumable `done_keys`. The imperative `run()` (ingest per chunk-size, build `RagEngine` per top_k, loop) is exercised by Plan 2's Pi integration.

- [ ] **Step 1: Write the failing test.** Create `tests/test_eval_runner.py`:

```python
from faraday.eval.config import AblationConfig
from faraday.eval.runner import append_record, done_keys, record_from_answer
from faraday.models import Answer, Chunk, RetrievedChunk


def _answer():
    rc = RetrievedChunk(chunk=Chunk(doc_id="d", ord=3, text="...", source="moon.txt"), score=0.9)
    return Answer(text="July 1969 [1].", sources=[rc], cited_indices=[1], invalid_citations=[])


def test_record_from_answer_shape():
    cfg = AblationConfig(top_k=4, chunk_size=1200, chunk_overlap=200)
    rec = record_from_answer(cfg, "q1", _answer())
    assert rec["qid"] == "q1"
    assert rec["config"] == {"top_k": 4, "chunk_size": 1200, "chunk_overlap": 200}
    assert rec["retrieved"] == [{"source": "moon.txt", "ord": 3}]
    assert rec["cited"] == [1] and rec["invalid"] == [] and rec["abstained"] is False


def test_append_and_done_keys_roundtrip(tmp_path):
    cfg = AblationConfig(top_k=2, chunk_size=600, chunk_overlap=100)
    path = tmp_path / "raw.jsonl"
    append_record(path, record_from_answer(cfg, "q1", _answer()))
    append_record(path, record_from_answer(cfg, "q2", _answer()))
    assert done_keys(path) == {(cfg.slug, "q1"), (cfg.slug, "q2")}
    assert done_keys(tmp_path / "missing.jsonl") == set()
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement.** Create `src/faraday/eval/runner.py`:

```python
"""On-Pi eval runner: drive the real RagEngine per ablation config and record raw
outputs as JSONL. Resumable (skip done (config, qid)). The pure record/IO helpers
unit-test off-Pi; run() is exercised by the Pi integration (Plan 2).
"""
from __future__ import annotations

import json
from pathlib import Path

from faraday.eval.config import AblationConfig
from faraday.eval.metrics import is_abstention
from faraday.models import Answer


def record_from_answer(cfg: AblationConfig, qid: str, answer: Answer) -> dict:
    return {
        "config": {"top_k": cfg.top_k, "chunk_size": cfg.chunk_size,
                   "chunk_overlap": cfg.chunk_overlap},
        "slug": cfg.slug,
        "qid": qid,
        "retrieved": [{"source": rc.chunk.source, "ord": rc.chunk.ord}
                      for rc in answer.sources],
        "answer": answer.text,
        "cited": list(answer.cited_indices),
        "invalid": list(answer.invalid_citations),
        "abstained": is_abstention(answer.text),
    }


def append_record(path: Path, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as f:
        f.write(json.dumps(record) + "\n")


def done_keys(path: Path) -> set[tuple[str, str]]:
    """(slug, qid) pairs already recorded, so a re-run skips them."""
    p = Path(path)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            out.add((d["slug"], d["qid"]))
    return out
```

- [ ] **Step 4: Run — expect PASS** (2 passed). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_eval_runner.py -q && ruff check src/faraday/eval/runner.py tests/test_eval_runner.py"
git add src/faraday/eval/runner.py tests/test_eval_runner.py
git commit -m "feat(m4b): eval runner record/append/resumable core"
```

---

### Task 6: `report.py` — aggregate + scorecard + ablation plot (TDD)

**Files:**
- Create: `src/faraday/eval/report.py`
- Test: `tests/test_eval_report.py`

Combines deterministic metrics (per config) with judge scores into a scorecard + an ablation plot. Judge is injected (fake in tests).

- [ ] **Step 1: Write the failing test.** Create `tests/test_eval_report.py`:

```python
from faraday.eval.dataset import EvalItem
from faraday.eval.judge import JudgeVerdict
from faraday.eval.report import judge_rows, make_scorecard, render_ablation


class FakeJudge:
    def score(self, **kwargs):
        return JudgeVerdict(faithfulness=5, correctness=4, rationale="ok")


def _item(qid, answerable):
    return EvalItem(qid, "?", answerable, "moon.txt" if answerable else "",
                    (100, 260) if answerable else (0, 0), "ref")


def _rows(slug):
    return [
        {"slug": slug, "qid": "q1", "retrieved": [{"source": "moon.txt", "ord": 0}],
         "answer": "July 1969 [1].", "cited": [1], "invalid": [], "abstained": False},
    ]


def test_judge_rows_scores_answered_questions():
    items = {"q1": _item("q1", True)}
    scores = judge_rows(_rows("k2_c200_o0"), items, FakeJudge())
    assert scores["q1"].faithfulness == 5 and scores["q1"].correctness == 4


def test_make_scorecard_has_a_row_per_config():
    items = {"q1": _item("q1", True)}
    per_config = {
        "k2_c200_o0": {"recall_at_k": 1.0, "mrr": 1.0, "citation_validity": 1.0,
                       "abstention_accuracy": 1.0, "faithfulness": 5.0, "correctness": 4.0},
    }
    md = make_scorecard(per_config)
    assert "k2_c200_o0" in md and "recall@k" in md.lower()


def test_render_ablation_writes_png(tmp_path):
    per_config = {
        "k2_c600_o100": {"recall_at_k": 0.8, "faithfulness": 4.0},
        "k4_c600_o100": {"recall_at_k": 0.9, "faithfulness": 4.5},
    }
    out = tmp_path / "ablations.png"
    render_ablation(per_config, out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement.** Create `src/faraday/eval/report.py`:

```python
"""Aggregate raw rows + deterministic metrics + judge scores into a scorecard and
an ablation plot. Judge is injected (fake in tests; AnthropicJudge live).
"""
from __future__ import annotations

from pathlib import Path

from faraday.eval.dataset import EvalItem
from faraday.eval.judge import Judge, JudgeVerdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


def _context_of(row: dict) -> str:
    # Re-derive a context string from retrieved chunk sources (text isn't recorded;
    # the judge grades faithfulness against the cited sources by reference).
    return "\n".join(f"[{i}] (source: {c['source']})" for i, c in enumerate(row["retrieved"], 1))


def judge_rows(rows: list[dict], items_by_id: dict[str, EvalItem],
               judge: Judge) -> dict[str, JudgeVerdict]:
    """Score each ANSWERED (non-abstained, answerable) row's answer quality."""
    out: dict[str, JudgeVerdict] = {}
    for r in rows:
        item = items_by_id[r["qid"]]
        if not item.answerable or r["abstained"]:
            continue
        out[r["qid"]] = judge.score(
            question=item.question, reference_answer=item.reference_answer,
            context=_context_of(r), answer=r["answer"],
        )
    return out


def make_scorecard(per_config: dict[str, dict]) -> str:
    """Markdown table, one row per config, columns = the metric keys present."""
    cols = ["recall_at_k", "mrr", "citation_validity", "abstention_accuracy",
            "faithfulness", "correctness"]
    header = "| config | " + " | ".join(c.replace("_at_k", "@k") for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = ["# Faraday M4b — RAG Eval Scorecard", "", header, sep]
    for slug in sorted(per_config):
        m = per_config[slug]
        cells = [f"{m.get(c, float('nan')):.3f}" for c in cols]
        lines.append(f"| {slug} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_ablation(per_config: dict[str, dict], out_path: Path,
                    metric: str = "recall_at_k") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slugs = sorted(per_config)
    values = [per_config[s].get(metric, float("nan")) for s in slugs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(slugs)), values)
    ax.set_xticks(range(len(slugs)))
    ax.set_xticklabels(slugs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title(f"Faraday M4b — {metric} by ablation config")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run — expect PASS** (3 passed). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_eval_report.py -q && ruff check src/faraday/eval/report.py tests/test_eval_report.py"
git add src/faraday/eval/report.py tests/test_eval_report.py
git commit -m "feat(m4b): report aggregator (scorecard + ablation plot)"
```

---

### Task 7: Full-suite regression + lint

**Files:** none (verification only)

- [ ] **Step 1: Whole suite on the Pi.** Run:

`ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest -q"`
Expected: all prior tests + the ~12 new eval-engine tests pass; integration still deselected.

- [ ] **Step 2: Full lint.** Run:

`ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && ruff check src tests"`
Expected: `All checks passed!`

No commit (nothing changed). Fix any failure in the owning task's file.

---

## Self-Review

**1. Spec coverage** (engine portion of the spec):

| Spec section | Task |
|---|---|
| §5 components: config / dataset / metrics / judge / runner / report | Tasks 1 / 2 / 3 / 4 / 5 / 6 |
| §6 EvalItem schema (id, question, relevant_source span, reference, answerable) | Task 2 (`EvalItem`, `load_golden`) |
| §7 metrics: recall@k (span overlap), MRR, citation validity, abstention | Task 3 (`aggregate`, `chunk_is_relevant`, `is_abstention`) |
| §7 answer quality (judge faithfulness + correctness) | Task 4 (`AnthropicJudge`) + Task 6 (`judge_rows`) |
| §2/§7 source-span labels, chunk-size-invariant | Task 3 recomputes span from `ord`+`(size,overlap)` — no chunk-level labels |
| §4/§9 record-then-judge + resumability | Task 5 (`record_from_answer`, `append_record`, `done_keys`) |
| §11 testing (metrics/dataset/judge-with-fake/report/runner), Protocol-DI | Tasks 2–6 with fakes |
| §5 components corpus/generate, §8 run, §13 scorecard/findings | **Plan 2** (out of scope here, by design) |

**Gaps (intentional, → Plan 2):** corpus build, `generate.py` (Claude golden draft), `golden.jsonl` curation, `scripts/80_run_evals.sh`, the imperative `runner.run()` (ingest-per-config + RagEngine loop), the live Pi run, `report.main()` wiring + judge-output freezing, `findings.md`. Each needs the live corpus/Pi, so they belong with the data+run plan.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases". Every code step is complete. The `*.jsonl` git-ignore decision is explicitly deferred to Plan 2 (where raw/golden paths are committed/ignored).

**3. Type consistency:** `AblationConfig(top_k, chunk_size, chunk_overlap)` + `.slug` consistent (config/runner/report/tests). `EvalItem(id, question, answerable, relevant_doc, relevant_span, reference_answer)` consistent (dataset/metrics/report/tests). Raw row dict keys (`slug, qid, retrieved:[{source,ord}], answer, cited, invalid, abstained`) consistent (runner ↔ metrics ↔ report ↔ tests). `JudgeVerdict(faithfulness, correctness, rationale)` + `Judge.score(*, question, reference_answer, context, answer)` consistent (judge ↔ report ↔ FakeJudge). `aggregate(rows, items_by_id, size, overlap)` signature matches its call sites.

**Verdict:** engine plan is complete, spec-covering (engine scope), placeholder-free.
