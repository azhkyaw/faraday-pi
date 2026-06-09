# Faraday M4b — RAG Quality Evals
## Design Document

| | |
|---|---|
| **Status** | Approved (brainstorming) — ready for implementation planning |
| **Date** | 2026-06-10 |
| **Milestone** | M4b (second of three M4 sub-studies; M4a quant sweep · **M4b RAG evals** · M4c optimization) |
| **Builds on** | M0–M3 (RAG core, serving, observability); M4a (the record-then-judge harness pattern) |

---

## 1. Overview

M4b is a **reproducible evaluation harness** that scores Faraday's RAG quality — retrieval,
answer quality, citations, and abstention — over a themed Wikipedia corpus, and runs
**ablations** to show what actually moves the needle. Where M4a measured the *model's
efficiency* (perplexity / speed / footprint), M4b measures the *system's task quality*: does
it retrieve the right context, answer correctly and faithfully, cite accurately, and abstain
when the answer isn't in the corpus?

**Architecture: record-then-judge** (mirrors M4a). The **Pi runs the real `RagEngine`** and
records raw outputs per ablation config; **dev-side** computes deterministic metrics and
Claude-as-judge answer scores; results aggregate into a scorecard + ablation study. The
expensive, non-deterministic step (Pi generation) is run *once* and frozen; the cheap,
re-runnable steps (metrics, judging) read those frozen artifacts.

## 2. Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Scope | **Full RAG science**: retrieval + answer quality + citations + abstention + **ablations** | Portfolio-grade; the complete quality picture |
| Eval-set construction | **Hybrid generate-then-curate** (Claude drafts → human curates → commit) | Scale *and* trust; the real-world synthetic-eval workflow |
| Judge / generator | **Claude API** (Sonnet-class, e.g. `claude-sonnet-4-6`) | Strongest reliable judge; **dev-time only** (the appliance's air-gap is a runtime property, intact); demonstrates the Anthropic API |
| Corpus | **Themed Wikipedia set** (~10–20 articles, committed) | Factual, retrieval-discriminable, big enough for meaningful recall@k + chunk-size ablations |
| Execution | **Record-then-judge** (Pi records raw, dev scores) | Reproducible; separates slow generation from cheap, re-scorable metrics/judging |
| Relevance labels | **Source-span level, not chunk level** | Chunk boundaries change under the chunk-size ablation; span labels keep the golden set **chunk-size-invariant** |

## 3. Goals / non-goals

**Goals**
- A committed **golden set** (~30–50 items incl. out-of-corpus abstention questions) over a committed Wikipedia corpus.
- An **on-Pi, resumable runner** producing raw RAG outputs across the ablation grid.
- **Dev-side** deterministic metrics + **Claude-judge** answer scores (faithfulness, correctness).
- A committed **scorecard + ablation plot + findings**.
- Pure metric/judge-parsing logic **unit-tested with fakes** (no Pi, no network); ruff clean.

**Non-goals (YAGNI; later/never)**
- Fine-tuning or training; an eval UI; external frameworks (RAGAS etc.) — we build focused, legible metrics.
- The **embedding-model ablation is optional/stretch** (a second embed model is heavier).
- **GBNF grammar-constrained citations** (deferred from M2) — evaluated here only if implemented; otherwise an M5 item.
- Online/continuous eval; multi-judge consensus (a single strong judge, with frozen outputs, is the baseline).

## 4. Architecture (the four phases)

```
BUILD  [dev, once]   themed Wikipedia → examples/eval_corpus/*.txt (+ SOURCES.md)
                     generate.py (Claude drafts candidates) → human curate → golden.jsonl  ✔commit
   │
RUN    [Pi]          scripts/80_run_evals.sh → for each ablation config:
                       (re)ingest corpus at that chunk-size/embedding
                       for each eval question:  RagEngine.answer(top_k)
                         → record raw row {retrieved chunk spans, answer, citations, abstained?}
                       → results/evals/raw_<config>.jsonl   (resumable: skip done config×question)
   │
SCORE  [dev]         metrics.py  : recall@k, MRR, citation validity, abstention  (pure, no network)
                     judge.py    : Claude scores faithfulness + correctness on the RECORDED answers
                                   (outputs frozen to disk → re-score without re-calling the API)
   │
REPORT [dev]         report.py → results/evals/{scorecard.md, ablations.png, findings.md}  ✔commit
```

Raw Pi outputs are **born on the Pi, scored and committed from the dev machine** (the M4a
reverse-direction workflow). The RAG system under test is the **real `RagEngine`** (real
`Retriever` → sqlite-vec, real `LLMClient` → llama-server) — faithful to the appliance; only
the HTTP/SSE transport (M2) is out of scope, which doesn't affect RAG *quality*.

## 5. Components (`src/faraday/eval/`, mirroring `bench/`)

| Unit | Path | Responsibility |
|---|---|---|
| Config | `eval/config.py` | ablation grid (top_k × chunk-size [× embedding]), paths, schemas |
| Dataset | `eval/dataset.py` | `EvalItem` schema + `golden.jsonl` loader/validator |
| Generator | `eval/generate.py` | **dev/one-time**: Claude drafts candidate eval items from the corpus → for human curation |
| Metrics | `eval/metrics.py` | **pure fns**: `recall_at_k`, `mrr`, `citation_validity` (reuses `classify_citations`), `abstention_accuracy`. The unit-tested core |
| Judge | `eval/judge.py` | Claude judge: answer **faithfulness** + **correctness** (+ citation faithfulness), structured output; **injected client** (fake in tests) |
| Runner | `eval/runner.py` | **on-Pi**: per config × question → `RagEngine.answer` → record raw row; resumable |
| Report | `eval/report.py` | aggregate raw + labels + judge → `scorecard.md` + ablation plot + tables |
| On-Pi runner | `scripts/80_run_evals.sh` | ingest corpus + run the eval runner over configs |
| Corpus | `examples/eval_corpus/*.txt` + `SOURCES.md` | committed Wikipedia articles + attribution |
| Results | `results/evals/` | `golden.jsonl`, `scorecard.md`, `ablations.png`, `findings.md` (committed); `raw_*.jsonl` (git-ignored) |

## 6. The eval set (schema + construction)

`results/evals/golden.jsonl`, one JSON object per line:
```json
{ "id": "q001",
  "question": "…",
  "relevant_source": { "doc_id": "wiki_<article>", "span": [start_char, end_char] },
  "reference_answer": "…",
  "answerable": true }
```
- **~30–50 items**, ~20% `answerable:false` (out-of-corpus) for abstention.
- `relevant_source.span` is a char range in the source doc — **chunk-size-invariant** (§2).
- **Construction:** `generate.py` feeds corpus passages to Claude → drafts `{question,
  reference_answer, supporting span}` candidates → **human curates** (prune/fix) → commit. The
  generator runs once; the committed golden set is the durable artifact the eval depends on.

## 7. Metrics (precise definitions)

- **recall@k** — fraction of *answerable* questions where ≥1 of the top-k retrieved chunks
  overlaps the labeled `relevant_source.span`. "Overlap" = the chunk's char range in the
  source doc intersects the labeled span by **≥1 char** (presence-of-overlap, not a coverage
  threshold) — simple and chunk-size-invariant; a stricter coverage ratio is a deferred refinement.
- **MRR** — mean reciprocal rank of the first chunk overlapping the relevant span.
- **citation validity** — fraction of `[n]` citations that are in-range (deterministic; reuses
  `classify_citations` / `Answer.cited_indices` & `invalid_citations`).
- **citation faithfulness** — does the cited source actually support the sentence? (judge).
- **abstention accuracy** — on `answerable:false` questions, fraction correctly abstained;
  and on `answerable:true`, fraction that did *not* wrongly abstain. Abstention detected by
  heuristic (phrasing) cross-checked by the judge.
- **answer quality** (judge) — **faithfulness** (grounded in retrieved sources, no
  hallucination) + **correctness** (matches `reference_answer`), each scored with a rationale.

## 8. Ablations

Core grid: **`top_k ∈ {2, 4, 8}` × chunk-size `∈ {256, 512, 1024}`** (cheap, high-signal).
Each config = a full pass over the golden set. **Embedding-model swap is an optional stretch
axis** (needs a second embed model). The report shows which knob moves recall@k and answer
quality. The grid and the ~30–50-item set are deliberately sized to keep this an
**overnight-ish** run, not multi-day (Pi generation dominates runtime).

## 9. Data flow & resumability

`runner.py` records one raw row per `(config, question)` to `raw_<config>.jsonl`; on restart
it reads what's done and skips it — so a reboot/interruption loses at most one question.
Deterministic metrics and judging read these frozen raw files, so **re-scoring never re-runs
Pi generation**, and re-judging (rubric tweak, second judge) never re-calls generation.

## 10. Error handling

- **Claude API:** injected client with retry/timeout/backoff; **judge outputs frozen to disk**
  (`judge_<config>.jsonl`) — re-runs read the cache unless forced.
- **Resumable runner** (skip done `(config, question)`).
- **Abstention** detected by heuristic phrasing + judge cross-check (neither alone is trusted).
- **Malformed golden items** skipped with a logged warning; **judge parse failures** recorded
  as `error` (not silently scored), surfaced in the report.

## 11. Testing

- **Unit (dev machine, no Pi, no network):** `metrics.py` against tiny hand-labeled fixtures
  (known recall@k/MRR/abstention); `dataset.py` loader/validator; `judge.py` with a **fake
  Claude client** asserting the prompt is well-formed and the structured response is parsed
  into scores; `report.py` smoke (sample → scorecard + PNG exists).
- **Integration (Pi):** a **1-question end-to-end** through the real `RagEngine` records a
  valid raw row.
- TDD throughout; the Claude client is a **`Protocol`** (real impl injected at the edge, fake
  in tests) — the project's DI convention. Author on Windows → `git push pi` → `pytest` on the Pi.

## 12. File structure (delta)

```
src/faraday/eval/
  __init__.py  config.py  dataset.py  generate.py  metrics.py  judge.py  runner.py  report.py
tests/
  test_eval_metrics.py  test_eval_dataset.py  test_eval_judge.py  test_eval_report.py
  test_eval_runner.py   test_eval_integration.py
scripts/80_run_evals.sh
examples/eval_corpus/*.txt + SOURCES.md            # committed Wikipedia + attribution
results/evals/golden.jsonl                          # committed golden set
results/evals/{scorecard.md, ablations.png, findings.md}   # committed (post-run)
results/evals/raw_*.jsonl, judge_*.jsonl            # git-ignored (frozen artifacts)
pyproject.toml                                      # + anthropic (an `eval` extra)
```

## 13. Definition of done

- Golden set (~30–50 items incl. abstention) + Wikipedia corpus + `SOURCES.md` committed.
- The runner produces raw results across the ablation grid on the Pi, resumably.
- Deterministic metrics + Claude-judge answer scores computed dev-side.
- `scorecard.md` (per-metric + headline), `ablations.png`, and `findings.md` committed.
- Pure metric/judge-parsing unit tests green (fakes; no network); ruff clean.

## 14. Runtime note

Pi generation dominates (minutes/question × ~30–50 questions × grid). The grid and set sizes
target an overnight run. Metrics + judging are fast and **re-runnable on frozen artifacts**, so
iterating on the rubric or fixing a metric costs seconds, not another Pi pass.

## 15. Deferred / open

- **Embedding-model ablation** — optional stretch (second embed model).
- **GBNF grammar-constrained citations** — evaluate here if built, else M5.
- **Second-judge validation** — a cheaper judge cross-checking the primary, for judge-bias
  candor — a stretch.
- **Exact judge model ID + structured-output (tool-use) call shape** — confirmed against the
  `claude-api` skill at implementation-plan time.
