# Faraday M4b — RAG Evaluation Findings

**Status:** ✅ **COMPLETE — full 9-cell grid** (top_k {2, 4, 8} × chunk_size {600, 1200,
2400} chars, overlap = size⁄6), 47 golden questions per cell = **423 recorded answers**,
on the appliance stack (Qwen2.5-1.5B-Q4_K_M gen + bge-small-en-v1.5 embed) on a 4 GB Pi 4,
`get_throttled=0x0` throughout. Deterministic retrieval metrics cover all 9 cells;
Claude-as-judge answer scoring runs at the **baseline cell only** (`k4_c1200_o200`) for
cost control. Run record + the five crashes that shaped it: §6.

**Deliverables:** [`scorecard.md`](./scorecard.md) · [`ablations.png`](./ablations.png) ·
[`golden.jsonl`](./golden.jsonl) (47-item gold set: 41 answerable with machine-verified
source spans + 6 out-of-corpus). Raw rows and frozen judge caches are gitignored
(regenerable; large).

---

## 1. Headline & recommendation

On the Apollo corpus, the retrieval frontier says: **chunk at ~1200 chars, retrieve
top_k=4** — the appliance baseline `k4_c1200_o200`, validated by the grid. Two knobs, two
clean stories:

| Knob | Finding | Pick |
|---|---|---|
| **chunk_size** | c1200 wins recall at every top_k; c600 fragments, **c2400 is self-defeating** | **1200** |
| **top_k** | recall rises monotonically but the knee is k4 (k2→k4 **+0.146**, k4→k8 **+0.049**) | **4** (8 for max recall) |

Baseline retrieval: **recall@k 0.805, MRR 0.628**. The retrieval ceiling is `k8_c1200`
(recall **0.854**) — +5 recall points for 2× the prefill cost, and prefill is this board's
bottleneck (M4a §3), so **k4 is the interactive knee and k8 the batch option**. Answer
quality at baseline (Claude `claude-opus-4-8` judge): **faithfulness 4.22/5, correctness
3.95/5**. The model's real weakness is **abstention** — it correctly declines only **2 of
6** out-of-corpus questions (§5).

## 2. Retrieval across the grid

**recall@k** — fraction of the 41 answerable questions with ≥1 relevant chunk in top_k.
Relevance = the chunk's char-span overlaps the labeled gold source-span, recomputed from
the chunker's geometry, so it is **chunk-size-invariant by construction** (a c600 hit and a
c2400 hit are scored against the same gold span):

| | c600 | **c1200** | c2400 |
|---|---|---|---|
| **k=2** | 0.634 | **0.659** | 0.610 |
| **k=4** | 0.756 | **0.805** | 0.659 |
| **k=8** | 0.854 | **0.854** | 0.756 |

**MRR** — reciprocal rank of the first relevant chunk:

| | c600 | c1200 | c2400 |
|---|---|---|---|
| **k=2** | 0.573 | 0.585 | 0.549 |
| **k=4** | 0.610 | 0.628 | 0.563 |
| **k=8** | 0.626 | 0.636 | 0.577 |

**top_k helps, with sharp diminishing returns.** At c1200, recall climbs 0.659 → 0.805 →
0.854; the k2→k4 step (+0.146) is **3×** the k4→k8 step (+0.049). MRR barely moves with k
(0.585 → 0.628 → 0.636) — more chunks raise the *chance* of a hit but don't improve the
*first* hit's rank, i.e. extra retrievals add depth, not a better top result. **Knee = k4.**

**c1200 is the chunk-size sweet spot** — it wins or ties every top_k row. c600 trails
(smaller chunks split the answer span and hand the LLM less context per hit); c2400 trails
harder, for a reason worth its own section.

## 3. The c2400 collapse: a chunk size past the embedder's context is self-defeating

The sharpest result in the study. **c2400 has the *worst* recall at every top_k — despite
its chunks being the biggest "nets."** It loses to c1200 by 0.049 / 0.146 / 0.098
(k2/k4/k8) and even loses to the *smallest* chunks, c600, at k4 (0.659 vs 0.756) and k8
(0.756 vs 0.854). "Biggest chunks, worst recall" has exactly one explanation, and we
measured it directly:

- **bge-small-en-v1.5 caps at 512 tokens.** Tokenizing all 409 c2400 chunks against the
  live embed model: **max 718 tokens, mean 513, and 52% exceed 512.** Over half the c2400
  chunks cannot be embedded whole.
- The embedder therefore vectorizes only each chunk's **first ~1450 chars** (the truncation
  the run was forced to add — §6, crash #5). A relevant span sitting in a chunk's *tail* is
  invisible to retrieval — even though the relevance metric, scored on the full chunk
  geometry, *would* have counted that chunk as a hit if it had been retrieved.

So the bigger net is full of holes: c2400's span-overlap advantage is more than cancelled
by its embeddings ignoring up to half of each chunk. **Chunk size must not exceed what the
embedder can encode** (~512 tokens ≈ ~1800 chars for bge). The appliance's 1200-char
default (~280 tokens) sits comfortably inside that ceiling; 2400 sits well past it. This is
the load-bearing lesson of M4b: the chunker and the embedder share a budget, and the
embedder sets it.

## 4. Answer quality at the baseline (the judged cell)

Claude scored the 41 answerable answers at `k4_c1200_o200` (1–5 scales):

- **faithfulness = 4.22 / 5** — answers stay grounded in the retrieved context; little
  free-floating fabrication on in-corpus questions.
- **correctness = 3.95 / 5** — and they're right ~4/5 of the time versus the reference.
  Solid for a 1.5B Q4_K_M model doing extractive QA on a dense, overlapping corpus.
- **citation_validity = 1.000 (all 9 cells)** — but read this narrowly. It means *no
  out-of-range citation indices*, **not** that the citation supports the claim. The
  counter-example is in §5 (`_005`): the model attached a citation `[2]` to a hallucinated
  answer whose retrieved chunk says nothing about it. Faithfulness (4.22) is the honest
  grounding measure; citation_validity is a cheaper structural check that can be passed by a
  wrong answer.

## 5. Abstention: the model over-answers (the trust gap)

The corpus is Apollo-only; 6 gold questions are deliberately out-of-corpus, where the
correct behavior is to **abstain**. By the judge (the reliable classifier), the model
abstains correctly on **only 2 of 6**:

| qid | question | model behavior |
|---|---|---|
| `_000` | Hubble launch year | ✅ abstained — "…not included in the sources" |
| `_004` | first SpaceX crewed mission | ✅ abstained — "I cannot provide a specific name" |
| `_001` | first American woman in space | ❌ answered "Sally Ride, STS-7" (right fact, wrong behavior) |
| `_002` | first ISS module | ❌ answered "Zvezda… June 1998" (also factually wrong) |
| `_003` | how many have walked on Mars | ❌ "only two people… the Viking landers" (egregiously wrong) |
| `_005` | who developed Falcon 9 | ❌ "SpaceX… [2]" — hallucination **with a spurious citation** |

**Two metrics, one conclusion.** The phrase-heuristic abstention score (0.872 at baseline)
and the judge cross-check (0.915) differ only because the heuristic's fixed phrase-list
misses `_000`/`_004` — the model declined in words ("not included in the sources", "I
cannot provide a specific name") that aren't in the list, a **false-negative** the semantic
judge corrects. But both agree on the substance: **the 1.5B model rarely refuses.** It will
confidently answer — sometimes wrongly (`_002`, `_003`), sometimes with an ungrounded
citation (`_005`) — questions it has no corpus support for. (Heuristic abstention_accuracy
is a near-flat 0.85–0.89 across the whole grid because it is dominated by the 41 answerable
questions the model answers correctly; abstention is a 6-question signal, best read from the
judged baseline, not the aggregate.)

This is the trustworthiness gap that motivates the **GBNF-grounded, citation-required
generation deferred to M5**: constrain the model to cite-or-abstain at decode time rather
than trusting it to volunteer a refusal.

## 6. Run record (and the five crashes)

- **Grid:** 9 cells × 47 questions = **423 rows**, each a record-then-judge JSONL row (the
  answer, the retrieved chunks *with text*, citations, abstention flag). Recorded on the Pi;
  scored off the raw rows on the dev machine. `get_throttled=0x0` start-to-finish.
- **Resumability earned its keep.** The run survived **five crashes and three resumes** (two
  of them deliberate power-offs while the operator was away) with **zero dropped or
  duplicated rows** — every one of the 9 files holds exactly 47, because `done_keys()` skips
  already-recorded (config, qid) pairs on restart.
- **The five crashes**, each a distinct root cause (the first four shared a `ReadTimeout`
  symptom — the lesson is to read the *server's* log, not just the client's):
  1. unbounded per-document embed batches → batch 16 texts/POST (`78c7574`).
  2. k8_c2400 prompts exceed gen `-c 4096` + an interactive-sized LLM timeout → `GEN_CTX=8192`
     + 1800 s eval timeout (`5395037`).
  3. deep-context prefill is legitimately slow (6.75 tok/s, ≈13 min/q) → timeout sized to
     measurement, not vibes (`348e278`).
  4. the llama-server prompt cache (default **8192 MiB** ceiling, larger than the board) grew
     to 2.4 GB and evicted the mmap'd weights → decode collapsed 3.6 → 0.07 tok/s →
     `--no-cache-prompt --cache-ram 512` (`0260b76`).
  5. a c2400 chunk over bge's 512-token limit → a hard embed **HTTP 500**, *not* a timeout →
     clip embed inputs to a **measured** 1450 chars (`ba5879a` → `cda2f8f`; §3).
- **k8_c2400 is the long pole:** ~4.6k-token prompts at ~6 tok/s prefill ≈ 13 min/row,
  ~92% prefill. Pricing a long-context RAG batch by *prompt tokens* — not per question — is
  the only estimate that held (M4a §3).

## 7. Process notes (engineering candor)

- **Measure the whole distribution, not one sample.** Crash #5's first fix sized the
  embed-truncation budget from a *single* chunk (4.3 chars/tok → 1800 chars) and crashed
  again on a denser one (514 tokens). The run dies on the *first* over-limit chunk, so each
  guess reveals only one — tokenizing all 409 chunks (max 718, 52% over 512) was the only
  way to set a budget that holds (1450 chars → ≤490 tokens, 22-token margin, measured).
- **Truncating the embedding ≠ truncating the context.** The 1450-char clip applies only to
  what bge *vectorizes*; the LLM still receives the full chunk text. That separation is what
  keeps c2400 a legitimate (if losing) ablation point rather than a silent c1200 clone — and
  it's why the c2400 *generation* prompts stayed ~4.6k tokens even after the fix.
- **A reboot-resume pays a cold-cache tax.** After a power-cycle the re-ingest re-embeds
  ~1.3k chunks off a cold page cache — ~64 min vs ~20 warm — long enough to trip a 45-min
  stall alarm mid-re-ingest. Expected, not a fault; the alarm is sized for the ~13-min
  generation cadence, not the one-time cold ingest.
- **`citation_validity = 1.0` flattered the system** until `_005` showed a valid-index
  citation on an ungrounded claim (§4–5). Structural citation validity and semantic
  faithfulness are different guarantees — report both, and never let the cheap one stand in
  for the dear one.
