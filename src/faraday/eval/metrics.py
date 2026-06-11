"""Deterministic RAG metrics over recorded raw rows. Pure — no Pi, no network.

Retrieved rows carry (source, ord, text); relevance uses only (source, ord) — the
chunk's char span is recomputed from the chunker's geometry: step = size - overlap,
start = ord*step, end = start+size. (The text is for the judge, not for relevance.)
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
