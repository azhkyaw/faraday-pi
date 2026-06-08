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
