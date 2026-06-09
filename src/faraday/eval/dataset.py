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
