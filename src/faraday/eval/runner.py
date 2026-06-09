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
