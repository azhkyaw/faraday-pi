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
