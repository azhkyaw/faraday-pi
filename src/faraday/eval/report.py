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
