"""Aggregate raw rows + deterministic metrics + judge scores into a scorecard and
an ablation plot. Judge is injected (fake in tests; AnthropicJudge live).
"""
from __future__ import annotations

import json
from pathlib import Path

from faraday.eval.dataset import EvalItem
from faraday.eval.judge import Judge, JudgeVerdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


def _context_of(row: dict) -> str:
    # Chunk text is recorded in raw rows so the judge grades faithfulness against
    # what the model actually saw (old text-less rows degrade to source-only).
    return "\n".join(
        f"[{i}] (source: {c['source']}) {c.get('text', '')}".rstrip()
        for i, c in enumerate(row["retrieved"], 1)
    )


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


def load_or_classify_abstentions(rows: list[dict], items_by_id: dict[str, EvalItem],
                                 judge: Judge, cache_path: Path) -> dict[str, bool]:
    """Judge-classify whether each row's answer abstained (spec §10: the phrasing
    heuristic alone isn't trusted). Frozen to cache_path like load_or_score."""
    if Path(cache_path).exists():
        out: dict[str, bool] = {}
        for line in Path(cache_path).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["qid"]] = bool(d["abstained"])
        return out
    checks = {
        r["qid"]: judge.classify_abstention(
            question=items_by_id[r["qid"]].question, answer=r["answer"])
        for r in rows
    }
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(cache_path).open("w") as f:
        for qid, abstained in checks.items():
            f.write(json.dumps({"qid": qid, "abstained": abstained}) + "\n")
    return checks


def abstention_cross_check(rows: list[dict], items_by_id: dict[str, EvalItem],
                           judged: dict[str, bool]) -> dict:
    """Judge-based abstention accuracy + the qids where heuristic and judge disagree
    (each disagreement is either a heuristic miss or a judge error — review by hand)."""
    n = len(rows)
    acc = (sum(1 for r in rows
               if judged[r["qid"]] == (not items_by_id[r["qid"]].answerable)) / n
           if n else 0.0)
    disagreements = [r["qid"] for r in rows if judged[r["qid"]] != r["abstained"]]
    return {"abstention_judged": acc, "disagreements": disagreements}


def make_scorecard(per_config: dict[str, dict]) -> str:
    """Markdown table, one row per config, columns = the metric keys present."""
    cols = ["recall_at_k", "mrr", "citation_validity", "abstention_accuracy",
            "abstention_judged", "faithfulness", "correctness"]
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


def main() -> None:
    """Score the recorded run: deterministic metrics over ALL configs + judge answer
    quality at the BASELINE config only (cost control). Writes scorecard + plot."""
    from faraday.eval import config
    from faraday.eval.dataset import load_golden
    from faraday.eval.judge import AnthropicJudge
    from faraday.eval.metrics import aggregate

    items = load_golden(config.GOLDEN_PATH)
    by_id = {it.id: it for it in items}
    per_config: dict[str, dict] = {}
    for cfg in config.configs():
        raw = config.RAW_DIR / f"{cfg.slug}.jsonl"
        if not raw.exists():
            continue
        rows = [json.loads(ln) for ln in raw.read_text().splitlines() if ln.strip()]
        m = aggregate(rows, by_id, size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        if cfg.slug == config.BASELINE.slug:  # judge at baseline only (cost control)
            judge = AnthropicJudge()
            verdicts = load_or_score(rows, by_id, judge,
                                     config.JUDGE_DIR / f"{cfg.slug}.jsonl")
            if verdicts:
                m["faithfulness"] = sum(v.faithfulness for v in verdicts.values()) / len(verdicts)
                m["correctness"] = sum(v.correctness for v in verdicts.values()) / len(verdicts)
            checks = load_or_classify_abstentions(
                rows, by_id, judge, config.JUDGE_DIR / f"{cfg.slug}.abstention.jsonl")
            cc = abstention_cross_check(rows, by_id, checks)
            m["abstention_judged"] = cc["abstention_judged"]
            if cc["disagreements"]:
                print(f"abstention heuristic vs judge disagree on: {cc['disagreements']}")
        per_config[cfg.slug] = m

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (config.EVAL_DIR / "scorecard.md").write_text(make_scorecard(per_config))
    render_ablation(per_config, config.EVAL_DIR / "ablations.png", metric="recall_at_k")
    print(f"wrote scorecard.md + ablations.png for {len(per_config)} configs")


if __name__ == "__main__":
    main()
