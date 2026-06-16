"""Render the optimization CSV into deliverables: per-lever gains, the baseline->best
waterfall, the TTFT/decode-vs-context curve, and a leaderboard. Pure helpers tested;
rendering is a smoke test. Reuses the headless-Agg pattern from M4a's plot.py.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


def load_rows(csv_path: Path) -> list[dict]:
    with Path(csv_path).open(newline="") as f:
        return list(csv.DictReader(f))


def _by_label(rows: list[dict]) -> dict[str, dict]:
    return {r["label"]: r for r in rows}


def lever_gains(rows: list[dict]) -> dict[str, float]:
    """label -> percent decode gain over baseline, for the single-lever cells."""
    base = float(_by_label(rows)["baseline"]["decode_tps"])
    out: dict[str, float] = {}
    for r in rows:
        if r["component"] in ("governor", "threads", "batch", "kvquant", "flashattn") \
                and r["decode_tps"]:
            out[r["label"]] = round((float(r["decode_tps"]) - base) / base * 100, 4)
    return out


def make_leaderboard(rows: list[dict]) -> str:
    ranked = sorted((r for r in rows if r["decode_tps"]),
                    key=lambda r: float(r["decode_tps"]), reverse=True)
    lines = ["# Faraday M4c — Optimization Leaderboard", "",
             "Sorted by decode tok/s (higher = better).", "",
             "| Rank | Cell | decode t/s | prefill t/s | accept % | throttled |",
             "|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        lines.append(f"| {i} | {r['label']} | {r['decode_tps']} | {r['prefill_tps']} | "
                     f"{r['accept_rate'] or '-'} | {r['throttled']} |")
    return "\n".join(lines) + "\n"


def render_waterfall(rows: list[dict], out_path: Path) -> None:
    bl = _by_label(rows)
    labels, values = ["baseline"], [float(bl["baseline"]["decode_tps"])]
    for r in rows:
        if r["component"] == "stacked_best" and r["decode_tps"]:
            labels.append("stacked_best")
            values.append(float(r["decode_tps"]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), values, color=["#888", "#2a9d8f"][:len(labels)])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("decode tok/s")
    ax.set_title("Faraday M4c — baseline → best-tuned decode throughput")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_context_curve(rows: list[dict], out_path: Path) -> None:
    pts = sorted((int(r["label"].split("=")[1]), float(r["prefill_tps"] or "nan"),
                  float(r["decode_tps"] or "nan"))
                 for r in rows if r["component"] == "context")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if pts:
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], marker="o", label="prefill t/s")
        ax.plot(xs, [p[2] for p in pts], marker="s", label="decode t/s")
        ax.legend()
    ax.set_xlabel("context size (prompt tokens)")
    ax.set_ylabel("tok/s")
    ax.set_title("Faraday M4c — throughput vs context length")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_lever_bars(rows: list[dict], out_path: Path) -> None:
    gains = lever_gains(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(gains)
    ax.bar(range(len(labels)), [gains[k] for k in labels])
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("decode gain vs baseline (%)")
    ax.set_title("Faraday M4c — per-lever marginal gain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    from faraday.bench.optimize_config import CSV_PATH, RESULTS_DIR
    rows = load_rows(CSV_PATH)
    render_waterfall(rows, RESULTS_DIR / "waterfall.png")
    render_lever_bars(rows, RESULTS_DIR / "lever_gains.png")
    render_context_curve(rows, RESULTS_DIR / "context_curve.png")
    (RESULTS_DIR / "leaderboard.md").write_text(make_leaderboard(rows))
    print(f"wrote waterfall/lever_gains/context_curve + leaderboard ({len(rows)} rows)")


if __name__ == "__main__":
    main()
