"""Render the sweep CSV into the deliverables: the quality-vs-footprint frontier
PNG and a markdown leaderboard. Pure helpers (pareto_front, make_leaderboard) are
unit-tested; rendering is a smoke test (file exists, non-empty).
"""
from __future__ import annotations

import csv
from pathlib import Path

from faraday.bench import config

import matplotlib

matplotlib.use("Agg")  # headless: no display on the Pi
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _ok_points(rows: list[dict]) -> list[tuple[int, float, str]]:
    """(rss_bytes, ppl, label) for cells that produced both numbers."""
    pts = []
    for r in rows:
        if r["status"] != "ok" or not r["peak_rss_bytes"] or not r["perplexity"]:
            continue
        pts.append((int(r["peak_rss_bytes"]), float(r["perplexity"]),
                    f'{r["size"]}-{r["quant"]}'))
    return pts


def pareto_front(points: list[tuple[float, float]]) -> list[bool]:
    """Mark non-dominated points for (x=rss, y=ppl), minimizing BOTH.
    A point is dominated if another has x<= and y<= with at least one strictly <."""
    flags = []
    for i, (xi, yi) in enumerate(points):
        dominated = any(
            (xj <= xi and yj <= yi) and (xj < xi or yj < yi)
            for j, (xj, yj) in enumerate(points) if j != i
        )
        flags.append(not dominated)
    return flags


def make_leaderboard(rows: list[dict]) -> str:
    pts = _ok_points(rows)
    front = pareto_front([(x, y) for x, y, _ in pts]) if pts else []
    ranked = sorted(zip(pts, front), key=lambda t: t[0][1])  # by ppl ascending
    lines = [
        "# Faraday M4a — Pi-4 Quantization Leaderboard",
        "",
        "Sorted by perplexity (lower = better). "
        "★ = on the quality/footprint Pareto frontier.",
        "",
        "| Rank | Cell | Perplexity | Peak RSS (MB) | ★ |",
        "|---|---|---|---|---|",
    ]
    for i, ((rss, ppl, label), on_front) in enumerate(ranked, 1):
        star = "★" if on_front else ""
        lines.append(f"| {i} | {label} | {ppl:.4f} | {rss / 1024 / 1024:.0f} | {star} |")
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        lines += ["", "**Did not complete:**", ""]
        for r in bad:
            note = f": {r['notes']}" if r["notes"] else ""
            lines.append(f"- `{r['size']}-{r['quant']}` — {r['status']}{note}")
    return "\n".join(lines) + "\n"


def render_frontier(rows: list[dict], out_path: Path) -> None:
    pts = _ok_points(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    if pts:
        for x, y, label in pts:
            ax.scatter(x / 1024 / 1024, y)
            ax.annotate(label, (x / 1024 / 1024, y), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        front = pareto_front([(x, y) for x, y, _ in pts])
        fxy = sorted((x / 1024 / 1024, y) for (x, y, _), on in zip(pts, front) if on)
        if fxy:
            ax.plot([p[0] for p in fxy], [p[1] for p in fxy],
                    linestyle="--", marker="o", label="Pareto frontier")
            ax.legend()
    ax.axvline(4096, color="red", linestyle=":", alpha=0.6)  # the 4 GB wall
    ax.set_xlabel("Peak RSS (MB)  →  footprint")
    ax.set_ylabel("Perplexity  →  lower is better")
    ax.set_title("Faraday M4a — quality vs footprint on a 4 GB Pi 4")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    rows = load_rows(config.CSV_PATH)
    render_frontier(rows, config.RESULTS_DIR / "frontier.png")
    (config.RESULTS_DIR / "leaderboard.md").write_text(make_leaderboard(rows))
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"wrote {config.RESULTS_DIR / 'frontier.png'} and leaderboard.md ({ok} ok cells)")


if __name__ == "__main__":
    main()
