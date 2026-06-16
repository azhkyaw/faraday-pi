"""The M4c optimization matrix: each lever as a pure cell (governor + llama-bench
flags + which tool/parser), plus the CSV schema and paths. Pure data — unit-tests
off the Pi. The model under test is fixed (1.5B Q4_K_M, the M4a frontier pick).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# CSV schema (one row per measured config). accept_rate is blank except speculative.
CSV_COLUMNS: tuple[str, ...] = (
    "component", "label", "prefill_tps", "decode_tps", "peak_rss_bytes",
    "accept_rate", "throttled", "notes",
)

RESULTS_DIR = Path("results/optimize")
CSV_PATH = RESULTS_DIR / "optimize.csv"
RAW_DIR = RESULTS_DIR / "raw"

# Context sizes for the TTFT/decode-vs-context characterization.
CONTEXT_SIZES = (128, 512, 1024, 2048, 4096)

_BASE = ("-p", "512", "-n", "128")  # baseline llama-bench workload


@dataclass(frozen=True)
class LeverCell:
    component: str          # baseline|governor|threads|batch|kvquant|flashattn|context|speculative|ollama|stacked_best
    label: str             # unique cell name, e.g. "threads=3", "ctx=2048"
    governor: str | None   # set this CPU governor first; None = leave as-is
    flags: tuple[str, ...] # llama-bench flags (empty for ollama/speculative)
    kind: str              # "llama_bench" | "ollama" | "speculative"

    @property
    def key(self) -> tuple[str, str]:
        return (self.component, self.label)


def cells() -> list[LeverCell]:
    """Baseline + one cell per lever-setting + context sizes + speculative + ollama.
    Single-value levers are one cell; multi-value (threads) are several."""
    out = [LeverCell("baseline", "baseline", "ondemand", _BASE, "llama_bench")]
    out.append(LeverCell("governor", "governor=performance", "performance", _BASE, "llama_bench"))
    for t in (2, 3):  # 4 = baseline default (nproc)
        out.append(LeverCell("threads", f"threads={t}", "ondemand",
                             (*_BASE, "-t", str(t)), "llama_bench"))
    out.append(LeverCell("batch", "ubatch=1024", "ondemand", (*_BASE, "-ub", "1024"), "llama_bench"))
    out.append(LeverCell("kvquant", "kv=q8_0", "ondemand",
                         (*_BASE, "-ctk", "q8_0", "-ctv", "q8_0", "-fa"), "llama_bench"))
    out.append(LeverCell("flashattn", "flash_attn", "ondemand", (*_BASE, "-fa"), "llama_bench"))
    for ctx in CONTEXT_SIZES:
        out.append(LeverCell("context", f"ctx={ctx}", "ondemand",
                             ("-p", str(ctx), "-n", "128"), "llama_bench"))
    out.append(LeverCell("speculative", "speculative", "ondemand", (), "speculative"))
    out.append(LeverCell("ollama", "ollama-default", None, (), "ollama"))
    return out
