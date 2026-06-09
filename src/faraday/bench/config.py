"""The M4a sweep matrix: which (model size x quant) cells to benchmark, where to
fetch each GGUF, and the CSV schema we record. Pure data + tiny helpers, so it
unit-tests off the Pi.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Qwen2.5 Instruct sizes that fit (or nearly fit) the 4 GB Pi, smallest first.
SIZES: tuple[str, ...] = ("0.5B", "1.5B", "3B")

# K-quant ladder, near-lossless -> aggressive.
QUANTS: tuple[str, ...] = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")

# Best-practice imatrix GGUFs published per-size by bartowski on Hugging Face.
HF_REPO_TEMPLATE = "bartowski/Qwen2.5-{size}-Instruct-GGUF"
HF_FILE_TEMPLATE = "Qwen2.5-{size}-Instruct-{quant}.gguf"

# CSV schema (one row per cell). Order is the on-disk column order.
CSV_COLUMNS: tuple[str, ...] = (
    "size", "quant", "status", "disk_bytes", "peak_rss_bytes",
    "prefill_tps", "decode_tps", "perplexity", "notes",
)

# Perplexity: chunks of the wikitext sample (each chunk = one forward pass).
PERPLEXITY_CHUNKS = 20

# Repo-relative outputs (committed) and Pi-side scratch paths (expanded at run).
RESULTS_DIR = Path("results/sweep")
CSV_PATH = RESULTS_DIR / "sweep.csv"
RAW_DIR = RESULTS_DIR / "raw"
BENCH_MODELS_DIR = Path.home() / "faraday" / "models" / "bench"
PERPLEXITY_CORPUS = Path.home() / "faraday" / "bench_data" / "wiki.test.raw"


@dataclass(frozen=True)
class Cell:
    size: str
    quant: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.size, self.quant)

    @property
    def repo(self) -> str:
        return HF_REPO_TEMPLATE.format(size=self.size)

    @property
    def filename(self) -> str:
        return HF_FILE_TEMPLATE.format(size=self.size, quant=self.quant)


def cells() -> list[Cell]:
    """All size x quant cells, smallest-model-first (so a partial run covers the
    cheap, certain-to-fit cells before the big, risky ones)."""
    return [Cell(size, quant) for size in SIZES for quant in QUANTS]
