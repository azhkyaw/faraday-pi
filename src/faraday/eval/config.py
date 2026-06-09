"""M4b eval config: the ablation grid, paths, and the judge model. Pure data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Judge model. Default per the claude-api skill; swap to "claude-sonnet-4-6" to cut cost.
JUDGE_MODEL = "claude-opus-4-8"

CORPUS_DIR = Path("examples/eval_corpus")
EVAL_DIR = Path("results/evals")
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"
RAW_DIR = EVAL_DIR / "raw"      # raw_<slug>.jsonl
JUDGE_DIR = EVAL_DIR / "judge"  # judge_<slug>.jsonl (frozen judge outputs)

# Ablation grid: chunk-size brackets the real default (1200 chars); overlap = size//6.
CHUNK_SIZES = (600, 1200, 2400)
TOP_KS = (2, 4, 8)


@dataclass(frozen=True)
class AblationConfig:
    top_k: int
    chunk_size: int
    chunk_overlap: int

    @property
    def slug(self) -> str:
        return f"k{self.top_k}_c{self.chunk_size}_o{self.chunk_overlap}"


def configs() -> list[AblationConfig]:
    """The core grid: 3 chunk-sizes x 3 top_k = 9 configs (overlap = size//6)."""
    return [
        AblationConfig(top_k=k, chunk_size=cs, chunk_overlap=cs // 6)
        for cs in CHUNK_SIZES
        for k in TOP_KS
    ]
