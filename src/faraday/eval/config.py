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


# Judge answer-quality only at this baseline config (cost control); deterministic
# metrics still cover the full grid.
BASELINE = AblationConfig(top_k=4, chunk_size=1200, chunk_overlap=200)

# The themed corpus: Apollo-era crewed spaceflight (dense, overlapping facts so
# retrieval must be precise). ~15 articles.
CORPUS_TITLES = (
    "Apollo 11", "Apollo 13", "Apollo program", "Saturn V", "Neil Armstrong",
    "Buzz Aldrin", "Michael Collins (astronaut)", "Apollo Lunar Module",
    "Project Gemini", "Apollo 1", "Apollo 17", "Space Race",
    "Apollo command and service module", "Lunar Roving Vehicle", "Apollo 8",
)
