"""On-Pi eval runner: drive the real RagEngine per ablation config and record raw
outputs as JSONL. Resumable (skip done (config, qid)). The pure record/IO helpers
unit-test off-Pi; run() is exercised by the Pi integration (Plan 2).
"""
from __future__ import annotations

import json
from pathlib import Path

from faraday.config import Settings
from faraday.eval import config
from faraday.eval.config import AblationConfig
from faraday.eval.dataset import EvalItem, load_golden
from faraday.eval.metrics import is_abstention
from faraday.models import Answer


def record_from_answer(cfg: AblationConfig, qid: str, answer: Answer) -> dict:
    return {
        "config": {"top_k": cfg.top_k, "chunk_size": cfg.chunk_size,
                   "chunk_overlap": cfg.chunk_overlap},
        "slug": cfg.slug,
        "qid": qid,
        "retrieved": [{"source": rc.chunk.source, "ord": rc.chunk.ord,
                       "text": rc.chunk.text}
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


def _raw_path(cfg: AblationConfig) -> Path:
    return config.RAW_DIR / f"{cfg.slug}.jsonl"


def run_config(cfg: AblationConfig, engine, items: list[EvalItem], raw_path: Path) -> int:
    """Ask each not-yet-done question through `engine` and record raw. Returns #new."""
    done = done_keys(raw_path)
    n = 0
    for item in items:
        if (cfg.slug, item.id) in done:
            continue
        answer = engine.answer(item.question)
        append_record(raw_path, record_from_answer(cfg, item.id, answer))
        n += 1
    return n


def build_retriever(chunk_size: int, overlap: int, settings: Settings):
    """Pi-only: ingest the corpus at this chunk-size into a fresh store. Needs
    sqlite-vec + the embed server up. Called once per chunk-size — see run()."""
    from faraday.embedder import HttpEmbedder
    from faraday.index_store import SqliteVecStore
    from faraday.ingest import ingest
    from faraday.retriever import Retriever

    db = config.EVAL_DIR / f"store_c{chunk_size}.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()  # fresh store (CREATE TABLE IF NOT EXISTS would otherwise dup)
    embedder = HttpEmbedder(settings)
    store = SqliteVecStore(str(db), dim=settings.embed_dim)
    ingest(config.CORPUS_DIR, store, embedder, chunk_size=chunk_size, chunk_overlap=overlap)
    return Retriever(embedder, store)


def run(retriever_factory=None, llm=None) -> None:
    """Full grid on the Pi: ingest ONCE per chunk-size, then share that retriever
    across the top_k engines (the ingest is the expensive part; a RagEngine is cheap).
    Factories injectable so the grid logic unit-tests off-Pi."""
    from faraday.rag import RagEngine

    settings = Settings.from_env()
    items = load_golden(config.GOLDEN_PATH)
    make_retriever = retriever_factory or build_retriever
    if llm is None:
        from faraday.llm_client import HttpLLMClient
        # Batch job, not interactive chat: complete() is non-streaming and deep-
        # context prefill measured 6.75 tok/s on the Pi 4, so a k8_c2400 question
        # legitimately takes ~12-14 min before llama-server sends a single byte.
        # 120 s (the app default) killed the k8 cells; hangs are the run monitor's
        # job to catch, not this timeout's.
        llm = HttpLLMClient(settings, timeout=1800.0)
    by_size: dict[int, int] = {}
    for cfg in config.configs():
        by_size[cfg.chunk_size] = cfg.chunk_overlap
    for size, overlap in sorted(by_size.items()):
        print(f"=== ingest chunk_size={size} (overlap {overlap}) ===", flush=True)
        retriever = make_retriever(size, overlap, settings)
        for top_k in config.TOP_KS:
            cfg = AblationConfig(top_k=top_k, chunk_size=size, chunk_overlap=overlap)
            print(f"--- {cfg.slug} ---", flush=True)
            engine = RagEngine(retriever, llm, top_k=top_k)
            made = run_config(cfg, engine, items, _raw_path(cfg))
            print(f"    recorded {made} new rows", flush=True)


if __name__ == "__main__":
    run()
