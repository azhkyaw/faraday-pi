"""Eval-as-test (design spec §188): retrieval-only recall gate over the golden set.
Integration-marked: needs the embed server + examples/eval_corpus + golden.jsonl.
Run: pytest -m integration tests/test_retrieval_gate.py -q   (on the Pi, servers up)

THRESHOLD rule: M4b baseline recall@4 (0.805) minus a 0.10 margin, rounded down to
0.05 -> 0.70. This is the tightened value (M4b is closed and the scorecard exists),
not the initial 0.50 sanity floor — so the gate actually catches a recall regression.
"""
import pytest

from faraday.config import Settings
from faraday.embedder import HttpEmbedder
from faraday.eval import config as eval_config
from faraday.eval.dataset import load_golden
from faraday.eval.metrics import chunk_is_relevant
from faraday.index_store import SqliteVecStore
from faraday.ingest import ingest
from faraday.retriever import Retriever

THRESHOLD = 0.70  # M4b baseline recall@4 (0.805) - 0.10 margin, rounded down to 0.05


@pytest.mark.integration
def test_recall_at_4_meets_threshold(tmp_path):
    s = Settings.from_env()
    store = SqliteVecStore(str(tmp_path / "gate.sqlite"), dim=s.embed_dim)
    embedder = HttpEmbedder(s)
    ingest(eval_config.CORPUS_DIR, store, embedder, chunk_size=1200, chunk_overlap=200)
    retriever = Retriever(embedder, store)

    items = [i for i in load_golden(eval_config.GOLDEN_PATH) if i.answerable]
    hits = 0
    for item in items:
        retrieved = retriever.search(item.question, k=4)
        if any(chunk_is_relevant(rc.chunk.source, rc.chunk.ord, item, 1200, 200)
               for rc in retrieved):
            hits += 1
    store.close()
    recall = hits / len(items)
    assert recall >= THRESHOLD, f"recall@4={recall:.2f} below gate {THRESHOLD}"
