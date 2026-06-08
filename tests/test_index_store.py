from faraday.models import Chunk
from faraday.index_store import SqliteVecStore


def _chunk(i, text):
    return Chunk(doc_id="d1", ord=i, text=text, source="s.txt")


def test_add_and_search_returns_nearest(tmp_path):
    store = SqliteVecStore(str(tmp_path / "t.sqlite"), dim=3)
    store.add_chunks(
        [_chunk(0, "red"), _chunk(1, "blue")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    results = store.search([0.9, 0.1, 0.0], k=1)
    assert len(results) == 1
    assert results[0].chunk.text == "red"
    assert results[0].score >= 0.0
    store.close()


def test_search_respects_k(tmp_path):
    store = SqliteVecStore(str(tmp_path / "t.sqlite"), dim=3)
    store.add_chunks(
        [_chunk(0, "a"), _chunk(1, "b"), _chunk(2, "c")],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert len(store.search([1, 0, 0], k=2)) == 2
    store.close()
