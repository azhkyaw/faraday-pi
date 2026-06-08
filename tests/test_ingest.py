from pathlib import Path
from faraday.index_store import SqliteVecStore
from faraday.ingest import ingest


def test_ingest_indexes_documents(tmp_path: Path, fake_embedder):
    (tmp_path / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
    (tmp_path / "b.md").write_text("# H\n\ndelta epsilon", encoding="utf-8")
    store = SqliteVecStore(str(tmp_path / "i.sqlite"), dim=fake_embedder.dim)
    stats = ingest(tmp_path, store=store, embedder=fake_embedder,
                   chunk_size=100, chunk_overlap=10)
    assert stats.documents == 2
    assert stats.chunks >= 2
    assert len(store.search(fake_embedder.embed(["alpha"])[0], k=1)) == 1
    store.close()


def test_ingest_skips_unsupported_files(tmp_path: Path, fake_embedder):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "img.png").write_bytes(b"\x89PNG")
    store = SqliteVecStore(str(tmp_path / "i.sqlite"), dim=fake_embedder.dim)
    stats = ingest(tmp_path, store=store, embedder=fake_embedder)
    assert stats.documents == 1
    assert stats.skipped >= 1
    store.close()
