from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from faraday.chunker import chunk_document
from faraday.embedder import Embedder
from faraday.index_store import SqliteVecStore
from faraday.parsers import load_document, TEXT_EXTS

SUPPORTED = TEXT_EXTS | {".pdf"}


@dataclass
class IngestStats:
    documents: int = 0
    chunks: int = 0
    skipped: int = 0


def ingest(source_dir, store: SqliteVecStore, embedder: Embedder,
           chunk_size: int = 1200, chunk_overlap: int = 200) -> IngestStats:
    source_dir = Path(source_dir)
    stats = IngestStats()
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED:
            stats.skipped += 1
            continue
        doc = load_document(path)
        chunks = chunk_document(doc, size=chunk_size, overlap=chunk_overlap)
        vectors = embedder.embed([c.text for c in chunks])
        store.add_chunks(chunks, vectors)
        stats.documents += 1
        stats.chunks += len(chunks)
    return stats
