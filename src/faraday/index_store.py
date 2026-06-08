from __future__ import annotations
import os
import sqlite3
import sqlite_vec
from faraday.models import Chunk, RetrievedChunk


class SqliteVecStore:
    """Single-file vector store. Chunk text/metadata in `chunks`; vectors in a vec0 table."""

    def __init__(self, path: str, dim: int):
        self.dim = dim
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, ord INTEGER, "
            "text TEXT, source TEXT)"
        )
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding float[{self.dim}] distance_metric=cosine)"
        )
        self.db.commit()

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        for chunk, vec in zip(chunks, vectors):
            cur = self.db.execute(
                "INSERT INTO chunks(doc_id, ord, text, source) VALUES (?,?,?,?)",
                (chunk.doc_id, chunk.ord, chunk.text, chunk.source),
            )
            self.db.execute(
                "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                (cur.lastrowid, sqlite_vec.serialize_float32(vec)),
            )
        self.db.commit()

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        # KNN in a subquery (clean vec0 query with MATCH + LIMIT), then join metadata.
        rows = self.db.execute(
            "SELECT c.doc_id, c.ord, c.text, c.source, m.distance FROM "
            "(SELECT chunk_id, distance FROM vec_chunks "
            " WHERE embedding MATCH ? ORDER BY distance LIMIT ?) AS m "
            "JOIN chunks c ON c.id = m.chunk_id ORDER BY m.distance",
            (sqlite_vec.serialize_float32(vector), k),
        ).fetchall()
        out = []
        for doc_id, ord_, text, source, distance in rows:
            chunk = Chunk(doc_id=doc_id, ord=ord_, text=text, source=source)
            out.append(RetrievedChunk(chunk=chunk, score=1.0 - float(distance)))
        return out

    def close(self) -> None:
        self.db.close()
