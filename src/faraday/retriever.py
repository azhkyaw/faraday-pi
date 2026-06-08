from __future__ import annotations
from faraday.embedder import Embedder
from faraday.index_store import SqliteVecStore
from faraday.models import RetrievedChunk


class Retriever:
    def __init__(self, embedder: Embedder, store: SqliteVecStore):
        self.embedder = embedder
        self.store = store

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        query_vec = self.embedder.embed([query])[0]
        return self.store.search(query_vec, k=k)
