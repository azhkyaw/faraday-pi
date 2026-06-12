from __future__ import annotations
from typing import Protocol, runtime_checkable
import httpx
from faraday.config import Settings


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HttpEmbedder:
    """Calls a llama-server (started with --embeddings) OpenAI-compatible endpoint.

    Sends bounded batches: the server returns nothing until an entire request is
    embedded (~300 tok/s on the Pi 4), so an unbounded POST turns big documents
    into client read-timeouts. Worst measured batch (16 x ~600-token chunks) is
    ~31 s; the 120 s timeout leaves ~4x headroom for a busy board.
    """

    def __init__(self, settings: Settings | None = None, timeout: float = 120.0,
                 batch_size: int = 16):
        self.settings = settings or Settings()
        self.batch_size = batch_size
        self._client = httpx.Client(base_url=self.settings.embed_url, timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            resp = self._client.post("/v1/embeddings", json={"input": batch})
            resp.raise_for_status()
            data = resp.json()["data"]
            # Preserve input order (OpenAI returns an "index" per item).
            ordered = sorted(data, key=lambda d: d["index"])
            out.extend(d["embedding"] for d in ordered)
        return out

    def close(self) -> None:
        self._client.close()
