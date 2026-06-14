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

    Also caps each input's length. bge-small-en-v1.5 has only 512 trained position
    embeddings, and llama-server embeds an input in a single physical batch
    (n_ubatch=512) — so an input over 512 tokens is a hard 500 ("input is too large
    to process"), not a silent truncation. A 2400-char Apollo chunk tokenizes to
    as many as 718 tokens, and 52% of the c2400 corpus exceeds 512 -- so the
    chunk_size=2400 eval configs tripped it. Measured across all 409 c2400 chunks,
    clipping each input to max_input_chars=1450 keeps every chunk <= 490 tokens (a
    ~22-token margin under 512; even 1800 chars still left 6 chunks over). We clip
    BEFORE the POST. The stored chunk text is untouched (ingest
    passes c.text to the store on a separate path), so an over-long chunk is embedded
    from a truncated view while generation still sees its full text.
    """

    def __init__(self, settings: Settings | None = None, timeout: float = 120.0,
                 batch_size: int = 16, max_input_chars: int = 1450):
        self.settings = settings or Settings()
        self.batch_size = batch_size
        self.max_input_chars = max_input_chars
        self._client = httpx.Client(base_url=self.settings.embed_url, timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t[:self.max_input_chars] for t in texts[start:start + self.batch_size]]
            resp = self._client.post("/v1/embeddings", json={"input": batch})
            resp.raise_for_status()
            data = resp.json()["data"]
            # Preserve input order (OpenAI returns an "index" per item).
            ordered = sorted(data, key=lambda d: d["index"])
            out.extend(d["embedding"] for d in ordered)
        return out

    def close(self) -> None:
        self._client.close()
