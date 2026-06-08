from __future__ import annotations
from typing import Protocol, runtime_checkable
import httpx
from faraday.config import Settings


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HttpEmbedder:
    """Calls a llama-server (started with --embeddings) OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings | None = None, timeout: float = 60.0):
        self.settings = settings or Settings()
        self._client = httpx.Client(base_url=self.settings.embed_url, timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post("/v1/embeddings", json={"input": texts})
        resp.raise_for_status()
        data = resp.json()["data"]
        # Preserve input order (OpenAI returns an "index" per item).
        ordered = sorted(data, key=lambda d: d["index"])
        return [d["embedding"] for d in ordered]

    def close(self) -> None:
        self._client.close()
