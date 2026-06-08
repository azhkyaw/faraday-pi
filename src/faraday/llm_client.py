from __future__ import annotations
from typing import Protocol, runtime_checkable
import httpx
from faraday.config import Settings


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, messages: list[dict], max_tokens: int = 512) -> str: ...


class HttpLLMClient:
    """Non-streaming chat completion against a local llama-server (streaming lands in M2)."""

    def __init__(self, settings: Settings | None = None, timeout: float = 120.0):
        self.settings = settings or Settings()
        self._client = httpx.Client(base_url=self.settings.gen_url, timeout=timeout)

    def complete(self, messages: list[dict], max_tokens: int = 512) -> str:
        resp = self._client.post(
            "/v1/chat/completions",
            json={"messages": messages, "max_tokens": max_tokens, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._client.close()
