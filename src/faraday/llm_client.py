from __future__ import annotations
import json
from typing import Iterator, Protocol, runtime_checkable
import httpx
from faraday.config import Settings


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, messages: list[dict], max_tokens: int = 512) -> str: ...
    def stream(self, messages: list[dict], max_tokens: int = 512) -> Iterator[str]: ...


def _tokens_from_sse(lines: Iterator[str]) -> Iterator[str]:
    """Parse llama-server's OpenAI-style SSE lines into content tokens."""
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        content = json.loads(data)["choices"][0].get("delta", {}).get("content")
        if content:
            yield content


class HttpLLMClient:
    """Chat completion against a local llama-server (streaming + non-streaming)."""

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

    def stream(self, messages: list[dict], max_tokens: int = 512) -> Iterator[str]:
        with self._client.stream(
            "POST", "/v1/chat/completions",
            json={"messages": messages, "max_tokens": max_tokens, "stream": True},
        ) as resp:
            resp.raise_for_status()
            yield from _tokens_from_sse(resp.iter_lines())

    def close(self) -> None:
        self._client.close()
