from __future__ import annotations
from typing import Iterator
from faraday.citations import classify_citations
from faraday.events import Event, SourcesEvent, TokenEvent, DoneEvent
from faraday.llm_client import LLMClient
from faraday.models import Answer
from faraday.prompt import build_messages
from faraday.retriever import Retriever


class RagEngine:
    def __init__(self, retriever: Retriever, llm: LLMClient, top_k: int = 4,
                 max_tokens: int = 512):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_tokens = max_tokens

    def answer(self, query: str) -> Answer:
        sources = self.retriever.search(query, k=self.top_k)
        messages = build_messages(query, sources)
        text = self.llm.complete(messages, max_tokens=self.max_tokens)
        valid, invalid = classify_citations(text, n_sources=len(sources))
        return Answer(text=text, sources=sources,
                      cited_indices=valid, invalid_citations=invalid)

    def answer_stream(self, query: str) -> Iterator[Event]:
        sources = self.retriever.search(query, k=self.top_k)
        yield SourcesEvent(sources)
        messages = build_messages(query, sources)
        parts: list[str] = []
        for token in self.llm.stream(messages, max_tokens=self.max_tokens):
            parts.append(token)
            yield TokenEvent(token)
        valid, invalid = classify_citations("".join(parts), n_sources=len(sources))
        yield DoneEvent(cited_indices=valid, invalid_citations=invalid)
