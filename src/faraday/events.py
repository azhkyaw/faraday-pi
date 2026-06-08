from __future__ import annotations
from dataclasses import dataclass
from faraday.models import RetrievedChunk


@dataclass(frozen=True)
class SourcesEvent:
    sources: list[RetrievedChunk]


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class DoneEvent:
    cited_indices: list[int]
    invalid_citations: list[int]


@dataclass(frozen=True)
class ErrorEvent:
    message: str


Event = SourcesEvent | TokenEvent | DoneEvent | ErrorEvent
