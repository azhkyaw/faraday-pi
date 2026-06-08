from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    source: str          # filename / path
    text: str


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    ord: int             # position within the document
    text: str
    source: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
