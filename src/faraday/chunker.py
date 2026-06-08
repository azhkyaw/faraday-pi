from __future__ import annotations
import hashlib
from faraday.models import Chunk, Document


def _doc_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def chunk_document(doc: Document, size: int = 1200, overlap: int = 200) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap >= size:
        raise ValueError("overlap must be < size")
    text = doc.text
    doc_id = _doc_id(doc.source)
    chunks: list[Chunk] = []
    start, ord_ = 0, 0
    step = size - overlap
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(Chunk(doc_id=doc_id, ord=ord_, text=text[start:end], source=doc.source))
        ord_ += 1
        if end >= len(text):  # reached the end — stop (avoids a trailing sliver chunk)
            break
        start += step
    if not chunks:  # empty document
        chunks.append(Chunk(doc_id=doc_id, ord=0, text="", source=doc.source))
    return chunks
