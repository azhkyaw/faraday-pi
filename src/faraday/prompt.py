from __future__ import annotations
from faraday.models import RetrievedChunk

SYSTEM = (
    "You answer strictly from the provided sources. "
    "Cite every claim with bracketed source numbers like [1] or [2]. "
    "If the answer is not in the sources, say you don't know. "
    "Do not use outside knowledge."
)


def build_messages(query: str, chunks: list[RetrievedChunk]) -> list[dict]:
    blocks = []
    for i, rc in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {rc.chunk.source})\n{rc.chunk.text}")
    context = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    user = f"Sources:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
