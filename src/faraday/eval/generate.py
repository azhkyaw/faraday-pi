"""Draft candidate golden-set items from the corpus with Claude (dev-time). The
model returns a verbatim supporting quote per item; we locate it in the source to
get reproducible char-offset spans. Output is a DRAFT for human curation, never
committed as-is. Pure helpers (prompt, span location) are unit-tested.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from faraday.eval import config


def locate_span(text: str, quote: str) -> tuple[int, int] | None:
    """Find quote in text -> (start, end). Exact first, then whitespace-tolerant."""
    i = text.find(quote)
    if i >= 0:
        return (i, i + len(quote))
    pattern = re.escape(quote.strip()).replace(r"\ ", r"\s+")
    m = re.search(pattern, text)
    return (m.start(), m.end()) if m else None


def build_item_prompt(title: str, article_text: str, n: int) -> str:
    return (
        f"From the Wikipedia article below, write {n} factual question/answer pairs "
        "for evaluating a retrieval system. Each must be answerable SOLELY from this "
        "article, with a short factual reference answer and a VERBATIM supporting "
        "quote copied exactly from the text.\n\n"
        f"Article: {title}\n\"\"\"\n{article_text}\n\"\"\"\n\n"
        "Return a JSON list; each element: "
        '{"question": str, "reference_answer": str, "supporting_quote": str}.'
    )


class AnthropicGenerator:
    """Live generator. Requires `anthropic` + ANTHROPIC_API_KEY. Not unit-tested."""

    def __init__(self, client=None, model: str = config.JUDGE_MODEL):
        import anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model

    def draft_for_article(self, title: str, text: str, n: int = 4) -> list[dict]:
        from pydantic import BaseModel

        class _Item(BaseModel):
            question: str
            reference_answer: str
            supporting_quote: str

        class _Items(BaseModel):
            items: list[_Item]

        resp = self.client.messages.parse(
            model=self.model, max_tokens=4096,
            messages=[{"role": "user", "content": build_item_prompt(title, text, n)}],
            output_format=_Items,
        )
        return [i.model_dump() for i in resp.parsed_output.items]


# Out-of-corpus questions for the abstention axis (hand-authored: plausible
# spaceflight questions NOT answerable from the Apollo-era corpus).
UNANSWERABLE = (
    "Which Space Shuttle was the first to reach orbit?",
    "Who was the first person to walk in space?",
    "What year did the International Space Station launch its first module?",
    "How many people have walked on Mars?",
    "What was the name of the first SpaceX crewed mission?",
    "Which country launched the Sputnik 1 satellite?",
)


def draft_golden(corpus_dir: Path, out_path: Path, gen: AnthropicGenerator,
                 per_article: int = 3) -> int:
    """Draft answerable items from each corpus file + append unanswerable ones."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for txt in sorted(Path(corpus_dir).glob("*.txt")):
            text = txt.read_text(encoding="utf-8")
            for it in gen.draft_for_article(txt.stem, text, per_article):
                span = locate_span(text, it["supporting_quote"])
                rec = {
                    "id": f"{txt.stem}_{n:03d}", "question": it["question"],
                    "answerable": True, "relevant_doc": txt.name,
                    "relevant_span": list(span) if span else None,
                    "reference_answer": it["reference_answer"],
                    "_quote": it["supporting_quote"],  # kept for human review; strip on curate
                }
                f.write(json.dumps(rec) + "\n")
                n += 1
        for i, q in enumerate(UNANSWERABLE):
            f.write(json.dumps({
                "id": f"unanswerable_{i:03d}", "question": q, "answerable": False,
                "relevant_doc": "", "relevant_span": None, "reference_answer": "",
            }) + "\n")
            n += 1
    return n


def main() -> None:
    out = config.EVAL_DIR / "golden_draft.jsonl"
    count = draft_golden(config.CORPUS_DIR, out, AnthropicGenerator())
    print(f"wrote {count} draft items to {out} (review -> {config.GOLDEN_PATH})")


if __name__ == "__main__":
    main()
