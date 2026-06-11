"""Claude-as-judge for answer faithfulness + correctness. Dev-time only (never on
the Pi). The judge is a Protocol; tests inject a fake. The real AnthropicJudge uses
the anthropic SDK's messages.parse (structured output) with model claude-opus-4-8.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from faraday.eval import config


@dataclass(frozen=True)
class JudgeVerdict:
    faithfulness: int   # 1-5: is the answer grounded in the retrieved context (no hallucination)?
    correctness: int    # 1-5: does it match the reference answer?
    rationale: str


class Judge(Protocol):
    def score(self, *, question: str, reference_answer: str,
              context: str, answer: str) -> JudgeVerdict: ...

    def classify_abstention(self, *, question: str, answer: str) -> bool: ...


def build_judge_prompt(*, question: str, reference_answer: str,
                       context: str, answer: str) -> str:
    return (
        "You are grading a retrieval-augmented answer. Score two axes from 1-5.\n"
        "- faithfulness: is EVERY claim supported by the Retrieved context? "
        "(5 = fully grounded, 1 = hallucinated)\n"
        "- correctness: does the Answer match the Reference answer? "
        "(5 = fully correct, 1 = wrong)\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Answer to grade:\n{answer}\n\n"
        "Return faithfulness, correctness, and a one-sentence rationale."
    )


def build_abstention_prompt(*, question: str, answer: str) -> str:
    return (
        "You are checking whether a RAG system's answer ABSTAINED — i.e. declined to "
        "answer because the sources don't contain the information — or actually "
        "committed to an answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Return abstained: true if the answer declines or says the information is "
        "missing, false if it commits to an answer."
    )


class AnthropicJudge:
    """Real judge. Requires `anthropic` + ANTHROPIC_API_KEY. Not unit-tested (live)."""

    def __init__(self, client=None, model: str = config.JUDGE_MODEL):
        import anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model

    def score(self, *, question: str, reference_answer: str,
              context: str, answer: str) -> JudgeVerdict:
        from pydantic import BaseModel  # provided transitively by anthropic

        class _Scores(BaseModel):
            faithfulness: int
            correctness: int
            rationale: str

        prompt = build_judge_prompt(question=question, reference_answer=reference_answer,
                                    context=context, answer=answer)
        resp = self.client.messages.parse(
            model=self.model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=_Scores,
        )
        s = resp.parsed_output
        return JudgeVerdict(faithfulness=int(s.faithfulness),
                            correctness=int(s.correctness), rationale=s.rationale)

    def classify_abstention(self, *, question: str, answer: str) -> bool:
        from pydantic import BaseModel  # provided transitively by anthropic

        class _Abstained(BaseModel):
            abstained: bool

        prompt = build_abstention_prompt(question=question, answer=answer)
        resp = self.client.messages.parse(
            model=self.model, max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
            output_format=_Abstained,
        )
        return bool(resp.parsed_output.abstained)
