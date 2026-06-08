from faraday.prompt import build_messages
from faraday.models import RetrievedChunk, Chunk


def _rc(i, text):
    return RetrievedChunk(chunk=Chunk(doc_id="d", ord=i, text=text, source="s.txt"), score=1.0)


def test_messages_number_sources_and_instruct_citations():
    msgs = build_messages("What RAM?", [_rc(0, "4GB RAM"), _rc(1, "ARM CPU")])
    assert msgs[0]["role"] == "system"
    user = msgs[-1]["content"]
    assert "[1]" in user and "[2]" in user      # sources numbered from 1
    assert "4GB RAM" in user
    assert "What RAM?" in user


def test_abstention_instruction_present():
    msgs = build_messages("q", [_rc(0, "ctx")])
    assert "don't know" in msgs[0]["content"].lower()
