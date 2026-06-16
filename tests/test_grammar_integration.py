"""Live GBNF smoke: one grammar-ON request must yield text whose '[' usage is only
valid citations. If this FAILS while plain requests succeed, the OAI endpoint is
ignoring `grammar` — switch HttpLLMClient's grammar path to the native /completion
endpoint (documented contingency in the M5 plan)."""
import re

import pytest

from faraday.config import Settings
from faraday.grammar import build_citation_grammar
from faraday.llm_client import HttpLLMClient


@pytest.mark.integration
def test_grammar_constrains_live_output():
    llm = HttpLLMClient(Settings.from_env())
    g = build_citation_grammar(2)
    text = llm.complete(
        [{"role": "user", "content": "Say something brief and cite source one as [1]."}],
        max_tokens=64, grammar=g)
    assert text  # got output at all
    for m in re.finditer(r"\[([^\]]*)\]", text):
        assert m.group(1) in ("1", "2"), f"grammar leaked invalid citation: {m.group(0)!r}"
