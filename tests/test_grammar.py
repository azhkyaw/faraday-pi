from faraday.grammar import build_citation_grammar


def test_grammar_enumerates_exactly_the_retrieved_indices():
    g = build_citation_grammar(4)
    assert '"[1]"' in g and '"[4]"' in g
    assert '"[5]"' not in g
    assert "root ::=" in g and "cite ::=" in g


def test_grammar_text_rule_blocks_free_brackets():
    # prose may be anything except '[', so the only way to emit '[' is a valid cite
    assert "[^\\[]" in build_citation_grammar(2)


def test_grammar_zero_sources_allows_no_citations():
    g = build_citation_grammar(0)
    assert "cite" not in g
    assert "root ::=" in g
