from faraday.eval.generate import build_item_prompt, locate_span


def test_locate_span_finds_quote_offsets():
    text = "Apollo 11 was the spaceflight that first landed humans on the Moon."
    span = locate_span(text, "first landed humans on the Moon")
    assert span is not None
    assert text[span[0]:span[1]] == "first landed humans on the Moon"


def test_locate_span_returns_none_when_absent():
    assert locate_span("some text", "not present here") is None


def test_build_item_prompt_includes_article_and_count():
    p = build_item_prompt("Apollo 11", "Apollo 11 landed in 1969.", n=5)
    assert "Apollo 11" in p
    assert "Apollo 11 landed in 1969." in p
    assert "5" in p
    assert "verbatim" in p.lower()  # instructs an exact supporting quote
