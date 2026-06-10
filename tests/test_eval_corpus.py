from faraday.eval.corpus import parse_pages, slugify


def test_slugify_makes_safe_filenames():
    assert slugify("Apollo 11") == "apollo_11"
    assert slugify("Michael Collins (astronaut)") == "michael_collins_astronaut"


def test_parse_pages_extracts_text_and_url():
    payload = {"query": {"pages": {"123": {
        "title": "Apollo 11", "extract": "Apollo 11 was a spaceflight.",
        "fullurl": "https://en.wikipedia.org/wiki/Apollo_11"}}}}
    text, url = parse_pages(payload)
    assert text == "Apollo 11 was a spaceflight."
    assert url == "https://en.wikipedia.org/wiki/Apollo_11"


def test_parse_pages_raises_on_missing_extract():
    import pytest
    with pytest.raises(ValueError):
        parse_pages({"query": {"pages": {"-1": {"title": "Nope", "missing": ""}}}})
