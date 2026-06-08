from faraday.citations import extract_citations, classify_citations


def test_extract_unique_sorted_indices():
    assert extract_citations("Foo [2] bar [1][1].") == [1, 2]


def test_classify_valid_and_invalid():
    valid, invalid = classify_citations("Uses [1] and [3].", n_sources=2)
    assert valid == [1]          # [1] is in range (1..2)
    assert invalid == [3]        # [3] is hallucinated
