from faraday.llm_client import _tokens_from_sse


def test_tokens_from_sse_parses_deltas_and_stops_on_done():
    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        '',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: [DONE]',
        'data: {"choices":[{"delta":{"content":"ignored"}}]}',
    ]
    assert list(_tokens_from_sse(lines)) == ["Hel", "lo"]


def test_tokens_from_sse_skips_empty_and_non_data_lines():
    lines = ['', ': comment', 'data: {"choices":[{"delta":{}}]}',
             'data: {"choices":[{"delta":{"content":"x"}}]}']
    assert list(_tokens_from_sse(lines)) == ["x"]
