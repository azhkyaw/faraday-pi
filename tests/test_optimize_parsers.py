import pytest

from faraday.bench.parsers import parse_ollama_bench, parse_speculative

OLLAMA = """\
total duration:       13.5s
load duration:        612ms
prompt eval count:    26 token(s)
prompt eval duration: 1.21s
prompt eval rate:     21.49 tokens/s
eval count:           298 token(s)
eval duration:        12.3s
eval rate:            24.23 tokens/s
"""

SPECULATIVE = """\
n_draft   = 16
n_predict = 128
n_drafted = 144
n_accept  = 95
accept    = 65.97%

encoded   14 tokens in  2.64 seconds, speed: 5.31 t/s
decoded  128 tokens in 30.50 seconds, speed: 4.20 t/s
"""


def test_parse_ollama_bench_returns_prefill_and_decode():
    prefill, decode = parse_ollama_bench(OLLAMA)
    assert prefill == 21.49
    assert decode == 24.23


def test_parse_ollama_bench_raises_when_absent():
    with pytest.raises(ValueError):
        parse_ollama_bench("no stats here")


def test_parse_speculative_returns_decode_and_accept():
    decode, accept = parse_speculative(SPECULATIVE)
    assert decode == 4.20
    assert accept == 65.97


def test_parse_speculative_raises_when_absent():
    with pytest.raises(ValueError):
        parse_speculative("nothing useful")
