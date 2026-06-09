import pytest

from bench_samples import LLAMA_BENCH_MD, PERPLEXITY, TIME_V
from faraday.bench.parsers import parse_llama_bench, parse_perplexity, parse_time_v


def test_parse_llama_bench_returns_prefill_and_decode():
    prefill, decode = parse_llama_bench(LLAMA_BENCH_MD)
    assert prefill == 7.71
    assert decode == 3.87


def test_parse_llama_bench_raises_when_rows_missing():
    with pytest.raises(ValueError):
        parse_llama_bench("no table here")


def test_parse_perplexity_takes_final_estimate():
    assert parse_perplexity(PERPLEXITY) == 6.9543


def test_parse_perplexity_raises_when_absent():
    with pytest.raises(ValueError):
        parse_perplexity("perplexity: calculating ...")


def test_parse_time_v_returns_bytes():
    # 1093284 KiB * 1024 = bytes
    assert parse_time_v(TIME_V) == 1093284 * 1024


def test_parse_time_v_raises_when_absent():
    with pytest.raises(ValueError):
        parse_time_v("Exit status: 0")
