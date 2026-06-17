from pathlib import Path

from faraday.preflight import HEADROOM_BYTES, fits, pick_model

MB = 1024 * 1024


def test_fits_requires_model_plus_headroom():
    assert fits(1000 * MB, available_bytes=1000 * MB + HEADROOM_BYTES) is True
    assert fits(1000 * MB, available_bytes=1000 * MB + HEADROOM_BYTES - 1) is False


def test_pick_model_prefers_largest_that_fits():
    cands = [(Path("big.gguf"), 3000 * MB), (Path("mid.gguf"), 1000 * MB),
             (Path("small.gguf"), 500 * MB)]
    assert pick_model(cands, available_bytes=1200 * MB + HEADROOM_BYTES).name == "mid.gguf"


def test_pick_model_none_when_nothing_fits():
    assert pick_model([(Path("big.gguf"), 3000 * MB)], available_bytes=1000 * MB) is None
