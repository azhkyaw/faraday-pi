"""Pure parsers for the M4a quantization sweep.

Each function takes raw text emitted by a llama.cpp tool (or GNU time) and
returns the single number we record. No I/O, no subprocess — so they unit-test
on any machine (no Pi, no native deps) and stay the trustworthy core of the
sweep. See docs/superpowers/specs/2026-06-09-faraday-m4a-quant-sweep-design.md.
"""
from __future__ import annotations

import re

_PPL_RE = re.compile(r"PPL\s*=\s*([0-9]+\.[0-9]+)")
_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*([0-9]+)")


def parse_llama_bench(text: str) -> tuple[float, float]:
    """Parse a llama-bench markdown table -> (prefill_tps, decode_tps).

    Data rows look like:
      | qwen2 1.5B Q4_K - Medium | 1.04 GiB | 1.54 B | CPU | 4 | pp512 | 7.71 ± 0.05 |
    The test column starts with 'pp' (prefill) or 'tg' (decode); the t/s column
    is last (we take the value before the '±'). Header/separator/build lines are
    skipped because their last column doesn't parse as a float.
    """
    prefill: float | None = None
    decode: float | None = None
    for line in text.splitlines():
        if "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|") if c.strip()]
        if len(cols) < 2:
            continue
        test, tps = cols[-2], cols[-1]
        try:
            value = float(tps.split()[0])  # "7.71 ± 0.05" -> 7.71
        except (ValueError, IndexError):
            continue
        if test.startswith("pp"):
            prefill = value
        elif test.startswith("tg"):
            decode = value
    if prefill is None or decode is None:
        raise ValueError(f"no pp/tg rows in llama-bench output: {text!r}")
    return prefill, decode


def parse_perplexity(text: str) -> float:
    """Parse llama-perplexity output -> final PPL (number after 'PPL =')."""
    matches = _PPL_RE.findall(text)
    if not matches:
        raise ValueError(f"no 'PPL = ...' in perplexity output: {text!r}")
    return float(matches[-1])


def parse_time_v(text: str) -> int:
    """Parse `/usr/bin/time -v` stderr -> peak RSS in BYTES.

    GNU time reports 'Maximum resident set size (kbytes)' in KiB; x1024 -> bytes.
    """
    m = _RSS_RE.search(text)
    if not m:
        raise ValueError(f"no 'Maximum resident set size' in time -v output: {text!r}")
    return int(m.group(1)) * 1024


_OLLAMA_PROMPT_RE = re.compile(r"prompt eval rate:\s*([0-9.]+)")
_OLLAMA_EVAL_RE = re.compile(r"(?m)^\s*eval rate:\s*([0-9.]+)")  # line-anchored: not "prompt eval rate"
_SPEC_ACCEPT_RE = re.compile(r"accept\s*=\s*([0-9.]+)\s*%")
_SPEC_SPEED_RE = re.compile(r"speed:\s*([0-9.]+)\s*t/s")


def parse_ollama_bench(text: str) -> tuple[float, float]:
    """Parse `ollama run --verbose` stats -> (prefill_tps, decode_tps).
    'prompt eval rate' = prefill; the line-anchored 'eval rate' = decode."""
    p = _OLLAMA_PROMPT_RE.search(text)
    d = _OLLAMA_EVAL_RE.search(text)
    if not p or not d:
        raise ValueError(f"no ollama eval rates in output: {text!r}")
    return float(p.group(1)), float(d.group(1))


def parse_speculative(text: str) -> tuple[float, float]:
    """Parse `llama-speculative` output -> (decode_tps, accept_rate_pct)."""
    speed = _SPEC_SPEED_RE.search(text)
    accept = _SPEC_ACCEPT_RE.search(text)
    if not speed or not accept:
        raise ValueError(f"no speculative speed/accept in output: {text!r}")
    return float(speed.group(1)), float(accept.group(1))
