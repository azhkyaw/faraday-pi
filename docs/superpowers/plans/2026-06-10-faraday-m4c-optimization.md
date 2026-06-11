# Faraday M4c — Inference Optimization Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable optimization harness (extending `faraday.bench`) that ablates CPU/inference tuning levers on 1.5B Q4_K_M, stacks the winners into a best config, compares speculative decoding and an Ollama baseline, characterizes TTFT/decode vs context, and plots a baseline→best throughput waterfall.

**Architecture:** Extend the tested M4a `faraday.bench` package. A pure `optimize_config` describes each lever cell (governor/flags/kind); `optimize.py` applies a cell and times `llama-bench`/`llama-speculative`/`ollama` via the **reused** injected `Runner`, parsing tok/s with the **reused** `parse_llama_bench`/`parse_time_v` plus two new parsers; an ablate-then-stack planner picks winners; `optimize_plot.py` renders the deliverables. Resumable, throttle-flagged, all on 1.5B Q4_K_M.

**Tech Stack:** Python 3.11+, llama.cpp (`llama-bench`, `llama-speculative`), Ollama, GNU `time -v`, matplotlib (Agg), pytest, ruff. Reuses `faraday.bench.parsers` + `faraday.bench.sweep` (`Completed`/`Runner`/`subprocess_runner`).

**Spec:** [../specs/2026-06-10-faraday-m4c-optimization-design.md](../specs/2026-06-10-faraday-m4c-optimization-design.md)

**Reused interfaces (verified):** `parse_llama_bench(text)->(prefill,decode)`, `parse_time_v(text)->int bytes` (`faraday.bench.parsers`); `Completed(returncode,stdout,stderr)`, `Runner = Callable[[list[str]],Completed]`, `subprocess_runner(argv)->Completed` (`faraday.bench.sweep`). The injected-Runner pattern (real subprocess at the edge, fake in tests) and resumable-CSV pattern mirror M4a exactly.

> **Execution note:** the harness/tests need no Pi (author on Windows → `git push pi` → `pytest` on the Pi). The *run* (Task 8) needs a **quiet board** + the verified PSU, so it sequences after the M4a clean sweep + M4b run. Overclock is a separate reboot procedure (Task 7 script), not part of the resumable runner.

---

### Task 1: `optimize_config.py` — the lever matrix (TDD)

**Files:**
- Create: `src/faraday/bench/optimize_config.py`
- Test: `tests/test_optimize_config.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_optimize_config.py`:

```python
from faraday.bench.optimize_config import CSV_COLUMNS, LeverCell, cells


def test_csv_schema_is_stable():
    assert CSV_COLUMNS == (
        "component", "label", "prefill_tps", "decode_tps", "peak_rss_bytes",
        "accept_rate", "throttled", "notes",
    )


def test_cells_has_baseline_first_and_expected_components():
    cs = cells()
    assert cs[0] == LeverCell("baseline", "baseline", "ondemand",
                              ("-p", "512", "-n", "128"), "llama_bench")
    comps = {c.component for c in cs}
    # every designed component present
    assert comps == {"baseline", "governor", "threads", "batch",
                     "kvquant", "flashattn", "context", "speculative", "ollama"}


def test_lever_cells_carry_the_right_flags():
    by_label = {c.label: c for c in cells()}
    assert by_label["governor=performance"].governor == "performance"
    assert by_label["threads=3"].flags == ("-p", "512", "-n", "128", "-t", "3")
    assert by_label["flash_attn"].flags == ("-p", "512", "-n", "128", "-fa")
    # V-cache quant requires flash-attn in llama.cpp, so kvquant implies -fa:
    assert by_label["kv=q8_0"].flags == (
        "-p", "512", "-n", "128", "-ctk", "q8_0", "-ctv", "q8_0", "-fa")
    assert by_label["ctx=2048"].flags == ("-p", "2048", "-n", "128")
    assert by_label["speculative"].kind == "speculative"
    assert by_label["ollama-default"].kind == "ollama"
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`):

`ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_optimize_config.py -q"`

- [ ] **Step 3: Implement.** Create `src/faraday/bench/optimize_config.py`:

```python
"""The M4c optimization matrix: each lever as a pure cell (governor + llama-bench
flags + which tool/parser), plus the CSV schema and paths. Pure data — unit-tests
off the Pi. The model under test is fixed (1.5B Q4_K_M, the M4a frontier pick).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# CSV schema (one row per measured config). accept_rate is blank except speculative.
CSV_COLUMNS: tuple[str, ...] = (
    "component", "label", "prefill_tps", "decode_tps", "peak_rss_bytes",
    "accept_rate", "throttled", "notes",
)

RESULTS_DIR = Path("results/optimize")
CSV_PATH = RESULTS_DIR / "optimize.csv"
RAW_DIR = RESULTS_DIR / "raw"

# Context sizes for the TTFT/decode-vs-context characterization.
CONTEXT_SIZES = (128, 512, 1024, 2048, 4096)

_BASE = ("-p", "512", "-n", "128")  # baseline llama-bench workload


@dataclass(frozen=True)
class LeverCell:
    component: str          # baseline|governor|threads|batch|kvquant|flashattn|context|speculative|ollama|stacked_best
    label: str             # unique cell name, e.g. "threads=3", "ctx=2048"
    governor: str | None   # set this CPU governor first; None = leave as-is
    flags: tuple[str, ...] # llama-bench flags (empty for ollama/speculative)
    kind: str              # "llama_bench" | "ollama" | "speculative"

    @property
    def key(self) -> tuple[str, str]:
        return (self.component, self.label)


def cells() -> list[LeverCell]:
    """Baseline + one cell per lever-setting + context sizes + speculative + ollama.
    Single-value levers are one cell; multi-value (threads) are several."""
    out = [LeverCell("baseline", "baseline", "ondemand", _BASE, "llama_bench")]
    out.append(LeverCell("governor", "governor=performance", "performance", _BASE, "llama_bench"))
    for t in (2, 3):  # 4 = baseline default (nproc)
        out.append(LeverCell("threads", f"threads={t}", "ondemand",
                             (*_BASE, "-t", str(t)), "llama_bench"))
    out.append(LeverCell("batch", "ubatch=1024", "ondemand", (*_BASE, "-ub", "1024"), "llama_bench"))
    out.append(LeverCell("kvquant", "kv=q8_0", "ondemand",
                         (*_BASE, "-ctk", "q8_0", "-ctv", "q8_0", "-fa"), "llama_bench"))
    out.append(LeverCell("flashattn", "flash_attn", "ondemand", (*_BASE, "-fa"), "llama_bench"))
    for ctx in CONTEXT_SIZES:
        out.append(LeverCell("context", f"ctx={ctx}", "ondemand",
                             ("-p", str(ctx), "-n", "128"), "llama_bench"))
    out.append(LeverCell("speculative", "speculative", "ondemand", (), "speculative"))
    out.append(LeverCell("ollama", "ollama-default", None, (), "ollama"))
    return out
```

- [ ] **Step 4: Run — expect PASS** (3 passed). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_optimize_config.py -q && ruff check src/faraday/bench/optimize_config.py tests/test_optimize_config.py"
git add src/faraday/bench/optimize_config.py tests/test_optimize_config.py
git commit -m "feat(m4c): optimization lever matrix + CSV schema"
```

---

### Task 2: `parsers.py` — Ollama + speculative parsers (TDD)

**Files:**
- Modify: `src/faraday/bench/parsers.py`
- Test: `tests/test_optimize_parsers.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_optimize_parsers.py`:

```python
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

decoded 128 tokens in 30.50 s, speed: 4.20 t/s
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
```

> The samples are representative of the tools' output. The Task 8 integration smoke saves the **real** stdout/stderr to `results/optimize/raw/` — if a live format differs, adjust the regex against that captured log (a ~2-min fix; the assertions stay real).

- [ ] **Step 2: Run — expect FAIL** (`ImportError`).

- [ ] **Step 3: Implement.** Append to `src/faraday/bench/parsers.py` (after `parse_time_v`):

```python
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
```

- [ ] **Step 4: Run — expect PASS** (4 passed). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_optimize_parsers.py -q && ruff check src/faraday/bench/parsers.py tests/test_optimize_parsers.py"
git add src/faraday/bench/parsers.py tests/test_optimize_parsers.py
git commit -m "feat(m4c): ollama + speculative output parsers"
```

---

### Task 3: `optimize.py` — argv builder + cell runner (TDD)

**Files:**
- Create: `src/faraday/bench/optimize.py`
- Test: `tests/test_optimize.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_optimize.py`:

```python
from faraday.bench.optimize_config import CSV_COLUMNS, LeverCell
from faraday.bench.optimize import append_row, build_argv, read_done, run_cell
from faraday.bench.sweep import Completed

BENCH_OUT = (
    "| qwen2 1.5B Q4_K - Medium | 1.04 GiB | 1.54 B | CPU | 4 | pp512 | 9.80 ± 0.01 |\n"
    "| qwen2 1.5B Q4_K - Medium | 1.04 GiB | 1.54 B | CPU | 4 | tg128 | 4.50 ± 0.00 |\n"
)
TIME_V = "\tMaximum resident set size (kbytes): 1100000\n\tExit status: 0\n"
OLLAMA = "prompt eval rate: 21.49 tokens/s\neval rate: 24.23 tokens/s\n"
SPEC = "accept = 65.97%\ndecoded 128 tokens in 30.5 s, speed: 4.20 t/s\n"


def _fake_run(*, bench=(0, BENCH_OUT, TIME_V), ollama=(0, OLLAMA, ""),
              spec=(0, SPEC, ""), throttle="throttled=0x0"):
    calls = []

    def run(argv):
        calls.append(argv)
        if argv[0] == "sudo":
            return Completed(0, "", "")
        if "get_throttled" in argv:
            return Completed(0, throttle, "")
        if "llama-bench" in argv:
            return Completed(*bench)
        if "ollama" in argv:
            return Completed(*ollama)
        if "llama-speculative" in argv:
            return Completed(*spec)
        raise AssertionError(argv)

    run.calls = calls
    return run


def test_build_argv_llama_bench_wraps_in_time():
    cell = LeverCell("threads", "threads=3", "ondemand",
                     ("-p", "512", "-n", "128", "-t", "3"), "llama_bench")
    argv = build_argv(cell, model="m.gguf", draft="d.gguf", ollama_model="q", prompt="hi")
    assert argv[:4] == ["/usr/bin/time", "-v", "llama-bench", "-m"]
    assert "-t" in argv and "3" in argv and "-o" in argv


def test_build_argv_ollama_and_speculative():
    base = dict(model="m.gguf", draft="d.gguf", ollama_model="qwen2.5:1.5b", prompt="hi")
    oll = build_argv(LeverCell("ollama", "ollama-default", None, (), "ollama"), **base)
    assert oll == ["ollama", "run", "--verbose", "qwen2.5:1.5b", "hi"]
    spec = build_argv(LeverCell("speculative", "speculative", "ondemand", (), "speculative"), **base)
    assert spec[:3] == ["/usr/bin/time", "-v", "llama-speculative"]
    assert "-md" in spec and "d.gguf" in spec


def test_run_cell_llama_bench_records_row(tmp_path):
    cell = LeverCell("baseline", "baseline", "ondemand", ("-p", "512", "-n", "128"), "llama_bench")
    run = _fake_run()
    row = run_cell(cell, run, model="m.gguf", draft="d.gguf",
                   ollama_model="q", prompt="hi", raw_dir=tmp_path)
    assert row["component"] == "baseline"
    assert row["prefill_tps"] == 9.80 and row["decode_tps"] == 4.50
    assert row["peak_rss_bytes"] == 1100000 * 1024
    assert row["throttled"] == "throttled=0x0"
    assert row["notes"] == "ok"


def test_run_cell_sets_governor_then_benches():
    cell = LeverCell("governor", "governor=performance", "performance",
                     ("-p", "512", "-n", "128"), "llama_bench")
    run = _fake_run()
    run_cell(cell, run, model="m.gguf", draft="d", ollama_model="q", prompt="hi", raw_dir=None)
    assert any(a[0] == "sudo" and "performance" in " ".join(a) for a in run.calls)


def test_run_cell_speculative_records_accept(tmp_path):
    cell = LeverCell("speculative", "speculative", "ondemand", (), "speculative")
    row = run_cell(cell, _fake_run(), model="m", draft="d", ollama_model="q",
                   prompt="hi", raw_dir=tmp_path)
    assert row["decode_tps"] == 4.20 and row["accept_rate"] == 65.97


def test_append_and_read_done_roundtrip(tmp_path):
    csv_path = tmp_path / "optimize.csv"
    row = dict.fromkeys(CSV_COLUMNS, "")
    row["component"], row["label"] = "baseline", "baseline"
    append_row(csv_path, row)
    assert read_done(csv_path) == {("baseline", "baseline")}
    assert read_done(tmp_path / "nope.csv") == set()
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement.** Create `src/faraday/bench/optimize.py`:

```python
"""M4c optimization runner: apply a lever cell, time the tool, record a CSV row.
Reuses the M4a injected-Runner + parsers. Resumable. The pure helpers unit-test
off the Pi; main() (path resolution + the full sweep) runs on the Pi.
"""
from __future__ import annotations

import csv
from pathlib import Path

from faraday.bench import parsers
from faraday.bench.optimize_config import CSV_COLUMNS, LeverCell
from faraday.bench.sweep import Completed, Runner, subprocess_runner


def _governor_cmd(gov: str) -> list[str]:
    # Passwordless sudo on the Pi; sh -c for the cpu* glob.
    return ["sudo", "sh", "-c",
            f"echo {gov} | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"]


def build_argv(cell: LeverCell, *, model: str, draft: str,
               ollama_model: str, prompt: str) -> list[str]:
    if cell.kind == "llama_bench":
        return ["/usr/bin/time", "-v", "llama-bench", "-m", model, *cell.flags, "-o", "md"]
    if cell.kind == "speculative":
        return ["/usr/bin/time", "-v", "llama-speculative", "-m", model, "-md", draft,
                "-p", prompt, "-n", "128", "--draft-max", "16"]
    if cell.kind == "ollama":
        return ["ollama", "run", "--verbose", ollama_model, prompt]
    raise ValueError(f"unknown cell kind: {cell.kind!r}")


def _save_raw(raw_dir: Path | None, cell: LeverCell, c: Completed) -> None:
    if raw_dir is None:
        return
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    (Path(raw_dir) / f"{cell.component}-{cell.label}.log".replace("/", "_")).write_text(
        f"returncode={c.returncode}\n--- stdout ---\n{c.stdout}\n--- stderr ---\n{c.stderr}\n")


def run_cell(cell: LeverCell, run: Runner, *, model: str, draft: str,
             ollama_model: str, prompt: str, raw_dir: Path | None) -> dict:
    """Apply governor (if any) -> run the tool -> parse -> build a CSV row.
    Never raises for a measurement failure: records `notes` and returns."""
    row = dict.fromkeys(CSV_COLUMNS, "")
    row["component"], row["label"] = cell.component, cell.label
    if cell.governor:
        run(_governor_cmd(cell.governor))
    c = run(build_argv(cell, model=model, draft=draft,
                       ollama_model=ollama_model, prompt=prompt))
    _save_raw(raw_dir, cell, c)
    row["throttled"] = run(["vcgencmd", "get_throttled"]).stdout.strip()
    try:
        if cell.kind == "llama_bench":
            row["peak_rss_bytes"] = parsers.parse_time_v(c.stderr)
            row["prefill_tps"], row["decode_tps"] = parsers.parse_llama_bench(c.stdout)
        elif cell.kind == "ollama":
            row["prefill_tps"], row["decode_tps"] = parsers.parse_ollama_bench(c.stdout + c.stderr)
        elif cell.kind == "speculative":
            row["decode_tps"], row["accept_rate"] = parsers.parse_speculative(c.stdout + c.stderr)
        row["notes"] = "ok" if c.returncode == 0 else f"exit {c.returncode}"
    except Exception as e:  # parse/oom/unsupported — record and keep going
        row["notes"] = f"{type(e).__name__}: {e}"
    return row


def append_row(csv_path: Path, row: dict) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    is_new = not Path(csv_path).exists()
    with Path(csv_path).open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def read_done(csv_path: Path) -> set[tuple[str, str]]:
    p = Path(csv_path)
    if not p.exists():
        return set()
    with p.open(newline="") as f:
        return {(r["component"], r["label"]) for r in csv.DictReader(f)}


def load_rows(csv_path: Path) -> list[dict]:
    with Path(csv_path).open(newline="") as f:
        return list(csv.DictReader(f))
```

- [ ] **Step 4: Run — expect PASS** (6 passed). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_optimize.py -q && ruff check src/faraday/bench/optimize.py tests/test_optimize.py"
git add src/faraday/bench/optimize.py tests/test_optimize.py
git commit -m "feat(m4c): optimize runner (build_argv/run_cell/resumable CSV)"
```

---

### Task 4: `optimize.py` — ablate-then-stack planner + `main` (TDD)

**Files:**
- Modify: `src/faraday/bench/optimize.py` (extend)
- Test: `tests/test_optimize_plan.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_optimize_plan.py`:

```python
from faraday.bench.optimize_config import cells
from faraday.bench.optimize import stack_winners


def _row(component, label, decode, throttled="throttled=0x0"):
    return {"component": component, "label": label, "prefill_tps": "5",
            "decode_tps": str(decode), "peak_rss_bytes": "1", "accept_rate": "",
            "throttled": throttled, "notes": "ok"}


def test_stack_winners_picks_best_per_lever_over_baseline():
    cs = cells()
    rows = [
        _row("baseline", "baseline", 4.50),
        _row("governor", "governor=performance", 5.20),   # win
        _row("threads", "threads=2", 4.10),               # lose
        _row("threads", "threads=3", 4.90),               # win (best thread)
        _row("batch", "ubatch=1024", 4.40),               # lose
        _row("kvquant", "kv=q8_0", 4.80),                 # win
        _row("flashattn", "flash_attn", 4.55),            # win
    ]
    best = stack_winners(cs, rows)
    assert best.component == "stacked_best"
    assert best.governor == "performance"          # governor winner
    assert "-t" in best.flags and "3" in best.flags  # best thread setting only
    assert "-ctk" in best.flags                    # kvquant winner
    assert "-fa" in best.flags                      # flashattn (and kvquant) winner
    assert "-ub" not in best.flags                 # batch lost


def test_stack_winners_ignores_throttled_cells():
    cs = cells()
    rows = [
        _row("baseline", "baseline", 4.50),
        _row("governor", "governor=performance", 9.99, throttled="throttled=0x50000"),  # fast but DIRTY
    ]
    best = stack_winners(cs, rows)
    assert best.governor == "ondemand"  # throttled winner rejected
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError`).

- [ ] **Step 3: Implement.** Append to `src/faraday/bench/optimize.py`:

```python
import os  # noqa: E402  (kept with the other imports conceptually; grouped at top on final lint)
from collections import defaultdict  # noqa: E402

from faraday.bench import optimize_config  # noqa: E402
from faraday.bench.optimize_config import CSV_PATH, RAW_DIR


def _is_clean(row: dict) -> bool:
    return bool(row["decode_tps"]) and "0x0" in str(row["throttled"])


def stack_winners(all_cells: list[LeverCell], rows: list[dict]) -> LeverCell:
    """Per lever-group, pick the single setting whose CLEAN decode beats baseline;
    union the winners' flags + best governor into one 'stacked_best' cell."""
    by_label = {r["label"]: r for r in rows}
    base_decode = float(by_label["baseline"]["decode_tps"])
    groups: dict[str, list[LeverCell]] = defaultdict(list)
    for c in all_cells:
        if c.component in ("governor", "threads", "batch", "kvquant", "flashattn"):
            groups[c.component].append(c)
    governor = "ondemand"
    extra: list[str] = []
    base_set = set(("-p", "512", "-n", "128"))
    for comp, comp_cells in groups.items():
        best_cell, best_dec = None, base_decode
        for cell in comp_cells:
            r = by_label.get(cell.label)
            if not r or not _is_clean(r):
                continue
            d = float(r["decode_tps"])
            if d > best_dec:
                best_dec, best_cell = d, cell
        if best_cell is None:
            continue
        if comp == "governor":
            governor = best_cell.governor
        else:
            for f in best_cell.flags:
                if f not in base_set and f not in extra:
                    extra.append(f)
    return LeverCell("stacked_best", "stacked_best", governor,
                     ("-p", "512", "-n", "128", *extra), "llama_bench")


def main(run: Runner = subprocess_runner) -> None:
    model = os.environ["FARADAY_OPT_MODEL"]      # set by scripts/90_optimize.sh
    draft = os.environ.get("FARADAY_OPT_DRAFT", "")
    ollama_model = os.environ.get("FARADAY_OPT_OLLAMA", "qwen2.5:1.5b")
    prompt = "Summarize the history of crewed spaceflight in three sentences."
    all_cells = optimize_config.cells()

    done = read_done(CSV_PATH)
    for cell in all_cells:
        if cell.key in done:
            continue
        print(f"--- {cell.component}/{cell.label} ---", flush=True)
        row = run_cell(cell, run, model=model, draft=draft,
                       ollama_model=ollama_model, prompt=prompt, raw_dir=RAW_DIR)
        append_row(CSV_PATH, row)
        print(f"    decode={row['decode_tps']} throttled={row['throttled']}", flush=True)

    best = stack_winners(all_cells, load_rows(CSV_PATH))
    if best.key not in read_done(CSV_PATH):
        print(f"--- {best.component} (gov={best.governor} {' '.join(best.flags)}) ---", flush=True)
        append_row(CSV_PATH, run_cell(best, run, model=model, draft=draft,
                                      ollama_model=ollama_model, prompt=prompt, raw_dir=RAW_DIR))
    # restore the default governor
    run(_governor_cmd("ondemand"))


if __name__ == "__main__":
    main()
```

> The `# noqa: E402` markers are because these imports are appended below code in this incremental task; **Task 6 (lint) consolidates all imports to the top of `optimize.py`** and removes the noqa. (The M3 E402 lesson — do it in one cleanup commit.)

- [ ] **Step 4: Run — expect PASS** (2 passed). **Step 5: ruff** (will flag E402 — that's expected; Task 6 fixes). Just run the test green here, commit:

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_optimize_plan.py -q"
git add src/faraday/bench/optimize.py tests/test_optimize_plan.py
git commit -m "feat(m4c): ablate-then-stack planner + optimize main"
```

---

### Task 5: `optimize_plot.py` — waterfall + lever bars + context curve (TDD)

**Files:**
- Create: `src/faraday/bench/optimize_plot.py`
- Test: `tests/test_optimize_plot.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_optimize_plot.py`:

```python
from faraday.bench.optimize_plot import (
    lever_gains,
    make_leaderboard,
    render_context_curve,
    render_waterfall,
)


def _row(component, label, decode, prefill="5"):
    return {"component": component, "label": label, "prefill_tps": prefill,
            "decode_tps": str(decode), "peak_rss_bytes": "1", "accept_rate": "",
            "throttled": "throttled=0x0", "notes": "ok"}


def test_lever_gains_are_percent_over_baseline():
    rows = [_row("baseline", "baseline", 4.0),
            _row("governor", "governor=performance", 5.0),
            _row("threads", "threads=3", 4.4)]
    gains = lever_gains(rows)
    assert gains["governor=performance"] == 25.0   # +25%
    assert gains["threads=3"] == 10.0              # +10%


def test_make_leaderboard_sorts_by_decode_desc():
    rows = [_row("baseline", "baseline", 4.0),
            _row("stacked_best", "stacked_best", 6.0),
            _row("ollama", "ollama-default", 3.5)]
    md = make_leaderboard(rows)
    assert md.index("stacked_best") < md.index("baseline") < md.index("ollama-default")
    assert "decode" in md.lower()


def test_render_waterfall_and_context_write_pngs(tmp_path):
    rows = [_row("baseline", "baseline", 4.0),
            _row("governor", "governor=performance", 5.0),
            _row("stacked_best", "stacked_best", 6.0),
            _row("context", "ctx=512", 4.5, prefill="9.0"),
            _row("context", "ctx=2048", 4.0, prefill="7.0")]
    w = tmp_path / "waterfall.png"
    render_waterfall(rows, w)
    assert w.exists() and w.stat().st_size > 0
    c = tmp_path / "context.png"
    render_context_curve(rows, c)
    assert c.exists() and c.stat().st_size > 0
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement.** Create `src/faraday/bench/optimize_plot.py`:

```python
"""Render the optimization CSV into deliverables: per-lever gains, the baseline->best
waterfall, the TTFT/decode-vs-context curve, and a leaderboard. Pure helpers tested;
rendering is a smoke test. Reuses the headless-Agg pattern from M4a's plot.py.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


def load_rows(csv_path: Path) -> list[dict]:
    with Path(csv_path).open(newline="") as f:
        return list(csv.DictReader(f))


def _by_label(rows: list[dict]) -> dict[str, dict]:
    return {r["label"]: r for r in rows}


def lever_gains(rows: list[dict]) -> dict[str, float]:
    """label -> percent decode gain over baseline, for the single-lever cells."""
    base = float(_by_label(rows)["baseline"]["decode_tps"])
    out: dict[str, float] = {}
    for r in rows:
        if r["component"] in ("governor", "threads", "batch", "kvquant", "flashattn") \
                and r["decode_tps"]:
            out[r["label"]] = round((float(r["decode_tps"]) - base) / base * 100, 4)
    return out


def make_leaderboard(rows: list[dict]) -> str:
    ranked = sorted((r for r in rows if r["decode_tps"]),
                    key=lambda r: float(r["decode_tps"]), reverse=True)
    lines = ["# Faraday M4c — Optimization Leaderboard", "",
             "Sorted by decode tok/s (higher = better).", "",
             "| Rank | Cell | decode t/s | prefill t/s | accept % | throttled |",
             "|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        lines.append(f"| {i} | {r['label']} | {r['decode_tps']} | {r['prefill_tps']} | "
                     f"{r['accept_rate'] or '-'} | {r['throttled']} |")
    return "\n".join(lines) + "\n"


def render_waterfall(rows: list[dict], out_path: Path) -> None:
    bl = _by_label(rows)
    labels, values = ["baseline"], [float(bl["baseline"]["decode_tps"])]
    for r in rows:
        if r["component"] == "stacked_best" and r["decode_tps"]:
            labels.append("stacked_best")
            values.append(float(r["decode_tps"]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), values, color=["#888", "#2a9d8f"][:len(labels)])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("decode tok/s")
    ax.set_title("Faraday M4c — baseline → best-tuned decode throughput")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_context_curve(rows: list[dict], out_path: Path) -> None:
    pts = sorted((int(r["label"].split("=")[1]), float(r["prefill_tps"] or "nan"),
                  float(r["decode_tps"] or "nan"))
                 for r in rows if r["component"] == "context")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if pts:
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], marker="o", label="prefill t/s")
        ax.plot(xs, [p[2] for p in pts], marker="s", label="decode t/s")
        ax.legend()
    ax.set_xlabel("context size (prompt tokens)")
    ax.set_ylabel("tok/s")
    ax.set_title("Faraday M4c — throughput vs context length")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_lever_bars(rows: list[dict], out_path: Path) -> None:
    gains = lever_gains(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(gains)
    ax.bar(range(len(labels)), [gains[k] for k in labels])
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("decode gain vs baseline (%)")
    ax.set_title("Faraday M4c — per-lever marginal gain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    from faraday.bench.optimize_config import CSV_PATH, RESULTS_DIR
    rows = load_rows(CSV_PATH)
    render_waterfall(rows, RESULTS_DIR / "waterfall.png")
    render_lever_bars(rows, RESULTS_DIR / "lever_gains.png")
    render_context_curve(rows, RESULTS_DIR / "context_curve.png")
    (RESULTS_DIR / "leaderboard.md").write_text(make_leaderboard(rows))
    print(f"wrote waterfall/lever_gains/context_curve + leaderboard ({len(rows)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run — expect PASS** (3 passed). **Step 5: ruff + commit.**

```bash
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest tests/test_optimize_plot.py -q && ruff check src/faraday/bench/optimize_plot.py tests/test_optimize_plot.py"
git add src/faraday/bench/optimize_plot.py tests/test_optimize_plot.py
git commit -m "feat(m4c): optimization plots (waterfall + lever bars + context curve)"
```

---

### Task 6: Consolidate imports + full-suite + lint

**Files:** Modify `src/faraday/bench/optimize.py` (imports to top)

- [ ] **Step 1: Move the Task-4 appended imports to the top of `optimize.py`.** The file's import block should become exactly:

```python
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

from faraday.bench import optimize_config, parsers
from faraday.bench.optimize_config import CSV_COLUMNS, CSV_PATH, RAW_DIR, LeverCell
from faraday.bench.sweep import Completed, Runner, subprocess_runner
```

Delete the later `import os` / `from collections import defaultdict` / `from faraday.bench import optimize_config` / `from faraday.bench.optimize_config import CSV_PATH, RAW_DIR` lines (now at top) and their `# noqa: E402`.

- [ ] **Step 2: Full suite on the Pi.** Run:

`ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && pytest -q"`
Expected: all prior + the new ~18 M4c tests pass; integration deselected.

- [ ] **Step 3: Full lint.** Run:

`ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && ruff check src tests"`
Expected: `All checks passed!`

- [ ] **Step 4: Commit.**

```bash
git add src/faraday/bench/optimize.py
git commit -m "style(m4c): consolidate optimize.py imports to top (E402)"
```

---

### Task 7: `scripts/90_optimize.sh` — on-Pi runner + OC procedure

**Files:** Create `scripts/90_optimize.sh`

- [ ] **Step 1: Write the script.** Create `scripts/90_optimize.sh`:

```bash
#!/usr/bin/env bash
# Run ON the Raspberry Pi, on a QUIET board (nothing else competing). Drives the
# M4c optimization sweep on 1.5B Q4_K_M: ablate each tuning lever vs baseline,
# stack the winners, measure speculative decoding + an Ollama baseline + the
# context curve. Resumable (re-running skips done cells). Records throttle per cell.
#
# Overclock is a SEPARATE step (see bottom) — it needs a reboot, so it is not part
# of this runner; run it after, then re-run to fill the OC rows.
set -euo pipefail
cd "$HOME/faraday"
# shellcheck disable=SC1091
source .venv/bin/activate
export PATH="$HOME/llama.cpp/build/bin:$PATH"

# 1) GNU time (peak RSS) — same prereq as M4a.
if ! /usr/bin/time -v true >/dev/null 2>&1; then sudo apt-get install -y time; fi

# 2) llama-speculative must be built (the Pi build lacks it — like llama-perplexity).
if [[ ! -x "$HOME/llama.cpp/build/bin/llama-speculative" ]]; then
  echo "Building llama-speculative (-j3, 4GB-safe)..."
  cmake --build "$HOME/llama.cpp/build" --target llama-speculative -j3
fi

# 3) Models: the 1.5B Q4_K_M target + a 0.5B draft (same Qwen2.5 family = shared tokenizer).
export FARADAY_OPT_MODEL="$(ls "$HOME"/faraday/models/*q4_k_m.gguf | head -1)"
DRAFT="$(ls "$HOME"/faraday/models/*0.5B*q4_k_m.gguf 2>/dev/null | head -1 || true)"
if [[ -z "$DRAFT" ]]; then
  echo "Fetching 0.5B draft model..."
  hf download bartowski/Qwen2.5-0.5B-Instruct-GGUF Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
    --local-dir "$HOME/faraday/models"
  DRAFT="$(ls "$HOME"/faraday/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf)"
fi
export FARADAY_OPT_DRAFT="$DRAFT"

# 4) Ollama baseline.
if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama..."; curl -fsSL https://ollama.com/install.sh | sh
fi
ollama pull qwen2.5:1.5b
export FARADAY_OPT_OLLAMA="qwen2.5:1.5b"

# 5) Health gate + run + plot.
echo "throttle (0x0 = healthy): $(vcgencmd get_throttled)"
python -m faraday.bench.optimize
python -m faraday.bench.optimize_plot
echo "Done. Commit results/optimize/{optimize.csv,waterfall.png,lever_gains.png,context_curve.png,leaderboard.md}."

# --- OVERCLOCK (separate, manual; needs a reboot) -----------------------------
# To add the overclock rows after the stock-clock sweep:
#   1) sudo sh -c 'printf "\n[all]\narm_freq=2000\nover_voltage=6\n" >> /boot/firmware/config.txt'
#   2) sudo reboot   (re-run scripts/30_run_servers.sh etc. after boot if needed)
#   3) Verify cooling + PSU: watch `vcgencmd measure_temp` / `get_throttled` under load.
#   4) Re-run this script — it appends OC-clock rows for the still-pending cells.
#   5) Revert: remove those lines from config.txt + reboot.
```

- [ ] **Step 2: Make executable + commit.**

```bash
git update-index --add --chmod=+x scripts/90_optimize.sh
git add scripts/90_optimize.sh
git commit -m "feat(m4c): on-Pi optimization runner + documented overclock procedure"
```

---

### Task 8: Integration smoke (Pi-only, run when the board is quiet)

**Files:** Test `tests/test_optimize_integration.py`

- [ ] **Step 1: Write the integration test.** Create `tests/test_optimize_integration.py`:

```python
"""Real 1-cell optimization smoke. Deselected by default (`integration`); run on a
QUIET Pi with binaries on PATH + the model present:
  export PATH="$HOME/llama.cpp/build/bin:$PATH"
  pytest -m integration tests/test_optimize_integration.py -q
"""
import os
from pathlib import Path

import pytest

from faraday.bench.optimize import run_cell
from faraday.bench.optimize_config import LeverCell
from faraday.bench.sweep import subprocess_runner


@pytest.mark.integration
def test_baseline_cell_end_to_end(tmp_path):
    model = next(Path(os.path.expanduser("~/faraday/models")).glob("*q4_k_m.gguf"))
    cell = LeverCell("baseline", "baseline", "ondemand", ("-p", "128", "-n", "32"), "llama_bench")
    row = run_cell(cell, subprocess_runner, model=str(model), draft="", ollama_model="",
                   prompt="hi", raw_dir=tmp_path)
    assert row["notes"] == "ok", row["notes"]
    assert float(row["decode_tps"]) > 0
    assert int(row["peak_rss_bytes"]) > 0
    assert "0x" in row["throttled"]
```

- [ ] **Step 2: Commit (inert until run with `-m integration`).**

```bash
git add tests/test_optimize_integration.py
git commit -m "test(m4c): 1-cell optimization integration smoke (Pi-only)"
```

- [ ] **Step 3: When the board is quiet — smoke, then the full run** (after M4a sweep + M4b run):

```bash
git push pi <branch>
ssh pi@raspberrypi.local "cd ~/faraday && . .venv/bin/activate && export PATH=\$HOME/llama.cpp/build/bin:\$PATH && pytest -m integration tests/test_optimize_integration.py -q"
ssh pi@raspberrypi.local "cd ~/faraday && setsid nohup bash scripts/90_optimize.sh >/tmp/optimize.log 2>&1 </dev/null & echo started"
```

---

### Task 9: Results scaffold + README

**Files:** Create `results/optimize/README.md`; modify `.gitignore`

- [ ] **Step 1: Ignore raw logs, keep curated.** Add to `.gitignore` (after the M4b eval block):

```
# M4c optimize: ignore raw tool logs; keep optimize.csv + plots + leaderboard + findings.
results/optimize/raw/
```

- [ ] **Step 2: Write the results README.** Create `results/optimize/README.md`:

```markdown
# M4c — Inference Optimization Results

Throughput tuning of the shipped model (Qwen2.5-1.5B Q4_K_M) on a Raspberry Pi 4
(4 GB), via `scripts/90_optimize.sh` → `python -m faraday.bench.optimize` (+ `…optimize_plot`).

| File | What |
|---|---|
| `optimize.csv` | One row per config: prefill/decode tok/s, peak RSS, accept rate, throttle, notes. |
| `lever_gains.png` | Per-lever marginal decode gain (%) vs baseline. |
| `waterfall.png` | Baseline → best-tuned (stacked winners) decode throughput. |
| `context_curve.png` | Prefill/decode tok/s vs context length (motivates KV-cache quant). |
| `leaderboard.md` | All configs ranked by decode tok/s (incl. speculative + Ollama). |
| `findings.md` | The narrative: which levers won, the stacked speedup, speculative's CPU verdict, the Ollama delta. |
| `raw/` | Per-cell raw tool output (git-ignored). |

## Method
Ablate-then-stack: each lever measured independently vs a fixed baseline, winners
(clean `throttled=0x0` only) stacked into a best config. Speculative decoding (0.5B
draft + 1.5B target) and an Ollama default are compared to best-tuned. Overclock is a
separate reboot procedure (see `scripts/90_optimize.sh`). Run on a quiet board.

## Reproduce
`bash scripts/90_optimize.sh` on the Pi, then commit the curated outputs from the dev box.
```

- [ ] **Step 3: Commit.**

```bash
git add .gitignore results/optimize/README.md
git commit -m "docs(m4c): results/optimize scaffold + artifact guide"
```

> The curated `optimize.csv` / PNGs / `leaderboard.md` + a written `findings.md` are committed **after** the run (the milestone closeout), like M4a.

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
|---|---|
| §5 lever matrix (governor/threads/batch/KV-quant/flash-attn/overclock) | Task 1 (cells) + Task 7 (overclock procedure) |
| §2 ablate-then-stack | Task 4 (`stack_winners`) |
| §8 speculative (build + 0.5B draft + acceptance) | Task 2 (`parse_speculative`), 3 (`build_argv`), 7 (build+draft fetch) |
| §8 Ollama baseline (managed-vs-tuned) | Task 2 (`parse_ollama_bench`), 3, 7 (install+pull) |
| §8 characterization (TTFT/decode vs context) | Task 1 (`context` cells), 5 (`render_context_curve`) |
| §7 components config/runner/plot/parsers/script/results | Tasks 1/3-4/5/2/7/9 |
| §9 CSV schema | Task 1 `CSV_COLUMNS` + Task 3 writers |
| §10 throttle hygiene (flag + exclude throttled) | Task 3 (record `throttled`), Task 4 (`_is_clean` excludes) |
| §11 error handling (unsupported/oom/parse → notes; raw logs) | Task 3 (`run_cell` try/except, `_save_raw`) |
| §12 testing (config/parsers/runner/planner/plot pure; 1-cell integration) | Tasks 1/2/3/4/5 + Task 8 |
| §6 overclock = separate reboot procedure | Task 7 script footer |
| §13 file structure | Tasks 1-9 create exactly the listed files |
| §14 definition of done | Task 8 (run) + post-run closeout |

**Gap noted & resolved:** the spec's `optimize_plot` also lists a per-lever bar chart — covered by `render_lever_bars` (Task 5, tested via `lever_gains`). The 6th metric column `prefill_tps` for ollama/speculative may be blank — handled (leaderboard prints `-`/value as present).

**2. Placeholder scan:** No TBD/TODO. `<branch>` in Task 8 is a user-supplied execution value. The parser samples (Task 2) are concrete with real assertions; the note about confirming against live output is honest TDD (raw logs make it a 2-min regex tweak), not a placeholder. The Task-4 `# noqa: E402` is explicitly resolved in Task 6.

**3. Type consistency:** `LeverCell(component,label,governor,flags,kind)` + `.key` consistent across Tasks 1/3/4/8. `CSV_COLUMNS` (8 fields) consistent config↔runner↔plot↔tests. `run_cell(cell, run, *, model, draft, ollama_model, prompt, raw_dir)` signature identical across Task 3 impl, Task 4 `main`, Task 8 integration. `build_argv(cell, *, model, draft, ollama_model, prompt)` consistent. `stack_winners(all_cells, rows) -> LeverCell` matches its call in `main`. `parse_ollama_bench→(prefill,decode)`, `parse_speculative→(decode,accept)` match every call site. Reused `Completed`/`Runner`/`subprocess_runner`/`parse_llama_bench`/`parse_time_v` match the verified `faraday.bench` signatures.

**Verdict:** complete, spec-covering, placeholder-free.
