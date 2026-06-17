"""Startup memory guard (systemd ExecStartPre on the gen unit).

Checks the configured GGUF + headroom against MemAvailable. Fit -> write the model
path to /run/faraday/model.env (read by the unit's ExecStart via EnvironmentFile=)
and exit 0. No fit but a smaller gen GGUF exists -> fall back to it (logged). Nothing
fits -> exit 1 loudly, so the unit fails visibly instead of OOM-ing the board.

Pure decision logic (fits/pick_model) is unit-tested; only the /proc/meminfo read
and the env-file write are Pi-specific.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# KV cache at -c 4096 + compute buffers + slack, sized from M4a peak-RSS-vs-file-size
# deltas. Conservative on purpose: refusing too early beats OOM-ing the 4 GB board.
HEADROOM_BYTES = 700 * 1024 * 1024

MODELS_DIR = Path.home() / "faraday" / "models"
ENV_FILE = Path("/run/faraday/model.env")


def fits(model_bytes: int, available_bytes: int,
         headroom_bytes: int = HEADROOM_BYTES) -> bool:
    return model_bytes + headroom_bytes <= available_bytes


def pick_model(candidates: list[tuple[Path, int]], available_bytes: int,
               headroom_bytes: int = HEADROOM_BYTES) -> Path | None:
    """Largest candidate that fits (candidates need not be sorted)."""
    fitting = [(p, s) for p, s in candidates if fits(s, available_bytes, headroom_bytes)]
    if not fitting:
        return None
    return max(fitting, key=lambda t: t[1])[0]


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def _gen_candidates() -> list[tuple[Path, int]]:
    # All gen-model GGUFs (exclude the bge embedding model), any quant present.
    return [(p, p.stat().st_size) for p in MODELS_DIR.glob("*.gguf")
            if "bge" not in p.name.lower()]


def main() -> int:
    configured = os.environ.get("FARADAY_GEN_MODEL", "")
    avail = mem_available_bytes()
    if configured and Path(configured).exists() \
            and fits(Path(configured).stat().st_size, avail):
        chosen = Path(configured)
    else:
        chosen = pick_model(_gen_candidates(), avail)
        if chosen is None:
            print(f"preflight: NO model fits (MemAvailable={avail // 2**20} MiB, "
                  f"headroom={HEADROOM_BYTES // 2**20} MiB) — refusing to start",
                  file=sys.stderr)
            return 1
        if configured:
            print(f"preflight: '{configured}' does not fit — falling back to {chosen}")
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(f"FARADAY_GEN_MODEL={chosen}\n")
    print(f"preflight: ok — {chosen.name} (MemAvailable={avail // 2**20} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
