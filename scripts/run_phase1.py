#!/usr/bin/env python3
"""
Phase 1 launcher: L0 generation (only if needed) + L1 filtering.

If real data already exists at data/l0_raw/l0_real_sample.jsonl,
skips substitute generation — L1 will use the real data via l1_expanded.yaml.

Usage:
    PYTHONPATH=. python scripts/run_phase1.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

REAL_DATA_PATH = _PROJECT_ROOT / "data/l0_raw/l0_real_sample.jsonl"
L1_CONFIG = (_PROJECT_ROOT / "configs/l1_expanded.yaml").as_posix()


def main() -> None:
    if REAL_DATA_PATH.exists() and REAL_DATA_PATH.stat().st_size > 0:
        print(
            f"[Phase 1] Real data found at {REAL_DATA_PATH.as_posix()} "
            f"({REAL_DATA_PATH.stat().st_size // 1024} KB) — skipping substitute generation."
        )
    else:
        print("[Phase 1] No real data found — generating local substitute data...")
        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts/generate_l0_expanded.py")],
            cwd=str(_PROJECT_ROOT),
            check=False,
        )
        if result.returncode != 0:
            print("[Phase 1] Warning: substitute generation failed.")

    config_path = Path(L1_CONFIG)
    if not config_path.exists():
        print(f"[Phase 1] Error: L1 configuration file not found at {config_path.as_posix()}.")
        sys.exit(1)

    print(f"[Phase 1] Running L1 filtering with config: {L1_CONFIG}")
    result = subprocess.run(
        [sys.executable, str(_PROJECT_ROOT / "scripts/run_l1.py"), "--config", L1_CONFIG],
        cwd=str(_PROJECT_ROOT),
        check=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
