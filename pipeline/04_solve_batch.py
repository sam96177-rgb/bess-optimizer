"""PLAN_V5_REVISION.md V5-7, pipeline step 4 (renumbered from step 3 by
PLAN_V6_REVISION.md V6-4's new 03_select_blend.py).

Thin wrapper: prints what it will run and where the output lands, then
calls `python -m src.batch` in a subprocess using THIS repo's own venv. No
--strategies/--models are passed: src.batch already defaults to
config/study.yaml's strategies x (models + blends) (V5-6), so this
wrapper doesn't restate it.

    python pipeline/04_solve_batch.py [--dry-run]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_study_config  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running anything")
    args = parser.parse_args(argv)

    study = load_study_config()
    cmd = [sys.executable, "-m", "src.batch"]

    print(f"[04_solve_batch] {' '.join(cmd)}")
    print(f"  cwd:        {REPO_ROOT}")
    print(f"  study:      {study.name}")
    print(f"  strategies: {study.strategies}")
    print(f"  models:     {study.optimizer_models}")
    print(f"  writes:     {REPO_ROOT / 'results'}/<solve_id>/")

    if args.dry_run:
        print("--dry-run: not running.")
        return

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
