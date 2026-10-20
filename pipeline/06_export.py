"""PLAN_V5_REVISION.md V5-7, pipeline step 6 (renumbered from step 5 by
PLAN_V6_REVISION.md V6-4's new 03_select_blend.py).

Thin wrapper: prints what it will run and where the output lands, then
calls scripts/export_runs.py. No logic is duplicated here.

    python pipeline/06_export.py [--dry-run]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running anything")
    args = parser.parse_args(argv)

    cmd = [sys.executable, "scripts/export_runs.py"]
    print(f"[06_export] {' '.join(cmd)}")
    print(f"  cwd:    {REPO_ROOT}")
    print(f"  reads:  {REPO_ROOT / 'results'}")
    print(f"  writes: {REPO_ROOT / 'results' / 'run_catalogue.csv'}, {REPO_ROOT / 'results' / 'skip_log_all.csv'}")

    if args.dry_run:
        print("--dry-run: not running.")
        return

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
