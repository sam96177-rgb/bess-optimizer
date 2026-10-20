"""PLAN_V5_REVISION.md V5-7, pipeline step 5 (renumbered from step 4 by
PLAN_V6_REVISION.md V6-4's new 03_select_blend.py).

Thin wrapper: prints what it will run, then calls
scripts/lookahead_value.py over the study's derived window (the
intersection of the four markets' test windows -- see StudyConfig.
optimization_window in src/config.py), with --skip-chained (config 2
doesn't depend on the lookahead window length -- V5-1). Output is teed to
both stdout and a timestamped file under logs/, since a ~164-day run is
long enough to want a saved transcript.

    python pipeline/05_lookahead.py [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
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
    start, end = study.optimization_window
    cmd = [
        sys.executable, "scripts/lookahead_value.py",
        "--start-date", start.isoformat(), "--end-date", end.isoformat(),
        "--skip-chained",
    ]
    log_path = REPO_ROOT / "logs" / f"05_lookahead_{dt.datetime.now():%Y%m%d_%H%M%S}.log"

    print(f"[05_lookahead] {' '.join(cmd)}")
    print(f"  cwd:    {REPO_ROOT}")
    print(f"  reads:  {REPO_ROOT / 'results'} (baseline lookup)")
    print(f"  writes: stdout + {log_path}")

    if args.dry_run:
        print("--dry-run: not running.")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file, subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        returncode = proc.wait()
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
