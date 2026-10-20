"""PLAN_V5_REVISION.md V5-7, pipeline step 2.

Thin wrapper: prints what it will run and where the output lands, then
calls bess-market-forecast/scripts/run_backtest.py in a subprocess using
that repo's own venv interpreter. No --markets/--models are passed: that
script already defaults to config/study.yaml's markets and models (V5-6),
i.e. study markets x models, so this wrapper doesn't restate it.

    python pipeline/02_backtest_forecasts.py [--dry-run]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_paths_config, load_study_config  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running anything")
    args = parser.parse_args(argv)

    paths = load_paths_config()
    if paths.forecast_venv_python is None:
        raise SystemExit(
            "config/paths.yaml has no forecast_venv_python -- point it at the "
            "sibling bess-market-forecast repo's own venv interpreter."
        )
    study = load_study_config(paths)

    cmd = [str(paths.forecast_venv_python), "scripts/run_backtest.py"]
    print(f"[02_backtest_forecasts] {' '.join(cmd)}")
    print(f"  cwd:      {paths.forecast_repo_path}")
    print(f"  study:    {study.name}  ({len(study.models)} models x 4 markets)")
    print(f"  writes:   {paths.forecast_repo_path / 'results'}/*.parquet")

    if args.dry_run:
        print("--dry-run: not running.")
        return

    subprocess.run(cmd, cwd=paths.forecast_repo_path, check=True)


if __name__ == "__main__":
    main()
