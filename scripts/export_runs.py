"""Export what is in results/ as two CSVs, for looking at in a spreadsheet
(the dashboard's Run Catalogue page was replaced by this):

  results/run_catalogue.csv   one row per solved run (every run_meta.json field,
                              nested fields flattened to dotted column names)
  results/skip_log_all.csv    every day any run skipped, with the run it belongs to

    python scripts/export_runs.py [--results-dir results] [--out-dir results]

Pure CSV, no extra dependency. Every run is exported, including strategies the
dashboard hides.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT  # noqa: E402
from src.results_catalogue import scan_runs, scan_skip_logs  # noqa: E402


def export_runs(results_dir: Path, out_dir: Path) -> tuple[Path, Path, int, int]:
    """Writes both CSVs; returns (catalogue_path, skip_log_path, n_runs, n_skipped_days)."""
    runs = scan_runs(results_dir)
    if not runs.empty:
        runs = runs.sort_values(["strategy", "decision_source", "forecast_model", "start_date"], na_position="first")
    skips = scan_skip_logs(results_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    catalogue_path = out_dir / "run_catalogue.csv"
    skip_path = out_dir / "skip_log_all.csv"
    runs.to_csv(catalogue_path, index=False)
    skips.to_csv(skip_path, index=False)
    return catalogue_path, skip_path, len(runs), len(skips)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: the results dir.")
    args = parser.parse_args(argv)

    catalogue_path, skip_path, n_runs, n_skips = export_runs(args.results_dir, args.out_dir or args.results_dir)
    print(f"{n_runs} run(s)          -> {catalogue_path}")
    print(f"{n_skips} skipped day(s) -> {skip_path}")


if __name__ == "__main__":
    main()
