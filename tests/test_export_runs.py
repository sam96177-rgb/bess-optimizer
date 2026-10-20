"""scripts/export_runs.py and the skip-log collection behind it."""
import importlib.util
import json
from pathlib import Path

import pandas as pd

from src.results_catalogue import SKIP_LOG_COLUMNS, scan_skip_logs

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_runs.py"
_spec = importlib.util.spec_from_file_location("export_runs", _SCRIPT)
export_runs_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_runs_module)


def _write_run(results_dir: Path, solve_id: str, strategy: str, model, skip_csv: str | None):
    run_dir = results_dir / solve_id
    run_dir.mkdir(parents=True)
    meta = {
        "solve_id": solve_id, "strategy": strategy,
        "decision_source": "forecast" if model else "realized", "forecast_model": model,
        "start_date": "2026-06-01", "end_date": "2026-06-03", "battery": {"power_mw": 4.5},
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if skip_csv is not None:
        (run_dir / "skip_log.csv").write_text(skip_csv, encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    _write_run(tmp_path, "run-a", "day_ahead_only", None, "day,reason\n")  # header only: nothing skipped
    _write_run(tmp_path, "run-b", "full_cooptimized", "seasonal_naive", "day,reason\n2026-06-02,fcr: missing\n2026-06-03,fcr: missing\n")
    _write_run(tmp_path, "run-c", "full_cooptimized", "lgbm_point", "")  # legacy column-less empty file
    _write_run(tmp_path, "run-d", "full_cooptimized", "analog_day", None)  # no skip_log.csv at all
    return tmp_path


def test_scan_skip_logs_labels_each_skipped_day_with_its_run(tmp_path):
    skips = scan_skip_logs(_fixture(tmp_path))
    assert list(skips.columns) == SKIP_LOG_COLUMNS
    assert len(skips) == 2
    assert set(skips["solve_id"]) == {"run-b"}
    assert set(skips["forecast_model"]) == {"seasonal_naive"}
    assert list(skips["day"]) == ["2026-06-02", "2026-06-03"]


def test_scan_skip_logs_on_missing_or_empty_results_dir(tmp_path):
    assert scan_skip_logs(tmp_path / "nope").empty
    assert list(scan_skip_logs(tmp_path).columns) == SKIP_LOG_COLUMNS


def test_export_writes_both_csvs_with_one_row_per_run(tmp_path):
    results = _fixture(tmp_path / "results")
    out = tmp_path / "out"
    catalogue_path, skip_path, n_runs, n_skips = export_runs_module.export_runs(results, out)

    assert (n_runs, n_skips) == (4, 2)
    catalogue = pd.read_csv(catalogue_path)
    assert len(catalogue) == 4 and set(catalogue["solve_id"]) == {"run-a", "run-b", "run-c", "run-d"}
    assert "battery.power_mw" in catalogue.columns  # nested fields are flattened
    assert len(pd.read_csv(skip_path)) == 2


def test_export_on_an_empty_results_dir_still_writes_both_files(tmp_path):
    catalogue_path, skip_path, n_runs, n_skips = export_runs_module.export_runs(tmp_path / "empty", tmp_path / "out")
    assert (n_runs, n_skips) == (0, 0)
    assert catalogue_path.exists() and skip_path.exists()
