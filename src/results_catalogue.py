"""One discovery path for solved runs (SPEC.md §6a, Milestone 8 Part B),
shared by the CLI, batch.py, and the dashboard so they never disagree
about what's already been solved. A run directory with parquets but no
run_meta.json (crashed mid-write) is invisible here by design --
src/model/rolling.py::write_rolling_run_results writes run_meta.json
last for exactly this reason.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def scan_runs(results_dir) -> pd.DataFrame:
    """One row per results/<solve_id>/run_meta.json. Nested fields
    (battery, solver, model_flags, assumptions, price_basis,
    forecast_run_ids) are flattened to dotted column names (e.g.
    'battery.power_mw') via pd.json_normalize, so callers can filter or
    fingerprint-compare on them directly without re-parsing JSON.

    Returns an empty DataFrame (no columns) if results_dir doesn't exist
    or has no completed runs yet -- callers follow this repo's existing
    `if df.empty: ...` convention rather than getting a raised error for
    what is often just "nothing solved yet".
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return pd.DataFrame()

    meta_paths = sorted(results_dir.glob("*/run_meta.json"))
    if not meta_paths:
        return pd.DataFrame()

    metas = []
    for meta_path in meta_paths:
        with meta_path.open("r", encoding="utf-8") as f:
            metas.append(json.load(f))
    return pd.json_normalize(metas)


SKIP_LOG_COLUMNS = ["solve_id", "strategy", "decision_source", "forecast_model", "day", "reason"]


def scan_skip_logs(results_dir) -> pd.DataFrame:
    """Every run's skipped days in one frame, labelled with the run that
    skipped them (solve_id, strategy, decision_source, forecast_model).
    A run with no skipped days contributes no rows. A run written before
    skip_log.csv had a header row may leave a column-less empty file; that
    counts as no skipped days, same as the dashboard's loader.
    """
    runs = scan_runs(results_dir)
    if runs.empty:
        return pd.DataFrame(columns=SKIP_LOG_COLUMNS)

    frames = []
    for run in runs.itertuples():
        path = Path(results_dir) / run.solve_id / "skip_log.csv"
        if not path.exists():
            continue
        try:
            skipped = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if skipped.empty:
            continue
        skipped.insert(0, "forecast_model", run.forecast_model)
        skipped.insert(0, "decision_source", run.decision_source)
        skipped.insert(0, "strategy", run.strategy)
        skipped.insert(0, "solve_id", run.solve_id)
        frames.append(skipped)
    if not frames:
        return pd.DataFrame(columns=SKIP_LOG_COLUMNS)
    return pd.concat(frames, ignore_index=True)[SKIP_LOG_COLUMNS]
