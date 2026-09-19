"""src.batch (SPEC.md §6a / Milestone 8 Part B): the matrix builder, the
resumability fingerprint match, and an end-to-end resume/--force check
against real (tiny) solves.
"""
import datetime as dt

import pandas as pd
import pytest

from src.batch import ALL_STRATEGIES, DEFAULT_MODELS, DEFAULT_QUANTILES, _already_solved, _fingerprint, build_matrix, run_batch
from src.config import load_assumptions_config, load_battery_config, load_paths_config, load_solver_config
from src.model.battery import battery_from_config
from src.results_catalogue import scan_runs

PATHS = load_paths_config()
BATTERY = battery_from_config(load_battery_config())
SOLVER = load_solver_config()
ASSUMPTIONS = load_assumptions_config()
FINGERPRINT = _fingerprint(BATTERY, SOLVER, ASSUMPTIONS)


# ---- build_matrix: hand-verified counts -----------------------------------

def test_build_matrix_default_shape_is_18_combinations():
    combos = build_matrix(DEFAULT_MODELS, DEFAULT_QUANTILES, ALL_STRATEGIES, include_perfect_foresight=True)
    assert len(combos) == 18
    realized = [c for c in combos if c[1] == "realized"]
    forecast = [c for c in combos if c[1] == "forecast"]
    assert len(realized) == 3  # one per strategy
    assert len(forecast) == 15  # 5 models x 1 quantile x 3 strategies
    assert all(c[2] is None and c[3] is None for c in realized)
    assert all(c[2] is not None and c[3] is not None for c in forecast)


def test_build_matrix_without_perfect_foresight_drops_the_realized_combos():
    combos = build_matrix(DEFAULT_MODELS, DEFAULT_QUANTILES, ALL_STRATEGIES, include_perfect_foresight=False)
    assert len(combos) == 15
    assert all(c[1] == "forecast" for c in combos)


def test_build_matrix_scales_with_models_x_quantiles_x_strategies():
    combos = build_matrix(["m1", "m2"], [0.1, 0.5, 0.9], ["naive"], include_perfect_foresight=True)
    assert len(combos) == 1 + 2 * 3  # 1 perfect-foresight + 2 models x 3 quantiles


# ---- _already_solved: hand-built synthetic catalogue -----------------------

def _catalogue_row(strategy, decision_source, forecast_model, forecast_quantile, start_date, end_date, fingerprint):
    row = {
        "strategy": strategy, "decision_source": decision_source,
        "forecast_model": forecast_model, "forecast_quantile": forecast_quantile,
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
    }
    row.update(fingerprint)
    return row


def test_already_solved_true_for_exact_match():
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    catalogue = pd.DataFrame([_catalogue_row("naive", "realized", None, None, start, end, FINGERPRINT)])
    assert _already_solved(catalogue, "naive", "realized", None, None, start, end, FINGERPRINT) is True


def test_already_solved_true_for_matching_forecast_combo():
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    catalogue = pd.DataFrame(
        [_catalogue_row("full_cooptimized", "forecast", "lgbm_point", 0.5, start, end, FINGERPRINT)]
    )
    assert _already_solved(catalogue, "full_cooptimized", "forecast", "lgbm_point", 0.5, start, end, FINGERPRINT) is True


@pytest.mark.parametrize(
    "field,new_value",
    [("strategy", "naive"), ("decision_source", "realized"), ("forecast_model", "other_model"),
     ("start_date", dt.date(2026, 1, 1).isoformat()), ("end_date", dt.date(2099, 1, 1).isoformat())],
)
def test_already_solved_false_when_a_key_field_differs(field, new_value):
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    row = _catalogue_row("day_ahead_only", "forecast", "lgbm_point", 0.5, start, end, FINGERPRINT)
    row[field] = new_value
    catalogue = pd.DataFrame([row])
    assert _already_solved(catalogue, "day_ahead_only", "forecast", "lgbm_point", 0.5, start, end, FINGERPRINT) is False


def test_already_solved_false_when_fingerprint_mismatches():
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    stale_fingerprint = dict(FINGERPRINT)
    stale_fingerprint["battery.power_mw"] = FINGERPRINT["battery.power_mw"] + 1.0
    catalogue = pd.DataFrame([_catalogue_row("naive", "realized", None, None, start, end, stale_fingerprint)])
    assert _already_solved(catalogue, "naive", "realized", None, None, start, end, FINGERPRINT) is False


def test_already_solved_false_on_empty_catalogue():
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    assert _already_solved(pd.DataFrame(), "naive", "realized", None, None, start, end, FINGERPRINT) is False


# ---- end-to-end: resume skips, --force overrides ---------------------------

def test_resume_skips_already_solved_and_force_resolves(tmp_path, capsys):
    start = end = dt.date(2026, 6, 15)
    kwargs = dict(
        start_date=start, end_date=end, models=["lgbm_point"], quantiles=[0.5],
        strategies=["day_ahead_only"], include_perfect_foresight=True,
        results_dir=tmp_path, paths_config=PATHS,
    )

    run_batch(force=False, dry_run=False, **kwargs)
    first_catalogue = scan_runs(tmp_path)
    assert len(first_catalogue) == 2  # realized + lgbm_point q0.5

    capsys.readouterr()
    run_batch(force=False, dry_run=False, **kwargs)
    out = capsys.readouterr().out
    assert out.count("[skip]") == 2
    assert "[run]" not in out
    second_catalogue = scan_runs(tmp_path)
    assert set(second_catalogue["solve_id"]) == set(first_catalogue["solve_id"])

    capsys.readouterr()
    run_batch(force=True, dry_run=False, **kwargs)
    out = capsys.readouterr().out
    assert out.count("[run]") == 2
    third_catalogue = scan_runs(tmp_path)
    assert len(third_catalogue) == 4  # two fresh solve_ids alongside the original two
    assert set(first_catalogue["solve_id"]) < set(third_catalogue["solve_id"])
