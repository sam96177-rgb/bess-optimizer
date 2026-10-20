"""src.batch (SPEC.md §6a / Milestone 8 Part B): the matrix builder, the
resumability fingerprint match, and an end-to-end resume/--force check
against real (tiny) solves.
"""
import datetime as dt
import json

import pandas as pd
import pytest

from src.batch import ALL_STRATEGIES, DEFAULT_MODELS, DEFAULT_STRATEGIES, _already_solved, _fingerprint, _restamp_study_if_resumed, build_matrix, run_batch
from src.data.forecast_loader import MEDIAN_QUANTILE, load_forecast_run_ids
from src.config import load_paths_config
from src.results_catalogue import scan_runs
from tests.dispatch_helpers import ASSUMPTIONS, BATTERY, PRICE_BASIS, SOLVER

PATHS = load_paths_config()
FINGERPRINT = _fingerprint(BATTERY, SOLVER, ASSUMPTIONS, PRICE_BASIS)


# ---- build_matrix: hand-verified counts -----------------------------------

def test_default_strategies_hide_full_sequential_but_it_stays_selectable():
    assert DEFAULT_STRATEGIES == ["day_ahead_only", "full_cooptimized"]
    assert "full_sequential" in ALL_STRATEGIES


def test_build_matrix_default_shape_follows_the_study():
    combos = build_matrix(DEFAULT_MODELS, [MEDIAN_QUANTILE], DEFAULT_STRATEGIES, include_perfect_foresight=True)
    n_forecast = len(DEFAULT_MODELS) * len(DEFAULT_STRATEGIES)  # models x 1 quantile x strategies
    assert len(combos) == len(DEFAULT_STRATEGIES) + n_forecast
    realized = [c for c in combos if c[1] == "realized"]
    forecast = [c for c in combos if c[1] == "forecast"]
    assert len(realized) == len(DEFAULT_STRATEGIES)  # one ceiling per strategy
    assert len(forecast) == n_forecast
    assert all(c[2] is None and c[3] is None for c in realized)
    assert all(c[2] is not None and c[3] is not None for c in forecast)


def test_build_matrix_without_perfect_foresight_drops_the_realized_combos():
    combos = build_matrix(DEFAULT_MODELS, [MEDIAN_QUANTILE], DEFAULT_STRATEGIES, include_perfect_foresight=False)
    assert len(combos) == len(DEFAULT_MODELS) * len(DEFAULT_STRATEGIES)
    assert all(c[1] == "forecast" for c in combos)


def test_build_matrix_scales_with_models_x_quantiles_x_strategies():
    combos = build_matrix(["m1", "m2"], [0.1, 0.5, 0.9], ["day_ahead_only"], include_perfect_foresight=True)
    assert len(combos) == 1 + 2 * 3  # 1 perfect-foresight + 2 models x 3 quantiles


# ---- _already_solved: hand-built synthetic catalogue -----------------------

def _catalogue_row(strategy, decision_source, forecast_model, forecast_quantile, start_date, end_date, fingerprint):
    row = {
        "solve_id": "test-solve-id",
        "strategy": strategy, "decision_source": decision_source,
        "forecast_model": forecast_model, "forecast_quantile": forecast_quantile,
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
    }
    row.update(fingerprint)
    return row


def test_already_solved_true_for_exact_match():
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    catalogue = pd.DataFrame([_catalogue_row("day_ahead_only", "realized", None, None, start, end, FINGERPRINT)])
    assert _already_solved(catalogue, "day_ahead_only", "realized", None, None, start, end, FINGERPRINT) is True


def test_already_solved_true_for_matching_forecast_combo():
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    catalogue = pd.DataFrame(
        [_catalogue_row("full_cooptimized", "forecast", "lgbm_point", 0.5, start, end, FINGERPRINT)]
    )
    assert _already_solved(catalogue, "full_cooptimized", "forecast", "lgbm_point", 0.5, start, end, FINGERPRINT) is True


@pytest.mark.parametrize(
    "field,new_value",
    [("strategy", "full_cooptimized"), ("decision_source", "realized"), ("forecast_model", "other_model"),
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
    catalogue = pd.DataFrame([_catalogue_row("day_ahead_only", "realized", None, None, start, end, stale_fingerprint)])
    assert _already_solved(catalogue, "day_ahead_only", "realized", None, None, start, end, FINGERPRINT) is False


def test_fingerprint_includes_fcr_power_factor_so_pre_v8_runs_are_resolved():
    assert FINGERPRINT["model.fcr_power_factor"] == 1.25
    assert "battery.reserve_total_limit" not in FINGERPRINT
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    pre_v8 = {k: v for k, v in FINGERPRINT.items() if k != "model.fcr_power_factor"}
    catalogue = pd.DataFrame([_catalogue_row("day_ahead_only", "realized", None, None, start, end, pre_v8)])
    assert _already_solved(catalogue, "day_ahead_only", "realized", None, None, start, end, FINGERPRINT) is False


def test_fingerprint_includes_bid_granularity_so_pre_v4_runs_are_resolved():
    assert FINGERPRINT["market_prices.bid_granularity_mw.fcr"] == PRICE_BASIS.granularity_for("fcr")
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    pre_v4 = {k: v for k, v in FINGERPRINT.items() if not k.startswith("market_prices.")}
    catalogue = pd.DataFrame([_catalogue_row("day_ahead_only", "realized", None, None, start, end, pre_v4)])
    assert _already_solved(catalogue, "day_ahead_only", "realized", None, None, start, end, FINGERPRINT) is False
    flipped = {**FINGERPRINT, "market_prices.bid_granularity_mw.fcr": 0.0}
    catalogue = pd.DataFrame([_catalogue_row("day_ahead_only", "realized", None, None, start, end, flipped)])
    assert _already_solved(catalogue, "day_ahead_only", "realized", None, None, start, end, FINGERPRINT) is False


def test_restamp_study_if_resumed_updates_a_stale_study_tag(tmp_path):
    """PLAN_V6_REVISION.md V6-2: a ceiling ('realized') run doesn't depend
    on the forecast model or study, so a later study's batch run resumes
    it instead of re-solving -- restamping keeps every page that filters
    the catalogue by the CURRENT study able to find it."""
    run_dir = tmp_path / "solve-123"
    run_dir.mkdir()
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps({"solve_id": "solve-123", "study": "v5", "strategy": "day_ahead_only"}), encoding="utf-8")

    _restamp_study_if_resumed(tmp_path, "solve-123", "v6")

    assert json.loads(meta_path.read_text(encoding="utf-8"))["study"] == "v6"


def test_restamp_study_if_resumed_is_a_no_op_when_already_current(tmp_path):
    run_dir = tmp_path / "solve-123"
    run_dir.mkdir()
    meta_path = run_dir / "run_meta.json"
    original = json.dumps({"solve_id": "solve-123", "study": "v6", "strategy": "day_ahead_only"})
    meta_path.write_text(original, encoding="utf-8")

    _restamp_study_if_resumed(tmp_path, "solve-123", "v6")

    assert meta_path.read_text(encoding="utf-8") == original


def test_already_solved_false_on_empty_catalogue():
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    assert _already_solved(pd.DataFrame(), "day_ahead_only", "realized", None, None, start, end, FINGERPRINT) is False


# ---- V6-7: resume must notice a model's forecast was re-run ----------------

def test_already_solved_matches_todays_forecast_run_ids():
    """A forecast-driven candidate whose forecast_run_ids.<market> columns
    equal what load_forecast_run_ids resolves right now is still a match."""
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    run_ids = load_forecast_run_ids(PATHS, "seasonal_naive")
    row = _catalogue_row("day_ahead_only", "forecast", "seasonal_naive", 0.5, start, end, FINGERPRINT)
    row.update({f"forecast_run_ids.{market}": run_id for market, run_id in run_ids.items()})
    catalogue = pd.DataFrame([row])
    assert _already_solved(
        catalogue, "day_ahead_only", "forecast", "seasonal_naive", 0.5, start, end, FINGERPRINT,
        paths_config=PATHS,
    ) is True


def test_already_solved_false_when_the_model_was_re_forecast():
    """Same candidate, but its forecast_run_ids are stale (the model was
    re-backtested since this run solved) -- resume must re-solve it."""
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    run_ids = load_forecast_run_ids(PATHS, "seasonal_naive")
    stale_run_ids = {market: f"stale-{run_id}" for market, run_id in run_ids.items()}
    row = _catalogue_row("day_ahead_only", "forecast", "seasonal_naive", 0.5, start, end, FINGERPRINT)
    row.update({f"forecast_run_ids.{market}": run_id for market, run_id in stale_run_ids.items()})
    catalogue = pd.DataFrame([row])
    assert _already_solved(
        catalogue, "day_ahead_only", "forecast", "seasonal_naive", 0.5, start, end, FINGERPRINT,
        paths_config=PATHS,
    ) is False


def test_already_solved_ignores_forecast_run_ids_without_a_paths_config():
    """Without paths_config (e.g. older call sites), the resume check falls
    back to the fingerprint-only behaviour -- no regression for callers that
    don't pass it."""
    start, end = dt.date(2026, 6, 1), dt.date(2026, 6, 5)
    row = _catalogue_row("day_ahead_only", "forecast", "seasonal_naive", 0.5, start, end, FINGERPRINT)
    row.update({"forecast_run_ids.day_ahead": "whatever-was-recorded-at-the-time"})
    catalogue = pd.DataFrame([row])
    assert _already_solved(
        catalogue, "day_ahead_only", "forecast", "seasonal_naive", 0.5, start, end, FINGERPRINT,
    ) is True


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
