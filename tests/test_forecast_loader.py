import datetime as dt

import pandas as pd
import pytest

from src.config import MarketTimingConfig, PathsConfig, load_forecast_blends_config, load_paths_config, load_study_config
from src.data.forecast_loader import (
    DayPrices,
    SkipRecord,
    load_day_prices,
    load_forecast_run_ids,
    load_price_range,
    load_raw_y_pred,
    load_raw_y_true,
    resolve_per_market_models,
)
from tests.dispatch_helpers import TIMINGS

PATHS = load_paths_config()
DAY_AHEAD_TIMING = TIMINGS["day_ahead"]
FCR_TIMING = TIMINGS["fcr"]


# ---- against the real sibling repo (read-only) --------------------------

def test_load_raw_y_true_reads_real_data():
    raw = load_raw_y_true(PATHS)
    assert set(raw.index.get_level_values("market").unique()) == {
        "day_ahead", "fcr", "afrr_pos", "afrr_neg"
    }
    assert not raw.empty


def test_full_march_to_august_window_has_no_skips_under_v6s_validation_backtest():
    """Milestone 0 originally found day_ahead's real data starting
    2026-03-21, with the 20 days before that in March skipped. V6-4's
    validation-window backtest (PLAN_V6_REVISION.md) extended every
    forecast run's own coverage back to 2026-01-20 (day_ahead's
    validation_window start), so this whole March-August window -- entirely
    after that -- now has no skips at all.
    """
    days_ok, skipped = load_price_range(
        dt.date(2026, 3, 1), dt.date(2026, 8, 31), DAY_AHEAD_TIMING, FCR_TIMING, PATHS
    )
    assert len(days_ok) == 184
    assert len(skipped) == 0


def test_dst_day_loads_successfully_not_skipped():
    raw = load_raw_y_true(PATHS)
    result = load_day_prices(dt.date(2026, 3, 29), raw, DAY_AHEAD_TIMING, FCR_TIMING)
    assert isinstance(result, DayPrices)
    assert len(result.day_ahead) == 92
    assert len(result.fcr) == 6
    assert len(result.afrr_pos) == 6
    assert len(result.afrr_neg) == 6
    assert not result.day_ahead.isna().any()


def test_pre_day_ahead_day_is_skipped_with_reason():
    """2026-03-10 was before day_ahead's coverage under V5's test-window-only
    backtest; under V6-4's validation-window backtest that date is covered
    (see test_full_march_to_august_window_has_no_skips_under_v6s_validation_backtest),
    so this now uses a date before the validation window's own 2026-01-20 start."""
    raw = load_raw_y_true(PATHS)
    result = load_day_prices(dt.date(2026, 1, 10), raw, DAY_AHEAD_TIMING, FCR_TIMING)
    assert isinstance(result, SkipRecord)
    assert "day_ahead" in result.reason


def test_a_normal_march_day_loads_with_expected_shapes():
    raw = load_raw_y_true(PATHS)
    result = load_day_prices(dt.date(2026, 6, 15), raw, DAY_AHEAD_TIMING, FCR_TIMING)
    assert isinstance(result, DayPrices)
    assert len(result.day_ahead) == 96
    assert list(result.fcr.index) == [0, 1, 2, 3, 4, 5]
    assert not result.fcr.isna().any()
    assert not result.afrr_pos.isna().any()
    assert not result.afrr_neg.isna().any()


# ---- load_raw_y_pred, against the real sibling repo (read-only) ---------

def test_load_raw_y_pred_filters_to_one_model_and_quantile():
    raw = load_raw_y_pred(PATHS, "lgbm_point", 0.5)
    assert set(raw.index.get_level_values("market").unique()) == {
        "day_ahead", "fcr", "afrr_pos", "afrr_neg"
    }
    # (market, target_time) unique for this model+quantile (Verified facts).
    assert not raw.index.duplicated().any()
    assert not raw.empty


def test_load_raw_y_pred_raises_on_unknown_model():
    with pytest.raises(ValueError, match="Unknown forecast model"):
        load_raw_y_pred(PATHS, "not_a_real_model", 0.5)


def test_load_raw_y_pred_raises_for_quantile_absent_from_point_model():
    # lgbm_point only ever produces the median (q=0.5) -- Verified facts.
    with pytest.raises(ValueError, match="no rows at quantile"):
        load_raw_y_pred(PATHS, "lgbm_point", 0.05)


def test_load_raw_y_pred_preserves_nan_rows_for_seasonal_naive_dst_days():
    """seasonal_naive's 'same block last week' lookup lands on an
    irregular day for 2026-03-29 (DST) and its one-week echo 2026-04-05 --
    exactly 10 NaN y_pred rows in each of the three capacity markets
    (Verified facts). load_raw_y_pred must keep them, not drop them, so
    load_day_prices can skip those days with a legible reason.
    """
    raw = load_raw_y_pred(PATHS, "seasonal_naive", 0.5)
    nan_rows = raw[raw.isna()]
    assert len(nan_rows) == 30
    assert dict(nan_rows.index.get_level_values("market").value_counts()) == {
        "fcr": 10, "afrr_pos": 10, "afrr_neg": 10,
    }
    nan_dates = {pd.Timestamp(t).date() for t in nan_rows.index.get_level_values("target_time")}
    assert nan_dates == {dt.date(2026, 3, 29), dt.date(2026, 4, 5)}


def test_load_forecast_run_ids_returns_one_id_per_market():
    run_ids = load_forecast_run_ids(PATHS, "lgbm_point")
    assert set(run_ids.keys()) == {"day_ahead", "fcr", "afrr_pos", "afrr_neg"}
    assert all(isinstance(v, str) and v for v in run_ids.values())
    # Each (market, model) pair is its own parquet file (Verified facts).
    assert len(set(run_ids.values())) == 4


def test_load_forecast_run_ids_raises_on_unknown_model():
    with pytest.raises(ValueError, match="Unknown forecast model"):
        load_forecast_run_ids(PATHS, "not_a_real_model")


# ---- per-market blends (PLAN_V4_REVISION.md section 3) -------------------

def test_resolve_per_market_models_expands_a_blend():
    # The choice itself is generated by pipeline/03_select_blend.py (PLAN_V6_REVISION.md V6-4) and
    # changes with every study, so only its shape is pinned: one study model per market.
    blend = load_forecast_blends_config().resolve("per_market_best")
    assert resolve_per_market_models("per_market_best") == blend
    assert set(blend) == {"day_ahead", "fcr", "afrr_pos", "afrr_neg"}
    assert set(blend.values()) <= set(load_study_config().models)


def test_resolve_per_market_models_repeats_a_plain_model():
    assert resolve_per_market_models("lgbm_point") == {
        "day_ahead": "lgbm_point", "fcr": "lgbm_point", "afrr_pos": "lgbm_point", "afrr_neg": "lgbm_point",
    }


def test_load_raw_y_pred_for_a_blend_matches_each_markets_own_model():
    # Reads the real, current per_market_best mapping rather than hardcoding
    # specific model names, so this stays correct whenever
    # pipeline/03_select_blend.py regenerates it (PLAN_V6_REVISION.md V6-4).
    blend = load_forecast_blends_config().resolve("per_market_best")
    blended = load_raw_y_pred(PATHS, "per_market_best", 0.5)
    assert set(blended.index.get_level_values("market").unique()) == {
        "day_ahead", "fcr", "afrr_pos", "afrr_neg"
    }
    for market, model in blend.items():
        pd.testing.assert_series_equal(blended.loc[market], load_raw_y_pred(PATHS, model, 0.5).loc[market])


def test_load_forecast_run_ids_for_a_blend_matches_each_markets_own_model():
    blend = load_forecast_blends_config().resolve("per_market_best")
    blended = load_forecast_run_ids(PATHS, "per_market_best")
    for market, model in blend.items():
        assert blended[market] == load_forecast_run_ids(PATHS, model)[market]


# ---- synthetic edge cases (isolated from real data) ----------------------

_DAY_AHEAD_TIMING = MarketTimingConfig(
    name="day_ahead", timezone="Europe/Berlin", dst_convention="variable_period_count",
    product_length_hours=0.25, delivery_start_local=dt.time(0, 0),
)
_FCR_TIMING = MarketTimingConfig(
    name="fcr", timezone="Europe/Berlin", dst_convention="fixed_period_count",
    product_length_hours=4, delivery_start_local=dt.time(0, 0),
)


def _make_complete_raw(day: dt.date) -> pd.Series:
    from src.data.grid import build_day_grid

    grid = build_day_grid(day, _DAY_AHEAD_TIMING, _FCR_TIMING)
    rows = []
    for step in grid.steps:
        rows.append(("day_ahead", step, 50.0))
    for block in grid.blocks:
        rows.append(("fcr", block.start, 30.0))
        rows.append(("afrr_pos", block.start, 10.0))
        rows.append(("afrr_neg", block.start, 15.0))
    df = pd.DataFrame(rows, columns=["market", "target_time", "y_true"])
    return df.set_index(["market", "target_time"])["y_true"]


def test_synthetic_complete_day_loads():
    day = dt.date(2026, 6, 1)
    raw = _make_complete_raw(day)
    result = load_day_prices(day, raw, _DAY_AHEAD_TIMING, _FCR_TIMING)
    assert isinstance(result, DayPrices)
    assert (result.day_ahead == 50.0).all()
    assert (result.fcr == 30.0).all()


def test_synthetic_day_with_one_nan_fcr_block_is_skipped():
    day = dt.date(2026, 6, 1)
    raw = _make_complete_raw(day)
    from src.data.grid import build_day_grid

    grid = build_day_grid(day, _DAY_AHEAD_TIMING, _FCR_TIMING)
    nan_target = ("fcr", grid.blocks[3].start)
    raw = raw.copy()
    raw.loc[nan_target] = float("nan")

    result = load_day_prices(day, raw, _DAY_AHEAD_TIMING, _FCR_TIMING)
    assert isinstance(result, SkipRecord)
    assert "fcr" in result.reason


def test_synthetic_day_missing_market_entirely_is_skipped():
    day = dt.date(2026, 6, 1)
    raw = _make_complete_raw(day)
    raw = raw.drop("afrr_neg", level="market")

    result = load_day_prices(day, raw, _DAY_AHEAD_TIMING, _FCR_TIMING)
    assert isinstance(result, SkipRecord)
    assert "afrr_neg" in result.reason


def test_load_raw_y_true_raises_on_conflicting_values(tmp_path):
    ts = pd.Timestamp("2026-06-01 00:00", tz="Europe/Berlin")

    df1 = pd.DataFrame([{"market": "day_ahead", "target_time": ts, "y_true": 50.0}])
    df2 = pd.DataFrame([{"market": "day_ahead", "target_time": ts, "y_true": 999.0}])

    fake_repo = tmp_path
    (fake_repo / "results").mkdir(exist_ok=True)
    df1.to_parquet(fake_repo / "results" / "a.parquet")
    df2.to_parquet(fake_repo / "results" / "b.parquet")

    fake_paths = PathsConfig(forecast_repo_path=fake_repo)
    with pytest.raises(ValueError, match="conflicting y_true"):
        load_raw_y_true(fake_paths)


def test_load_price_range_rejects_start_after_end():
    with pytest.raises(ValueError, match="must not be after"):
        load_price_range(
            dt.date(2026, 6, 2), dt.date(2026, 6, 1), DAY_AHEAD_TIMING, FCR_TIMING, PATHS
        )
