import datetime as dt

import pandas as pd
import pytest

from src.config import MarketTimingConfig, PathsConfig, load_market_timing_configs, load_paths_config
from src.data.forecast_loader import (
    DayPrices,
    SkipRecord,
    load_day_prices,
    load_price_range,
    load_raw_y_true,
)

PATHS = load_paths_config()
TIMINGS = load_market_timing_configs(PATHS)
DAY_AHEAD_TIMING = TIMINGS["day_ahead"]
FCR_TIMING = TIMINGS["fcr"]


# ---- against the real sibling repo (read-only) --------------------------

def test_load_raw_y_true_reads_real_data():
    raw = load_raw_y_true(PATHS)
    assert set(raw.index.get_level_values("market").unique()) == {
        "day_ahead", "fcr", "afrr_pos", "afrr_neg"
    }
    assert not raw.empty


def test_full_day_ahead_to_august_window_matches_milestone_0_expectations():
    """day_ahead's real data starts 2026-03-21 (confirmed in Milestone 0);
    the 20 days before that in March must be skipped, nothing else in the
    March-August window should be.
    """
    days_ok, skipped = load_price_range(
        dt.date(2026, 3, 1), dt.date(2026, 8, 31), DAY_AHEAD_TIMING, FCR_TIMING, PATHS
    )
    assert len(days_ok) == 164
    assert len(skipped) == 20
    assert all(s.day < dt.date(2026, 3, 21) for s in skipped)
    assert all("day_ahead" in s.reason for s in skipped)


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
    raw = load_raw_y_true(PATHS)
    result = load_day_prices(dt.date(2026, 3, 10), raw, DAY_AHEAD_TIMING, FCR_TIMING)
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
