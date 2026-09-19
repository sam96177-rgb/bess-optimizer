"""DayPricePair / load_price_pair_range (SPEC.md §6a): a day is skipped
if EITHER side (decision or settlement) is incomplete, and the skip
reason says which side failed. Synthetic, isolated from the real sibling
repo -- see tests/test_forecast_loader.py for the real-data coverage of
the underlying load_raw_y_pred/load_raw_y_true.
"""
import datetime as dt

import pandas as pd

from src.config import MarketTimingConfig, PathsConfig
from src.data.forecast_loader import load_price_pair_range
from src.data.grid import build_day_grid

_DAY_AHEAD_TIMING = MarketTimingConfig(
    name="day_ahead", timezone="Europe/Berlin", dst_convention="variable_period_count",
    product_length_hours=0.25, delivery_start_local=dt.time(0, 0),
)
_FCR_TIMING = MarketTimingConfig(
    name="fcr", timezone="Europe/Berlin", dst_convention="fixed_period_count",
    product_length_hours=4, delivery_start_local=dt.time(0, 0),
)

MODEL = "test_model"
QUANTILE = 0.5


def _write_fake_repo(
    tmp_path,
    day: dt.date,
    y_true_nan_at: tuple[str, pd.Timestamp] | None = None,
    y_pred_nan_at: tuple[str, pd.Timestamp] | None = None,
) -> PathsConfig:
    """One parquet file carrying both y_true and y_pred for a single
    synthetic day, complete across all four markets unless a
    (market, target_time) cell is nulled out on one side.
    """
    grid = build_day_grid(day, _DAY_AHEAD_TIMING, _FCR_TIMING)
    rows = []
    for step in grid.steps:
        rows.append({"market": "day_ahead", "target_time": step, "y_true": 50.0, "y_pred": 55.0})
    for block in grid.blocks:
        for market, true_val, pred_val in (("fcr", 30.0, 32.0), ("afrr_pos", 10.0, 11.0), ("afrr_neg", 15.0, 16.0)):
            rows.append({"market": market, "target_time": block.start, "y_true": true_val, "y_pred": pred_val})

    df = pd.DataFrame(rows)
    df["model"] = MODEL
    df["quantile"] = QUANTILE

    if y_true_nan_at is not None:
        market, target_time = y_true_nan_at
        df.loc[(df["market"] == market) & (df["target_time"] == target_time), "y_true"] = float("nan")
    if y_pred_nan_at is not None:
        market, target_time = y_pred_nan_at
        df.loc[(df["market"] == market) & (df["target_time"] == target_time), "y_pred"] = float("nan")

    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    df.to_parquet(results_dir / "fake.parquet")
    return PathsConfig(forecast_repo_path=tmp_path)


def test_settlement_nan_with_complete_decision_skips_with_settlement_prefix(tmp_path):
    day = dt.date(2026, 6, 1)
    grid = build_day_grid(day, _DAY_AHEAD_TIMING, _FCR_TIMING)
    paths = _write_fake_repo(tmp_path, day, y_true_nan_at=("day_ahead", grid.steps[5]))

    pairs, skipped = load_price_pair_range(
        day, day, _DAY_AHEAD_TIMING, _FCR_TIMING, paths,
        decision_source="forecast", forecast_model=MODEL, forecast_quantile=QUANTILE,
    )
    assert pairs == []
    assert len(skipped) == 1
    assert skipped[0].day == day
    assert skipped[0].reason.startswith("settlement:")
    assert "day_ahead" in skipped[0].reason


def test_decision_nan_with_complete_settlement_skips_with_decision_prefix(tmp_path):
    day = dt.date(2026, 6, 1)
    grid = build_day_grid(day, _DAY_AHEAD_TIMING, _FCR_TIMING)
    paths = _write_fake_repo(tmp_path, day, y_pred_nan_at=("fcr", grid.blocks[2].start))

    pairs, skipped = load_price_pair_range(
        day, day, _DAY_AHEAD_TIMING, _FCR_TIMING, paths,
        decision_source="forecast", forecast_model=MODEL, forecast_quantile=QUANTILE,
    )
    assert pairs == []
    assert len(skipped) == 1
    assert skipped[0].reason.startswith(f"decision ({MODEL} q{QUANTILE}):")
    assert "fcr" in skipped[0].reason


def test_both_sides_complete_solves_with_distinct_decision_and_settlement_prices(tmp_path):
    day = dt.date(2026, 6, 1)
    paths = _write_fake_repo(tmp_path, day)

    pairs, skipped = load_price_pair_range(
        day, day, _DAY_AHEAD_TIMING, _FCR_TIMING, paths,
        decision_source="forecast", forecast_model=MODEL, forecast_quantile=QUANTILE,
    )
    assert skipped == []
    assert len(pairs) == 1
    pair = pairs[0]
    assert (pair.decision.day_ahead == 55.0).all()
    assert (pair.settlement.day_ahead == 50.0).all()


def test_realized_decision_source_uses_y_true_for_both_sides(tmp_path):
    day = dt.date(2026, 6, 1)
    paths = _write_fake_repo(tmp_path, day)

    pairs, skipped = load_price_pair_range(day, day, _DAY_AHEAD_TIMING, _FCR_TIMING, paths)
    assert skipped == []
    pair = pairs[0]
    assert (pair.decision.day_ahead == pair.settlement.day_ahead).all()
    assert (pair.decision.day_ahead == 50.0).all()
