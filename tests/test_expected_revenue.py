"""daily_revenue carries two views of the same decision: realized (settled on
real prices, the unprefixed columns) and expected (`expected_*`, valued at the
prices the decision was made on). Real one-day solves.
"""
import datetime as dt

import pytest

from src.config import (
    load_assumptions_config,
    load_battery_config,
    load_market_price_basis_config,
    load_market_timing_configs,
    load_paths_config,
    load_solver_config,
)
from src.model.battery import battery_from_config
from src.model.rolling import EXPECTED_COLUMNS, run_rolling

PATHS = load_paths_config()
TIMINGS = load_market_timing_configs(PATHS)
DAY = dt.date(2026, 6, 16)


def _run(**kwargs):
    return run_rolling(
        DAY, DAY, battery_from_config(load_battery_config()), load_solver_config(),
        TIMINGS["day_ahead"], TIMINGS["fcr"],
        assumptions=load_assumptions_config(), price_basis=load_market_price_basis_config(),
        paths_config=PATHS, **kwargs,
    ).daily_revenue.iloc[0]


def test_perfect_foresight_expected_equals_realized():
    row = _run()
    for column in EXPECTED_COLUMNS:
        assert row[f"expected_{column}"] == pytest.approx(row[column], abs=1e-6), column


def test_forecast_run_expected_differs_from_realized_and_is_self_consistent():
    row = _run(decision_source="forecast", forecast_model="lgbm_point", forecast_quantile=0.5)
    assert row["expected_day_ahead"] != pytest.approx(row["day_ahead"], abs=1e-3)
    assert row["expected_gross"] == pytest.approx(
        sum(row[f"expected_{c}"] for c in ("day_ahead", "fcr", "afrr_pos", "afrr_neg", "activation")), abs=1e-6
    )
    assert row["expected_net"] == pytest.approx(row["expected_gross"] - row["expected_degradation"], abs=1e-6)
    # Degradation is priced on throughput only, so both views agree on it.
    assert row["expected_degradation"] == pytest.approx(row["degradation"], abs=1e-9)
