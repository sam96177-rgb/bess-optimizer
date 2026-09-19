"""End-to-end check: real prices from bess-market-forecast, through the
full grid -> forecast_loader -> build_model -> solve -> revenue pipeline,
for one real day. Everything else tests these pieces in isolation with
synthetic data; this is the one place that proves they fit together.
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
from src.data.forecast_loader import DayPricePair, SkipRecord, load_day_prices, load_price_pair_range, load_raw_y_true
from src.model.battery import battery_from_config
from src.model.build import AncillaryInputs, build_model
from src.model.solve import solve_model
from src.valuation.revenue import compute_revenue

TOL = 1e-4


def test_real_day_end_to_end():
    paths = load_paths_config()
    timings = load_market_timing_configs(paths)
    battery = battery_from_config(load_battery_config())
    solver = load_solver_config()
    assumptions = load_assumptions_config()
    price_basis = load_market_price_basis_config()

    raw = load_raw_y_true(paths)
    day_prices = load_day_prices(dt.date(2026, 6, 15), raw, timings["day_ahead"], timings["fcr"])

    ancillary = AncillaryInputs(
        blocks=day_prices.grid.blocks,
        step_block_index=day_prices.grid.step_block_index,
        fcr_prices=day_prices.fcr,
        afrr_pos_prices=day_prices.afrr_pos,
        afrr_neg_prices=day_prices.afrr_neg,
        price_basis=price_basis,
        assumptions=assumptions,
    )
    model = build_model(
        steps=day_prices.grid.steps,
        day_ahead_prices=day_prices.day_ahead,
        battery=battery,
        delta_t_hours=day_prices.grid.delta_t_hours,
        ancillary=ancillary,
    )
    result = solve_model(model, solver)

    assert (result.e >= battery.soc_min_mwh - TOL).all()
    assert (result.e <= battery.soc_max_mwh + TOL).all()
    assert (result.c_fcr <= 0.8 * battery.power_mw + TOL).all()

    revenue = compute_revenue(result, day_prices.day_ahead, day_prices.grid.delta_t_hours, ancillary)
    assert revenue.total == pytest.approx(result.objective_value, rel=1e-6)


# ---- forecast-driven, real DST day (SPEC.md §6a) --------------------------

def test_real_dst_day_skips_for_seasonal_naive_but_solves_for_lgbm_point():
    """2026-03-29 is the DST day seasonal_naive's 'same block last week'
    lookup can't resolve, producing NaN y_pred for that day in all three
    capacity markets (Verified facts) -- a forecast-driven run must skip
    it with a decision-side reason. lgbm_point has no such gap and must
    solve the identical day cleanly on the identical settlement (y_true)
    prices.
    """
    paths = load_paths_config()
    timings = load_market_timing_configs(paths)
    day = dt.date(2026, 3, 29)

    seasonal_pairs, seasonal_skipped = load_price_pair_range(
        day, day, timings["day_ahead"], timings["fcr"], paths,
        decision_source="forecast", forecast_model="seasonal_naive", forecast_quantile=0.5,
    )
    assert seasonal_pairs == []
    assert len(seasonal_skipped) == 1
    skip = seasonal_skipped[0]
    assert isinstance(skip, SkipRecord)
    assert skip.day == day
    assert skip.reason.startswith("decision (seasonal_naive q0.5):")
    assert "fcr" in skip.reason

    point_pairs, point_skipped = load_price_pair_range(
        day, day, timings["day_ahead"], timings["fcr"], paths,
        decision_source="forecast", forecast_model="lgbm_point", forecast_quantile=0.5,
    )
    assert point_skipped == []
    assert len(point_pairs) == 1
    pair = point_pairs[0]
    assert isinstance(pair, DayPricePair)
    assert pair.day == day
    assert not pair.decision.day_ahead.isna().any()
    assert not pair.settlement.day_ahead.isna().any()
    # decide-on-forecast/settle-on-realized: the two sides are genuinely
    # different price series, not the same y_true reused for both.
    assert not pair.decision.day_ahead.equals(pair.settlement.day_ahead)
