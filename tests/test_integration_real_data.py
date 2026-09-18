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
from src.data.forecast_loader import load_day_prices, load_raw_y_true
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
