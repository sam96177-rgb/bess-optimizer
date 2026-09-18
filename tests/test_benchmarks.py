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
from src.model.build import build_model
from src.model.solve import solve_model
from src.valuation.benchmarks import (
    _cheapest_and_priciest_blocks,
    run_all_benchmarks,
    run_day_ahead_only_benchmark,
    run_full_cooptimized_benchmark,
    run_naive_benchmark,
)
from src.data.forecast_loader import load_day_prices, load_raw_y_true

PATHS = load_paths_config()
TIMINGS = load_market_timing_configs(PATHS)
BATTERY = battery_from_config(load_battery_config())
SOLVER = load_solver_config()
ASSUMPTIONS = load_assumptions_config()
PRICE_BASIS = load_market_price_basis_config()

START = dt.date(2026, 6, 15)
END = dt.date(2026, 6, 17)


def test_day_ahead_only_benchmark_matches_manual_build_model():
    result = run_day_ahead_only_benchmark(START, START, BATTERY, SOLVER, TIMINGS["day_ahead"], TIMINGS["fcr"], paths_config=PATHS)
    raw = load_raw_y_true(PATHS)
    day_prices = load_day_prices(START, raw, TIMINGS["day_ahead"], TIMINGS["fcr"])
    model = build_model(
        steps=day_prices.grid.steps, day_ahead_prices=day_prices.day_ahead, battery=BATTERY,
        delta_t_hours=day_prices.grid.delta_t_hours, ancillary=None,
    )
    manual = solve_model(model, SOLVER)
    assert result.daily_revenue["total"].iloc[0] == pytest.approx(manual.objective_value, rel=1e-6)
    assert result.daily_revenue["fcr"].iloc[0] == 0.0
    assert result.daily_revenue["activation"].iloc[0] == 0.0


def test_naive_benchmark_charges_only_cheapest_block_discharges_only_priciest():
    result = run_naive_benchmark(START, START, BATTERY, TIMINGS["day_ahead"], TIMINGS["fcr"], paths_config=PATHS)
    raw = load_raw_y_true(PATHS)
    day_prices = load_day_prices(START, raw, TIMINGS["day_ahead"], TIMINGS["fcr"])
    cheapest, priciest = _cheapest_and_priciest_blocks(day_prices)

    lf = result.long_format
    p_ch = lf[lf["variable"] == "p_ch"].set_index("target_time")["value"]
    p_dis = lf[lf["variable"] == "p_dis"].set_index("target_time")["value"]

    for step, block_idx in zip(day_prices.grid.steps, day_prices.grid.step_block_index):
        if block_idx != cheapest:
            assert p_ch.loc[step] == 0.0
        if block_idx != priciest:
            assert p_dis.loc[step] == 0.0

    e_vals = lf[lf["variable"] == "e"]["value"]
    assert (e_vals >= BATTERY.soc_min_mwh - 1e-6).all()
    assert (e_vals <= BATTERY.soc_max_mwh + 1e-6).all()


def test_naive_benchmark_never_charges_and_discharges_same_step():
    result = run_naive_benchmark(START, END, BATTERY, TIMINGS["day_ahead"], TIMINGS["fcr"], paths_config=PATHS)
    lf = result.long_format
    p_ch = lf[lf["variable"] == "p_ch"].set_index(["solve_date", "target_time"])["value"]
    p_dis = lf[lf["variable"] == "p_dis"].set_index(["solve_date", "target_time"])["value"]
    both = (p_ch > 1e-9) & (p_dis > 1e-9)
    assert not both.any()


def test_full_cooptimized_never_earns_less_than_day_ahead_only_per_day():
    """The day-ahead-only solution (zero ancillary commitment) is always
    feasible inside the full co-optimized model, so its optimum can never
    be worse -- a real correctness check on the joint MILP, not just a
    plausibility assumption.
    """
    day_ahead_only = run_day_ahead_only_benchmark(START, END, BATTERY, SOLVER, TIMINGS["day_ahead"], TIMINGS["fcr"], paths_config=PATHS)
    full = run_full_cooptimized_benchmark(
        START, END, BATTERY, SOLVER, ASSUMPTIONS, PRICE_BASIS, TIMINGS["day_ahead"], TIMINGS["fcr"], paths_config=PATHS
    )

    merged = day_ahead_only.daily_revenue.merge(
        full.daily_revenue, on="solve_date", suffixes=("_da_only", "_full")
    )
    assert (merged["total_full"] >= merged["total_da_only"] - 1e-4).all()
    assert merged["total_full"].sum() > merged["total_da_only"].sum()


def test_run_all_benchmarks_returns_three_strategies():
    results = run_all_benchmarks(
        START, START, BATTERY, SOLVER, ASSUMPTIONS, PRICE_BASIS, TIMINGS["day_ahead"], TIMINGS["fcr"], paths_config=PATHS
    )
    assert set(results.keys()) == {"day_ahead_only", "naive", "full_cooptimized"}
    for name, result in results.items():
        assert len(result.daily_revenue) == 1, name
