import datetime as dt

import pytest

from src.config import load_paths_config
from src.model.build import build_model
from src.model.solve import solve_model
from src.valuation.benchmarks import STRATEGY_FLAGS, run_all_benchmarks, run_strategy
from src.data.forecast_loader import load_day_prices, load_raw_y_true
from tests.dispatch_helpers import ASSUMPTIONS, BATTERY, PRICE_BASIS, SOLVER, TIMINGS

PATHS = load_paths_config()

START = dt.date(2026, 6, 15)
END = dt.date(2026, 6, 17)
RUN = dict(
    battery=BATTERY, solver_config=SOLVER, assumptions=ASSUMPTIONS, price_basis=PRICE_BASIS,
    day_ahead_timing=TIMINGS["day_ahead"], capacity_timing=TIMINGS["fcr"], paths_config=PATHS,
)


def test_day_ahead_only_benchmark_matches_manual_build_model():
    result = run_strategy("day_ahead_only", start_date=START, end_date=START, **RUN)
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


def test_full_cooptimized_never_earns_less_than_day_ahead_only_per_day():
    """The day-ahead-only solution (zero ancillary commitment) is always
    feasible inside the full co-optimized model, so its optimum can never
    be worse -- a real correctness check on the joint MILP, not just a
    plausibility assumption.
    """
    day_ahead_only = run_strategy("day_ahead_only", start_date=START, end_date=END, **RUN)
    full = run_strategy("full_cooptimized", start_date=START, end_date=END, **RUN)

    merged = day_ahead_only.daily_revenue.merge(
        full.daily_revenue, on="solve_date", suffixes=("_da_only", "_full")
    )
    assert (merged["total_full"] >= merged["total_da_only"] - 1e-4).all()
    assert merged["total_full"].sum() > merged["total_da_only"].sum()


def test_run_all_benchmarks_defaults_exclude_full_sequential():
    results = run_all_benchmarks(start_date=START, end_date=START, **RUN)
    assert set(results) == {"day_ahead_only", "full_cooptimized"}


def test_run_all_benchmarks_can_run_every_strategy():
    results = run_all_benchmarks(list(STRATEGY_FLAGS), start_date=START, end_date=START, **RUN)
    assert set(results) == set(STRATEGY_FLAGS)
    for name, result in results.items():
        assert len(result.daily_revenue) == 1, name
    # Perfect foresight: every sequential stage sees the real prices.
    assert results["full_sequential"].daily_revenue["net"].iloc[0] == pytest.approx(
        results["full_cooptimized"].daily_revenue["net"].iloc[0], rel=1e-3
    )


def test_run_strategy_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown strategy 'naive'"):
        run_strategy("naive", start_date=START, end_date=START, **RUN)
