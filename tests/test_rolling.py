import dataclasses
import datetime as dt

import pandas as pd
import pytest

from src.config import (
    MarketTimingConfig,
    PathsConfig,
    load_assumptions_config,
    load_battery_config,
    load_market_price_basis_config,
    load_market_timing_configs,
    load_paths_config,
    load_solver_config,
)
from src.data.forecast_loader import load_day_prices, load_raw_y_true
from src.data.grid import build_day_grid
from src.model.battery import battery_from_config
from src.model.build import AncillaryInputs, build_model
from src.model.rolling import LONG_FORMAT_COLUMNS, run_rolling, write_rolling_run_results
from src.model.solve import solve_model
from src.valuation.revenue import compute_revenue

PATHS = load_paths_config()
TIMINGS = load_market_timing_configs(PATHS)
BATTERY = battery_from_config(load_battery_config())
SOLVER = load_solver_config()
ASSUMPTIONS = load_assumptions_config()
PRICE_BASIS = load_market_price_basis_config()

# Synthetic timing configs for the forecast-driven test below -- isolated
# from the real sibling repo, same convention as
# tests/test_forecast_loader.py's own synthetic edge cases.
_SYNTH_DAY_AHEAD_TIMING = MarketTimingConfig(
    name="day_ahead", timezone="Europe/Berlin", dst_convention="variable_period_count",
    product_length_hours=0.25, delivery_start_local=dt.time(0, 0),
)
_SYNTH_FCR_TIMING = MarketTimingConfig(
    name="fcr", timezone="Europe/Berlin", dst_convention="fixed_period_count",
    product_length_hours=4, delivery_start_local=dt.time(0, 0),
)


def _run(start, end, **kwargs):
    return run_rolling(
        start_date=start,
        end_date=end,
        battery=BATTERY,
        solver_config=SOLVER,
        assumptions=ASSUMPTIONS,
        price_basis=PRICE_BASIS,
        day_ahead_timing=TIMINGS["day_ahead"],
        capacity_timing=TIMINGS["fcr"],
        paths_config=PATHS,
        **kwargs,
    )


def test_long_format_schema_and_shapes():
    result = _run(dt.date(2026, 6, 15), dt.date(2026, 6, 16))
    assert list(result.long_format.columns) == LONG_FORMAT_COLUMNS
    assert result.long_format["solve_id"].nunique() == 1
    assert set(result.long_format["solve_date"].unique()) == {dt.date(2026, 6, 15), dt.date(2026, 6, 16)}

    battery_rows = result.long_format[result.long_format["market"] == "battery"]
    assert set(battery_rows["variable"].unique()) == {"u", "e"}
    assert battery_rows["block_id"].isna().all()

    capacity_rows = result.long_format[result.long_format["market"].isin(["fcr", "afrr_pos", "afrr_neg"])]
    assert not capacity_rows["block_id"].isna().any()
    assert set(capacity_rows.groupby("market").size().index) == {"fcr", "afrr_pos", "afrr_neg"}
    # 2 days * 6 blocks per day = 12 rows per ancillary market
    assert (capacity_rows.groupby("market").size() == 12).all()

    e_rows = result.long_format[result.long_format["variable"] == "e"]
    assert e_rows["is_assumption"].all()  # include_activation_in_soc defaults True
    p_rows = result.long_format[result.long_format["variable"].isin(["p_ch", "p_dis"])]
    assert not p_rows["is_assumption"].any()


def test_daily_revenue_has_one_row_per_solved_day_and_matches_manual_solve():
    result = _run(dt.date(2026, 6, 15), dt.date(2026, 6, 15))
    assert len(result.daily_revenue) == 1
    row = result.daily_revenue.iloc[0]

    # Cross-check against calling build/solve/revenue directly, bypassing rolling.py.
    raw = load_raw_y_true(PATHS)
    day_prices = load_day_prices(dt.date(2026, 6, 15), raw, TIMINGS["day_ahead"], TIMINGS["fcr"])
    ancillary = AncillaryInputs(
        blocks=day_prices.grid.blocks,
        step_block_index=day_prices.grid.step_block_index,
        fcr_prices=day_prices.fcr,
        afrr_pos_prices=day_prices.afrr_pos,
        afrr_neg_prices=day_prices.afrr_neg,
        price_basis=PRICE_BASIS,
        assumptions=ASSUMPTIONS,
    )
    model = build_model(
        steps=day_prices.grid.steps, day_ahead_prices=day_prices.day_ahead, battery=BATTERY,
        delta_t_hours=day_prices.grid.delta_t_hours, ancillary=ancillary,
    )
    manual_result = solve_model(model, SOLVER)
    manual_revenue = compute_revenue(
        manual_result, day_prices.day_ahead, day_prices.grid.delta_t_hours, ancillary, battery=BATTERY,
    )

    assert row["total"] == pytest.approx(manual_revenue.net, rel=1e-6)
    assert row["net"] == pytest.approx(row["gross"] - row["degradation"])
    assert row["activation_is_assumption"] == True  # noqa: E712


def test_skipped_days_appear_in_skip_log_not_silently_dropped():
    result = _run(dt.date(2026, 3, 18), dt.date(2026, 3, 22))
    assert len(result.skip_log) == 3
    assert {s.day for s in result.skip_log} == {
        dt.date(2026, 3, 18), dt.date(2026, 3, 19), dt.date(2026, 3, 20),
    }
    assert all("day_ahead" in s.reason for s in result.skip_log)
    assert set(result.daily_revenue["solve_date"]) == {dt.date(2026, 3, 21), dt.date(2026, 3, 22)}


def test_cyclic_soc_true_days_are_solved_independently():
    two_day = _run(dt.date(2026, 6, 15), dt.date(2026, 6, 16), cyclic_soc=True)
    one_day = _run(dt.date(2026, 6, 16), dt.date(2026, 6, 16), cyclic_soc=True)

    two_day_second = two_day.daily_revenue[two_day.daily_revenue["solve_date"] == dt.date(2026, 6, 16)]
    assert two_day_second["total"].iloc[0] == pytest.approx(one_day.daily_revenue["total"].iloc[0], rel=1e-6)


def test_cyclic_soc_false_carries_e_across_day_boundary():
    result = _run(dt.date(2026, 6, 15), dt.date(2026, 6, 16), cyclic_soc=False)
    lf = result.long_format

    day1_last_e = lf[
        (lf["solve_date"] == dt.date(2026, 6, 15)) & (lf["variable"] == "e")
    ].sort_values("target_time")["value"].iloc[-1]

    day2_rows = lf[lf["solve_date"] == dt.date(2026, 6, 16)]
    day2_e = day2_rows[day2_rows["variable"] == "e"].sort_values("target_time")["value"]
    day2_p_ch = day2_rows[day2_rows["variable"] == "p_ch"].sort_values("target_time")["value"]
    day2_p_dis = day2_rows[day2_rows["variable"] == "p_dis"].sort_values("target_time")["value"]

    first_block_c_pos = day2_rows[(day2_rows["variable"] == "c_pos") & (day2_rows["block_id"] == 0)]["value"].iloc[0]
    first_block_c_neg = day2_rows[(day2_rows["variable"] == "c_neg") & (day2_rows["block_id"] == 0)]["value"].iloc[0]
    activation_0 = (
        ASSUMPTIONS.alpha_pos / BATTERY.eta * first_block_c_pos
        - ASSUMPTIONS.alpha_neg * BATTERY.eta * first_block_c_neg
    ) * 0.25

    expected_e0 = (
        day1_last_e
        + BATTERY.eta * 0.25 * day2_p_ch.iloc[0]
        - (0.25 / BATTERY.eta) * day2_p_dis.iloc[0]
        - BATTERY.aux_load_mw * 0.25
        - activation_0
    )
    assert day2_e.iloc[0] == pytest.approx(expected_e0, abs=1e-4)


# ---- forecast-driven: decide on y_pred, settle on y_true (SPEC.md §6a) ----

def _write_wrong_step_fake_repo(tmp_path, day: dt.date, model: str, quantile: float) -> PathsConfig:
    """One synthetic day, day-ahead only: y_true has the SAME cheap/
    expensive window shape as test_model_day_ahead.py's own known-answer
    test (cheap at steps[10:20], expensive at steps[50:60], flat 50
    elsewhere) -- but y_pred has the two windows SWAPPED. A day-ahead-only
    MILP decides on y_pred, so it schedules a full cyclic swing (verified
    empirically for this exact 96-step/flat-50-elsewhere shape: charge
    only in the predicted-cheap window, discharge only in the predicted-
    expensive window, nothing elsewhere) at exactly the WRONG times.
    Ancillary markets get flat, irrelevant prices -- load_day_prices still
    requires them present even though include_ancillary=False ignores them.
    """
    grid = build_day_grid(day, _SYNTH_DAY_AHEAD_TIMING, _SYNTH_FCR_TIMING)
    true_cheap, true_expensive = set(grid.steps[10:20]), set(grid.steps[50:60])
    pred_cheap, pred_expensive = true_expensive, true_cheap  # swapped

    def _price(step, cheap_set, expensive_set):
        if step in cheap_set:
            return 0.0
        if step in expensive_set:
            return 1000.0
        return 50.0

    rows = [
        {
            "market": "day_ahead", "target_time": step,
            "y_true": _price(step, true_cheap, true_expensive),
            "y_pred": _price(step, pred_cheap, pred_expensive),
        }
        for step in grid.steps
    ]
    for block in grid.blocks:
        for market in ("fcr", "afrr_pos", "afrr_neg"):
            rows.append({"market": market, "target_time": block.start, "y_true": 0.0, "y_pred": 0.0})

    df = pd.DataFrame(rows)
    df["model"] = model
    df["quantile"] = quantile
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    df.to_parquet(results_dir / "fake.parquet")
    return PathsConfig(forecast_repo_path=tmp_path)


def test_forecast_driven_wrong_step_settles_worse_than_perfect_foresight(tmp_path):
    """The most important new test for §6a: revenue settled at y_true for
    a forecast that points at the wrong step must equal a hand-computed
    value, and must be strictly worse than perfect foresight over the
    same day -- proving decide-on-forecast/settle-on-realized actually
    penalizes a bad forecast rather than quietly valuing it at its own
    (wrong) prices.

    Hand computation: the decision-side schedule (verified empirically --
    see the docstring on _write_wrong_step_fake_repo) charges a full
    cyclic swing (delta_e_max/eta raw MWh) only during the
    predicted-cheap window, and discharges the same swing (delta_e_max*eta
    MWh) only during the predicted-expensive window. Settling on y_true:
    the predicted-cheap window is where y_true is actually 1000 (the
    charge is a pure cost, no revenue there), and the predicted-expensive
    window is where y_true is actually 0 (the discharge earns nothing) --
    so settlement = -1000 * (delta_e_max / eta), a loss. Perfect foresight
    on the same day/shape is the mirror image: +1000 * delta_e_max * eta
    (matches tests/test_model_day_ahead.py's own known-answer test).
    """
    day = dt.date(2026, 6, 1)
    model, quantile = "wrong_step_model", 0.5
    fake_paths = _write_wrong_step_fake_repo(tmp_path, day, model, quantile)
    # The closed-form values below assume no aux drain and free cycling.
    battery = dataclasses.replace(BATTERY, aux_load_mw=0.0, degradation_cost_eur_mwh=0.0)

    forecast_result = run_rolling(
        start_date=day, end_date=day, battery=battery, solver_config=SOLVER,
        day_ahead_timing=_SYNTH_DAY_AHEAD_TIMING, capacity_timing=_SYNTH_FCR_TIMING,
        include_ancillary=False, paths_config=fake_paths,
        decision_source="forecast", forecast_model=model, forecast_quantile=quantile,
    )
    perfect_result = run_rolling(
        start_date=day, end_date=day, battery=battery, solver_config=SOLVER,
        day_ahead_timing=_SYNTH_DAY_AHEAD_TIMING, capacity_timing=_SYNTH_FCR_TIMING,
        include_ancillary=False, paths_config=fake_paths,
    )

    delta_e_max = BATTERY.soc_max_mwh - BATTERY.soc_min_mwh
    expected_settlement = -1000.0 * (delta_e_max / BATTERY.eta)
    expected_perfect = 1000.0 * delta_e_max * BATTERY.eta

    forecast_total = forecast_result.daily_revenue["total"].iloc[0]
    perfect_total = perfect_result.daily_revenue["total"].iloc[0]

    assert forecast_total == pytest.approx(expected_settlement, rel=1e-3)
    assert perfect_total == pytest.approx(expected_perfect, rel=1e-3)
    assert forecast_total < perfect_total


def test_write_rolling_run_results(tmp_path):
    start, end = dt.date(2026, 6, 15), dt.date(2026, 6, 15)
    result = _run(start, end)
    write_rolling_run_results(
        result, tmp_path,
        strategy="full_cooptimized", start_date=start, end_date=end,
        battery=BATTERY, solver_config=SOLVER, assumptions=ASSUMPTIONS, price_basis=PRICE_BASIS,
        paths_config=PATHS,
    )

    run_dir = tmp_path / result.solve_id
    assert (run_dir / "long_format.parquet").exists()
    assert (run_dir / "daily_revenue.parquet").exists()
    assert (run_dir / "skip_log.csv").exists()
    assert (run_dir / "run_meta.json").exists()

    reloaded = pd.read_parquet(run_dir / "long_format.parquet")
    assert len(reloaded) == len(result.long_format)
