import datetime as dt

import pandas as pd
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
from src.model.rolling import LONG_FORMAT_COLUMNS, run_rolling, write_rolling_run_results
from src.model.solve import solve_model
from src.valuation.revenue import compute_revenue

PATHS = load_paths_config()
TIMINGS = load_market_timing_configs(PATHS)
BATTERY = battery_from_config(load_battery_config())
SOLVER = load_solver_config()
ASSUMPTIONS = load_assumptions_config()
PRICE_BASIS = load_market_price_basis_config()


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
    manual_revenue = compute_revenue(manual_result, day_prices.day_ahead, day_prices.grid.delta_t_hours, ancillary)

    assert row["total"] == pytest.approx(manual_revenue.total, rel=1e-6)
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
        ASSUMPTIONS.alpha_pos * first_block_c_pos - ASSUMPTIONS.alpha_neg * first_block_c_neg
    ) * 0.25

    expected_e0 = (
        day1_last_e
        + BATTERY.eta * 0.25 * day2_p_ch.iloc[0]
        - (0.25 / BATTERY.eta) * day2_p_dis.iloc[0]
        - activation_0
    )
    assert day2_e.iloc[0] == pytest.approx(expected_e0, abs=1e-4)


def test_write_rolling_run_results(tmp_path):
    result = _run(dt.date(2026, 6, 15), dt.date(2026, 6, 15))
    write_rolling_run_results(result, tmp_path)

    run_dir = tmp_path / result.solve_id
    assert (run_dir / "long_format.parquet").exists()
    assert (run_dir / "daily_revenue.parquet").exists()
    assert (run_dir / "skip_log.csv").exists()

    reloaded = pd.read_parquet(run_dir / "long_format.parquet")
    assert len(reloaded) == len(result.long_format)
