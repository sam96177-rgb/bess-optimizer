"""Digital twin: hand-verifiable replay checks."""
import dataclasses
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from dispatch_helpers import (
    ASSUMPTIONS, BATTERY, NORMAL_DAY, PRICE_BASIS, SOLVER, TIMINGS, block_prices, grid_for, random_ancillary_case,
)
from src.data.grid import build_day_grid
from src.model.rolling import run_rolling, write_rolling_run_results
from src.twin.plot import plot_day
from src.twin.replay import DayPlan, TwinConfig, replay_day, replay_run, write_twin_outputs


def _plan_from_case(case) -> DayPlan:
    r = case.result
    return DayPlan(
        day=NORMAL_DAY, grid=case.grid, p_ch=r.p_ch, p_dis=r.p_dis, e=r.e,
        c_fcr=r.c_fcr, c_pos=r.c_pos, c_neg=r.c_neg,
    )


def _idle_plan(grid, c_pos=0.0, c_neg=0.0) -> DayPlan:
    zeros = pd.Series(0.0, index=grid.steps)
    return DayPlan(
        day=NORMAL_DAY, grid=grid, p_ch=zeros, p_dis=zeros, e=pd.Series(5.0, index=grid.steps),
        c_fcr=block_prices(grid, 0.0), c_pos=block_prices(grid, c_pos), c_neg=block_prices(grid, c_neg),
    )


def test_expected_mode_reproduces_the_plan_exactly():
    """With activation at its expected value the twin must trace the MILP's own SOC:
    proves the twin and planner share one physics."""
    case = random_ancillary_case(NORMAL_DAY, seed=1)
    assert case.result.c_pos.sum() > 0.1 and case.result.c_neg.sum() > 0.1
    plan = _plan_from_case(case)
    cfg = TwinConfig(n_draws=1, mode="expected", n_path_draws=1)
    summary, paths = replay_day(plan, float(plan.e.iloc[-1]), BATTERY, ASSUMPTIONS, cfg, np.random.default_rng(0))
    assert (paths["twin_e_mwh"] - paths["planned_e_mwh"]).abs().max() < 1e-5
    assert summary["n_soc_violation_steps"].iloc[0] == 0
    assert summary["dispatch_shortfall_mwh"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert summary["activation_delivered_mwh"].iloc[0] == pytest.approx(summary["activation_expected_mwh"].iloc[0], rel=1e-9)


def test_stochastic_requested_activation_has_the_planned_mean():
    case = random_ancillary_case(NORMAL_DAY, seed=2)
    plan = _plan_from_case(case)
    cfg = TwinConfig(n_draws=3000, seed=5)
    summary, _ = replay_day(plan, float(plan.e.iloc[-1]), BATTERY, ASSUMPTIONS, cfg, np.random.default_rng(5))
    requested = (
        summary["activation_delivered_mwh"] + summary["activation_shortfall_up_mwh"] + summary["activation_shortfall_down_mwh"]
    )
    expected = summary["activation_expected_mwh"].iloc[0]
    standard_error = requested.std() / np.sqrt(len(requested))
    assert abs(requested.mean() - expected) < 4 * standard_error


def test_activation_never_in_both_directions_in_one_step():
    case = random_ancillary_case(NORMAL_DAY, seed=3)
    plan = _plan_from_case(case)
    cfg = TwinConfig(n_draws=50, n_path_draws=50, seed=1)
    _, paths = replay_day(plan, float(plan.e.iloc[-1]), BATTERY, ASSUMPTIONS, cfg, np.random.default_rng(1))
    assert not ((paths["activation_up_mw"] > 0) & (paths["activation_down_mw"] > 0)).any()


def test_empty_battery_cannot_deliver_upward_activation_hand_check():
    """Idle battery 0.001 MWh above soc_min, no aux, alpha_pos=1 (called every step at 4 MW).
    Step 1 delivers only what is left: 0.001 stored -> 0.001*eta MWh at the meter; every later
    step delivers nothing. Requested = 4 MW * 0.25 h per step."""
    grid = grid_for(NORMAL_DAY)
    battery = dataclasses.replace(BATTERY, aux_load_mw=0.0)
    assumptions = dataclasses.replace(ASSUMPTIONS, alpha_pos=1.0, alpha_neg=0.0)
    plan = _idle_plan(grid, c_pos=4.0)
    cfg = TwinConfig(n_draws=4, n_path_draws=1, call_degree_mean=1.0)
    summary, paths = replay_day(plan, battery.soc_min_mwh + 0.001, battery, assumptions, cfg, np.random.default_rng(0))

    n = len(grid.steps)
    requested = 4.0 * grid.delta_t_hours * n
    delivered = 0.001 * battery.eta
    row = summary.iloc[0]
    assert row["activation_delivered_mwh"] == pytest.approx(delivered)
    assert row["activation_shortfall_up_mwh"] == pytest.approx(requested - delivered)
    assert row["n_soc_violation_steps"] == n  # first step is partial, every later one fails
    assert paths["twin_e_mwh"].min() >= battery.soc_min_mwh - 1e-12  # SOC is clipped, never breached
    assert row["min_soc_mwh"] == pytest.approx(battery.soc_min_mwh)


def test_full_battery_cannot_absorb_downward_activation_hand_check():
    grid = grid_for(NORMAL_DAY)
    battery = dataclasses.replace(BATTERY, aux_load_mw=0.0)
    assumptions = dataclasses.replace(ASSUMPTIONS, alpha_pos=0.0, alpha_neg=1.0)
    plan = _idle_plan(grid, c_neg=4.0)
    summary, _ = replay_day(
        plan, battery.soc_max_mwh - 0.001, battery, assumptions, TwinConfig(n_draws=2, n_path_draws=1, call_degree_mean=1.0),
        np.random.default_rng(0),
    )
    row = summary.iloc[0]
    assert row["activation_delivered_mwh"] == pytest.approx(0.001 / battery.eta)
    assert row["max_soc_mwh"] == pytest.approx(battery.soc_max_mwh)
    assert row["activation_shortfall_down_mwh"] == pytest.approx(4.0 * 0.25 * len(grid.steps) - 0.001 / battery.eta)


def test_planned_discharge_from_an_empty_battery_is_logged_as_dispatch_shortfall():
    grid = grid_for(NORMAL_DAY)
    battery = dataclasses.replace(BATTERY, aux_load_mw=0.0)
    p_dis = pd.Series(0.0, index=grid.steps)
    p_dis.iloc[0] = 2.0  # 2 MW for 0.25 h = 0.5 MWh at the meter needs 0.5/eta stored
    plan = dataclasses.replace(_idle_plan(grid), p_dis=p_dis)
    assumptions = dataclasses.replace(ASSUMPTIONS, alpha_pos=0.0, alpha_neg=0.0)
    summary, _ = replay_day(
        plan, battery.soc_min_mwh, battery, assumptions, TwinConfig(n_draws=1, n_path_draws=1), np.random.default_rng(0),
    )
    assert summary["dispatch_shortfall_mwh"].iloc[0] == pytest.approx(0.5)
    assert summary["n_soc_violation_steps"].iloc[0] == 1


def test_partial_calls_keep_the_mean_but_cut_the_soc_spread():
    """Same expected activation, but small frequent calls wander far less than
    always-full calls -- the point of the gentler default."""
    case = random_ancillary_case(NORMAL_DAY, seed=6)
    plan = _plan_from_case(case)
    e0 = float(plan.e.iloc[-1])
    run = lambda m: replay_day(
        plan, e0, BATTERY, ASSUMPTIONS, TwinConfig(n_draws=1500, seed=2, call_degree_mean=m), np.random.default_rng(2),
    )[0]
    full, partial = run(1.0), run(0.25)

    def requested(s):
        return s["activation_delivered_mwh"] + s["activation_shortfall_up_mwh"] + s["activation_shortfall_down_mwh"]

    expected = full["activation_expected_mwh"].iloc[0]
    for s in (full, partial):
        assert abs(requested(s).mean() - expected) < 4 * requested(s).std() / np.sqrt(len(s))
    assert partial["max_abs_soc_deviation_mwh"].mean() < 0.7 * full["max_abs_soc_deviation_mwh"].mean()


def test_partial_calls_hit_the_soc_window_less_often_from_mid_window():
    """From mid-window (nothing pinned at a limit) smaller calls wander less, so fewer days touch a limit.
    (A plan already sitting at a limit can go the other way: any call there is a shortfall.)"""
    grid = grid_for(NORMAL_DAY)
    plan = _idle_plan(grid, c_pos=4.0, c_neg=4.0)
    run = lambda m: replay_day(
        plan, 5.0, BATTERY, ASSUMPTIONS, TwinConfig(n_draws=1500, seed=4, call_degree_mean=m), np.random.default_rng(4),
    )[0]
    full, partial = run(1.0), run(0.25)
    assert partial["n_soc_violation_steps"].gt(0).mean() < full["n_soc_violation_steps"].gt(0).mean()


def test_a_call_never_exceeds_the_committed_capacity():
    case = random_ancillary_case(NORMAL_DAY, seed=7)
    plan = _plan_from_case(case)
    _, paths = replay_day(
        plan, float(plan.e.iloc[-1]), BATTERY, ASSUMPTIONS, TwinConfig(n_draws=40, n_path_draws=40, seed=3),
        np.random.default_rng(3),
    )
    grid = case.grid
    cap_up = pd.Series([float(plan.c_pos.loc[b]) for b in grid.step_block_index], index=grid.steps)
    per_step_max = paths.groupby("target_time")["activation_up_mw"].max()
    assert (per_step_max <= cap_up.reindex(per_step_max.index) + 1e-9).all()
    assert (paths["activation_up_mw"] > 0).any()


def test_same_seed_is_reproducible_and_a_different_seed_differs():
    plan = _plan_from_case(random_ancillary_case(NORMAL_DAY, seed=4))
    e0 = float(plan.e.iloc[-1])
    run = lambda seed: replay_day(plan, e0, BATTERY, ASSUMPTIONS, TwinConfig(n_draws=20, seed=seed), np.random.default_rng(seed))[0]
    pd.testing.assert_frame_equal(run(7), run(7))
    assert not run(7)["activation_delivered_mwh"].equals(run(8)["activation_delivered_mwh"])


def test_alpha_sum_above_one_is_rejected():
    plan = _idle_plan(grid_for(NORMAL_DAY), c_pos=1.0)
    bad = dataclasses.replace(ASSUMPTIONS, alpha_pos=0.7, alpha_neg=0.5)
    with pytest.raises(ValueError, match="alpha_pos \\+ alpha_neg"):
        replay_day(plan, 5.0, BATTERY, bad, TwinConfig(n_draws=1, call_degree_mean=1.0), np.random.default_rng(0))
    # A small mean call size raises the call probability: alpha 0.05 + 0.05 at m = 0.05 would need q > 1.
    with pytest.raises(ValueError, match="call_degree_mean"):
        replay_day(plan, 5.0, BATTERY, ASSUMPTIONS, TwinConfig(n_draws=1, call_degree_mean=0.05), np.random.default_rng(0))


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        TwinConfig(mode="bogus")


@pytest.mark.parametrize("kwargs", [{"call_degree_mean": 0.0}, {"call_degree_mean": 1.5}, {"call_degree_concentration": 0.0}])
def test_config_rejects_bad_call_shape(kwargs):
    with pytest.raises(ValueError, match="call_degree"):
        TwinConfig(**kwargs)


def test_replay_run_on_a_stored_real_run(tmp_path):
    from src.config import load_paths_config

    paths_cfg = load_paths_config()
    day = dt.date(2026, 6, 15)
    result = run_rolling(
        start_date=day, end_date=day, battery=BATTERY, solver_config=SOLVER, assumptions=ASSUMPTIONS,
        price_basis=PRICE_BASIS, day_ahead_timing=TIMINGS["day_ahead"], capacity_timing=TIMINGS["fcr"],
        paths_config=paths_cfg,
    )
    write_rolling_run_results(
        result, tmp_path, strategy="full_cooptimized", start_date=day, end_date=day, battery=BATTERY,
        solver_config=SOLVER, assumptions=ASSUMPTIONS, price_basis=PRICE_BASIS, paths_config=paths_cfg,
    )
    run_dir = tmp_path / result.solve_id
    grid_for_day = lambda d: build_day_grid(d, TIMINGS["day_ahead"], TIMINGS["fcr"])

    # Expected activation must trace the stored plan; stochastic must respect the SOC window.
    summary, paths, meta = replay_run(run_dir, grid_for_day, TwinConfig(n_draws=1, mode="expected", n_path_draws=1))
    assert (paths["twin_e_mwh"] - paths["planned_e_mwh"]).abs().max() < 1e-4
    assert summary["n_soc_violation_steps"].sum() == 0

    summary, paths, meta = replay_run(run_dir, grid_for_day, TwinConfig(n_draws=25, seed=3, n_path_draws=2))
    assert len(summary) == 25 and summary["solve_date"].nunique() == 1
    assert (summary["min_soc_mwh"] >= BATTERY.soc_min_mwh - 1e-9).all()
    assert (summary["max_soc_mwh"] <= BATTERY.soc_max_mwh + 1e-9).all()
    assert (summary["n_power_violation_steps"] == 0).all()
    assert meta["alpha_pos"] == ASSUMPTIONS.alpha_pos and any("PLACEHOLDER" in a for a in meta["assumptions"])

    write_twin_outputs(run_dir, summary, paths, meta)
    for name in ("twin_summary.parquet", "twin_paths.parquet", "twin_meta.json"):
        assert (run_dir / name).exists()
    assert json.loads((run_dir / "twin_meta.json").read_text(encoding="utf-8"))["n_draws"] == 25

    png = plot_day(paths, day, BATTERY, run_dir / "twin.png")
    assert png.exists() and png.stat().st_size > 5000
