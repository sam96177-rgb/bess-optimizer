"""Gate-ordered (sequential) clearing: FCR -> aFRR -> DA."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from tests.dispatch_helpers import (
    ASSUMPTIONS, BATTERY, NORMAL_DAY, PRICE_BASIS, SOLVER, TIMINGS, block_prices, grid_for, random_day_ahead,
)
from src.batch import build_matrix
from src.config import MarketTimingConfig, PathsConfig, load_paths_config
from src.data.forecast_loader import load_price_pair_range
from src.model.build import AncillaryInputs, FixedReserves, build_model
from src.model.rolling import run_rolling, write_rolling_run_results
from src.model.sequential import solve_day_sequential
from src.model.solve import solve_model

TOL = 1e-6


def _ancillary(grid, seed: int) -> AncillaryInputs:
    rng = np.random.default_rng(seed + 100)
    n = len(grid.blocks)
    return AncillaryInputs(
        blocks=grid.blocks, step_block_index=grid.step_block_index,
        fcr_prices=block_prices(grid, rng.uniform(20, 150, n)),
        afrr_pos_prices=block_prices(grid, rng.uniform(1, 30, n)),
        afrr_neg_prices=block_prices(grid, rng.uniform(1, 30, n)),
        price_basis=PRICE_BASIS, assumptions=ASSUMPTIONS,
    )


def _sequential(grid, ancillary, reserve_stage_da, final_da):
    return solve_day_sequential(
        grid.steps, reserve_stage_da, final_da, BATTERY, grid.delta_t_hours, SOLVER, ancillary,
    )


def _simultaneous(grid, ancillary, da):
    return solve_model(
        build_model(grid.steps, da, BATTERY, grid.delta_t_hours, ancillary=ancillary), SOLVER,
    )


def test_fixed_reserves_are_honoured():
    grid = grid_for(NORMAL_DAY)
    ancillary = _ancillary(grid, 0)
    da = random_day_ahead(grid, 0)
    fixed = FixedReserves(fcr=block_prices(grid, 1.5), pos=block_prices(grid, 0.7), neg=block_prices(grid, 0.0))
    result = solve_model(
        build_model(grid.steps, da, BATTERY, grid.delta_t_hours, ancillary=ancillary, fixed_reserves=fixed), SOLVER,
    )
    assert (result.c_fcr - 1.5).abs().max() < TOL
    assert (result.c_pos - 0.7).abs().max() < TOL
    assert result.c_neg.abs().max() < TOL


def test_partial_fix_leaves_other_reserves_free():
    grid = grid_for(NORMAL_DAY)
    ancillary = _ancillary(grid, 1)
    fixed = FixedReserves(fcr=block_prices(grid, 0.0))
    result = solve_model(
        build_model(
            grid.steps, random_day_ahead(grid, 1), BATTERY, grid.delta_t_hours,
            ancillary=ancillary, fixed_reserves=fixed,
        ),
        SOLVER,
    )
    assert result.c_fcr.abs().max() < TOL
    assert result.c_pos.sum() + result.c_neg.sum() > 0.1


def test_fixed_reserves_require_ancillary():
    grid = grid_for(NORMAL_DAY)
    with pytest.raises(ValueError, match="requires ancillary"):
        build_model(
            grid.steps, random_day_ahead(grid, 0), BATTERY, grid.delta_t_hours,
            fixed_reserves=FixedReserves(fcr=block_prices(grid, 1.0)),
        )


def test_identical_information_reproduces_the_simultaneous_optimum():
    """The reason sequential clearing only matters through information: with
    the same prices at every stage the staged solves land on the joint optimum."""
    grid = grid_for(NORMAL_DAY)
    ancillary = _ancillary(grid, 2)
    da = random_day_ahead(grid, 2)
    seq = _sequential(grid, ancillary, da, da)
    sim = _simultaneous(grid, ancillary, da)
    assert seq.objective_value == pytest.approx(sim.objective_value, rel=1e-6)


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_less_informed_reserve_stage_can_never_beat_simultaneous(seed):
    """Both objectives are evaluated on the SAME final prices, and the
    sequential schedule is a feasible point of the simultaneous problem, so
    sequential <= simultaneous is guaranteed."""
    grid = grid_for(NORMAL_DAY)
    ancillary = _ancillary(grid, seed)
    final_da = random_day_ahead(grid, seed)
    proxy_da = random_day_ahead(grid, seed + 50)
    seq = _sequential(grid, ancillary, proxy_da, final_da)
    sim = _simultaneous(grid, ancillary, final_da)
    assert seq.objective_value <= sim.objective_value + 1e-6


def test_a_bad_reserve_stage_view_actually_costs_money():
    """Reserve stage believes prices are flat; the real day has a huge
    evening spike. It commits reserves over the spike hours, blocking
    arbitrage the simultaneous model would take."""
    grid = grid_for(NORMAL_DAY)
    n = len(grid.blocks)
    ancillary = AncillaryInputs(
        blocks=grid.blocks, step_block_index=grid.step_block_index,
        fcr_prices=block_prices(grid, [30.0] * n), afrr_pos_prices=block_prices(grid, 0.0),
        afrr_neg_prices=block_prices(grid, 0.0),
        price_basis=PRICE_BASIS, assumptions=ASSUMPTIONS,
    )
    flat = pd.Series(50.0, index=grid.steps)
    spike = flat.copy()
    spike.iloc[72:80] = 1500.0  # block 4
    spike.iloc[8:16] = 0.0  # cheap charging window in block 0
    seq = _sequential(grid, ancillary, flat, spike)
    sim = _simultaneous(grid, ancillary, spike)
    assert seq.objective_value < sim.objective_value - 1.0


def test_final_reserves_are_the_ones_committed_at_earlier_stages():
    grid = grid_for(NORMAL_DAY)
    ancillary = _ancillary(grid, 6)
    proxy_da = random_day_ahead(grid, 56)
    final_da = random_day_ahead(grid, 6)
    seq = _sequential(grid, ancillary, proxy_da, final_da)

    stage1 = _simultaneous(grid, ancillary, proxy_da)  # the FCR stage IS a full joint solve on the proxy view
    assert (seq.c_fcr - stage1.c_fcr).abs().max() < 1e-4


def test_sequential_result_respects_battery_physics():
    grid = grid_for(NORMAL_DAY)
    seq = _sequential(grid, _ancillary(grid, 7), random_day_ahead(grid, 57), random_day_ahead(grid, 7))
    assert (seq.e >= BATTERY.soc_min_mwh - 1e-5).all() and (seq.e <= BATTERY.soc_max_mwh + 1e-5).all()
    assert (seq.p_ch * seq.p_dis).abs().max() < 1e-6


# ---- loader: reserve-stage DA proxy -----------------------------------------

_DA = MarketTimingConfig(
    name="day_ahead", timezone="Europe/Berlin", dst_convention="variable_period_count",
    product_length_hours=0.25, delivery_start_local=dt.time(0, 0),
)
_CAP = MarketTimingConfig(
    name="fcr", timezone="Europe/Berlin", dst_convention="fixed_period_count",
    product_length_hours=4, delivery_start_local=dt.time(0, 0),
)


def _fake_repo(tmp_path, day, proxy_da_nan: bool = False) -> PathsConfig:
    grid = grid_for(day)
    rows = []
    for model, da_level in (("main", 100.0), ("proxy", 40.0)):
        for step in grid.steps:
            value = float("nan") if (model == "proxy" and proxy_da_nan and step == grid.steps[3]) else da_level
            rows.append({"market": "day_ahead", "model": model, "quantile": 0.5, "target_time": step,
                         "y_pred": value, "y_true": 55.0})
        for block in grid.blocks:
            for market in ("fcr", "afrr_pos", "afrr_neg"):
                rows.append({"market": market, "model": model, "quantile": 0.5, "target_time": block.start,
                             "y_pred": 5.0, "y_true": 6.0})
    (tmp_path / "results").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "results" / "fake.parquet")
    return PathsConfig(forecast_repo_path=tmp_path)


def _pairs(paths, **kwargs):
    return load_price_pair_range(
        NORMAL_DAY, NORMAL_DAY, _DA, _CAP, paths, decision_source="forecast",
        forecast_model="main", forecast_quantile=0.5, **kwargs,
    )


def test_loader_builds_reserve_stage_view_with_proxy_day_ahead(tmp_path):
    pairs, skipped = _pairs(_fake_repo(tmp_path, NORMAL_DAY), da_proxy_model="proxy")
    assert not skipped
    pair = pairs[0]
    assert (pair.decision.day_ahead == 100.0).all()
    assert (pair.reserve_stage_decision.day_ahead == 40.0).all()
    assert (pair.settlement.day_ahead == 55.0).all()
    # Only the day-ahead leg differs; reserve prices are the chosen model's.
    assert (pair.reserve_stage_decision.fcr == pair.decision.fcr).all()


def test_loader_without_proxy_has_no_reserve_stage_view(tmp_path):
    pairs, _ = _pairs(_fake_repo(tmp_path, NORMAL_DAY))
    assert pairs[0].reserve_stage_decision is None


def test_loader_skips_day_when_proxy_day_ahead_is_incomplete(tmp_path):
    pairs, skipped = _pairs(_fake_repo(tmp_path, NORMAL_DAY, proxy_da_nan=True), da_proxy_model="proxy")
    assert pairs == []
    assert "reserve-stage DA proxy (proxy)" in skipped[0].reason


def test_proxy_rejected_for_realized_runs(tmp_path):
    with pytest.raises(ValueError, match="only applies to forecast-driven"):
        load_price_pair_range(
            NORMAL_DAY, NORMAL_DAY, _DA, _CAP, _fake_repo(tmp_path, NORMAL_DAY), da_proxy_model="proxy",
        )


# ---- batch matrix -----------------------------------------------------------

def test_batch_matrix_has_a_perfect_foresight_run_for_full_sequential():
    combos = build_matrix(["m1", "m2"], [0.5], ["full_cooptimized", "full_sequential"], include_perfect_foresight=True)
    assert ("full_cooptimized", "realized", None, None) in combos
    assert ("full_sequential", "realized", None, None) in combos
    assert ("full_sequential", "forecast", "m1", 0.5) in combos
    assert ("full_sequential", "forecast", "m2", 0.5) in combos


# ---- end to end on real data --------------------------------------------------

def test_run_rolling_sequential_end_to_end_on_real_data(tmp_path):
    paths = load_paths_config()
    day = dt.date(2026, 6, 15)
    kwargs = dict(
        start_date=day, end_date=day, battery=BATTERY, solver_config=SOLVER, assumptions=ASSUMPTIONS,
        price_basis=PRICE_BASIS, day_ahead_timing=TIMINGS["day_ahead"], capacity_timing=TIMINGS["fcr"],
        paths_config=paths, decision_source="forecast", forecast_model="lgbm_point", forecast_quantile=0.5,
    )
    result = run_rolling(sequential=True, da_proxy_model="seasonal_naive", **kwargs)
    assert result.da_proxy_model == "seasonal_naive"
    assert len(result.daily_revenue) == 1
    row = result.daily_revenue.iloc[0]
    assert row["net"] == pytest.approx(row["gross"] - row["degradation"])

    write_rolling_run_results(
        result, tmp_path, strategy="full_sequential", start_date=day, end_date=day, battery=BATTERY,
        solver_config=SOLVER, assumptions=ASSUMPTIONS, price_basis=PRICE_BASIS, paths_config=paths,
    )
    meta = json.loads((tmp_path / result.solve_id / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["strategy"] == "full_sequential" and meta["da_proxy_model"] == "seasonal_naive"


def test_da_proxy_requires_ancillary_markets():
    with pytest.raises(ValueError, match="requires include_ancillary"):
        run_rolling(
            start_date=NORMAL_DAY, end_date=NORMAL_DAY, battery=BATTERY, solver_config=SOLVER,
            day_ahead_timing=TIMINGS["day_ahead"], capacity_timing=TIMINGS["fcr"], include_ancillary=False,
            decision_source="forecast", forecast_model="lgbm_point", forecast_quantile=0.5,
            sequential=True,
        )


def test_da_proxy_requires_sequential_flag():
    with pytest.raises(ValueError, match="only applies to sequential"):
        run_rolling(
            start_date=NORMAL_DAY, end_date=NORMAL_DAY, battery=BATTERY, solver_config=SOLVER,
            assumptions=ASSUMPTIONS, price_basis=PRICE_BASIS,
            day_ahead_timing=TIMINGS["day_ahead"], capacity_timing=TIMINGS["fcr"],
            decision_source="forecast", forecast_model="lgbm_point", forecast_quantile=0.5,
            da_proxy_model="seasonal_naive",
        )


def _real_kwargs(day):
    return dict(
        start_date=day, end_date=day, battery=BATTERY, solver_config=SOLVER, assumptions=ASSUMPTIONS,
        price_basis=PRICE_BASIS, day_ahead_timing=TIMINGS["day_ahead"], capacity_timing=TIMINGS["fcr"],
        paths_config=load_paths_config(),
    )


def test_perfect_foresight_sequential_matches_full_cooptimized_on_real_prices():
    """Every stage sees the real prices, so gate ordering costs nothing."""
    kwargs = _real_kwargs(dt.date(2026, 6, 15))
    sim = run_rolling(**kwargs)
    seq = run_rolling(sequential=True, **kwargs)
    assert seq.daily_revenue["net"].iloc[0] == pytest.approx(sim.daily_revenue["net"].iloc[0], rel=1e-3)


def test_forecast_sequential_with_same_model_at_every_stage_matches_simultaneous():
    kwargs = _real_kwargs(dt.date(2026, 6, 15)) | dict(
        decision_source="forecast", forecast_model="lgbm_point", forecast_quantile=0.5,
    )
    sim = run_rolling(**kwargs)
    seq = run_rolling(sequential=True, **kwargs)
    assert seq.da_proxy_model is None
    assert seq.daily_revenue["net"].iloc[0] == pytest.approx(sim.daily_revenue["net"].iloc[0], rel=1e-3)
