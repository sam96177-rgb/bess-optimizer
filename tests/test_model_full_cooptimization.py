import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.config import (
    load_assumptions_config,
    load_battery_config,
    load_market_price_basis_config,
    load_market_timing_configs,
    load_solver_config,
)
from src.data.grid import build_day_grid
from src.model.battery import battery_from_config
from src.model.build import AncillaryInputs, build_model
from src.model.solve import solve_model

TOL = 1e-4

BATTERY = battery_from_config(load_battery_config())
SOLVER = load_solver_config()
ASSUMPTIONS = load_assumptions_config()
PRICE_BASIS = load_market_price_basis_config()
TIMINGS = load_market_timing_configs()

DAY = dt.date(2026, 6, 1)  # ordinary day, no DST nearby
GRID = build_day_grid(DAY, TIMINGS["day_ahead"], TIMINGS["fcr"])
N_BLOCKS = len(GRID.blocks)


def _random_prices(seed: int, low: float, high: float) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(low, high, size=len(GRID.steps)), index=GRID.steps)


def _block_prices(value: float | list[float]) -> pd.Series:
    if isinstance(value, (int, float)):
        value = [value] * N_BLOCKS
    return pd.Series(value, index=[b.index for b in GRID.blocks])


def _build_and_solve(
    day_ahead_prices: pd.Series,
    fcr_prices: pd.Series,
    afrr_pos_prices: pd.Series,
    afrr_neg_prices: pd.Series,
    enforce_exclusivity: bool = True,
):
    ancillary = AncillaryInputs(
        blocks=GRID.blocks,
        step_block_index=GRID.step_block_index,
        fcr_prices=fcr_prices,
        afrr_pos_prices=afrr_pos_prices,
        afrr_neg_prices=afrr_neg_prices,
        price_basis=PRICE_BASIS,
        assumptions=ASSUMPTIONS,
    )
    model = build_model(
        steps=GRID.steps,
        day_ahead_prices=day_ahead_prices,
        battery=BATTERY,
        delta_t_hours=GRID.delta_t_hours,
        enforce_exclusivity=enforce_exclusivity,
        ancillary=ancillary,
    )
    return solve_model(model, SOLVER)


def _random_scenario(seed: int = 0):
    return _build_and_solve(
        day_ahead_prices=_random_prices(seed, -20, 200),
        fcr_prices=_block_prices(list(np.random.default_rng(seed).uniform(20, 150, N_BLOCKS))),
        afrr_pos_prices=_block_prices(list(np.random.default_rng(seed + 1).uniform(1, 30, N_BLOCKS))),
        afrr_neg_prices=_block_prices(list(np.random.default_rng(seed + 2).uniform(1, 30, N_BLOCKS))),
    )


# ---- C4 — FCR cap -----------------------------------------------------

def test_fcr_never_exceeds_80pct_of_pmax():
    result = _random_scenario()
    assert (result.c_fcr <= 0.8 * BATTERY.power_mw + TOL).all()


# ---- C3 — shared power limits -------------------------------------------

def test_committed_capacity_plus_dispatch_never_breaches_pmax():
    result = _random_scenario()
    for step, block_idx in zip(GRID.steps, GRID.step_block_index):
        fcr_b = result.c_fcr.loc[block_idx]
        pos_b = result.c_pos.loc[block_idx]
        neg_b = result.c_neg.loc[block_idx]
        assert result.p_ch.loc[step] + fcr_b + neg_b <= BATTERY.power_mw + TOL
        assert result.p_dis.loc[step] + fcr_b + pos_b <= BATTERY.power_mw + TOL


# ---- C7 — capacity constant within block --------------------------------

def test_capacity_is_one_value_per_block_not_per_step():
    result = _random_scenario()
    assert len(result.c_fcr) == N_BLOCKS
    assert len(result.c_pos) == N_BLOCKS
    assert len(result.c_neg) == N_BLOCKS


# ---- C2 — SOC headroom is genuinely reserved -----------------------------

def test_soc_reserve_constraint_holds_for_solved_values():
    result = _random_scenario()
    for step, block_idx in zip(GRID.steps, GRID.step_block_index):
        fcr_b = result.c_fcr.loc[block_idx]
        pos_b = result.c_pos.loc[block_idx]
        neg_b = result.c_neg.loc[block_idx]
        low = BATTERY.soc_min_mwh + fcr_b * ASSUMPTIONS.fcr_reserve_hours + pos_b * ASSUMPTIONS.afrr_reserve_hours
        high = BATTERY.soc_max_mwh - (fcr_b * ASSUMPTIONS.fcr_reserve_hours + neg_b * ASSUMPTIONS.afrr_reserve_hours)
        assert result.e.loc[step] >= low - TOL
        assert result.e.loc[step] <= high + TOL


def test_reserved_headroom_actually_restricts_usable_soc_vs_no_ancillary():
    """Construct a case where high FCR/afrr prices push the optimizer to
    commit heavily in one block, while day-ahead prices simultaneously
    make that block's steps the cheapest of the day (so without C2 the
    optimizer would want to charge to soc_max there too). Ignoring C2
    would let the model treat that capacity commitment as compatible with
    a full-to-soc_max charge -- physically it isn't, since the committed
    capacity must keep energy headroom in reserve. Confirm C2 actually
    caps SOC below the unconstrained ceiling in that block.
    """
    day_ahead_prices = pd.Series(80.0, index=GRID.steps)
    target_block = GRID.blocks[2]
    target_steps = GRID.steps[[bi == target_block.index for bi in GRID.step_block_index]]
    day_ahead_prices.loc[target_steps] = 0.0  # cheapest steps of the day

    high_fcr = _block_prices(0.0)
    high_fcr.loc[target_block.index] = 500.0  # make max FCR commitment attractive there
    zero_afrr = _block_prices(0.0)

    result = _build_and_solve(day_ahead_prices, high_fcr, zero_afrr, zero_afrr)

    committed_fcr = result.c_fcr.loc[target_block.index]
    assert committed_fcr > 0.1 * BATTERY.power_mw  # confirms the incentive actually bit

    reserved = committed_fcr * ASSUMPTIONS.fcr_reserve_hours
    assert result.e.loc[target_steps].max() <= BATTERY.soc_max_mwh - reserved + TOL
    # And it's a REAL restriction, not a coincidence: strictly below the
    # raw ceiling by (approximately) the reserved amount.
    assert result.e.loc[target_steps].max() < BATTERY.soc_max_mwh - TOL


# ---- monotonicity --------------------------------------------------------

def test_raising_fcr_price_does_not_decrease_committed_capacity():
    flat_day_ahead = pd.Series(80.0, index=GRID.steps)
    zero_afrr = _block_prices(0.0)

    low_price_result = _build_and_solve(flat_day_ahead, _block_prices(5.0), zero_afrr, zero_afrr)
    high_price_result = _build_and_solve(flat_day_ahead, _block_prices(500.0), zero_afrr, zero_afrr)

    assert high_price_result.c_fcr.sum() >= low_price_result.c_fcr.sum() - TOL
    assert high_price_result.c_fcr.sum() > low_price_result.c_fcr.sum()


def test_raising_afrr_pos_price_does_not_decrease_committed_capacity():
    flat_day_ahead = pd.Series(80.0, index=GRID.steps)
    zero = _block_prices(0.0)

    low_price_result = _build_and_solve(flat_day_ahead, zero, _block_prices(1.0), zero)
    high_price_result = _build_and_solve(flat_day_ahead, zero, _block_prices(200.0), zero)

    assert high_price_result.c_pos.sum() >= low_price_result.c_pos.sum() - TOL
    assert high_price_result.c_pos.sum() > low_price_result.c_pos.sum()


# ---- C5 with enforce_exclusivity=False ----------------------------------

def test_enforce_exclusivity_false_reports_simultaneous_charge_discharge():
    """SPEC.md asks us to check empirically, not assume, whether the LP
    relaxation ever wants both p_ch>0 and p_dis>0 in the same step once
    capacity payments are in the objective. This constructs the scenario
    most likely to trigger it (high symmetric FCR price, which pays for
    power headroom on both sides regardless of net dispatch) and records
    what actually happens.
    """
    flat_day_ahead = pd.Series(80.0, index=GRID.steps)
    high_fcr = _block_prices(300.0)
    zero_afrr = _block_prices(0.0)

    result = _build_and_solve(
        flat_day_ahead, high_fcr, zero_afrr, zero_afrr, enforce_exclusivity=False
    )
    both_active = (result.p_ch > 1e-6) & (result.p_dis > 1e-6)
    # Empirical finding (see Milestone 4 report): record it either way
    # rather than asserting a preconceived answer, per SPEC.md's own
    # framing ("worth knowing rather than assuming").
    print(f"\nsimultaneous charge+discharge steps with exclusivity off: {both_active.sum()} / {len(both_active)}")
