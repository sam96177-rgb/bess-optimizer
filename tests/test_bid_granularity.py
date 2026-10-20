"""1 MW reserve bid granularity (PLAN_V4_REVISION.md section 4, SPEC.md C4/C8):
FCR/aFRR tender in 1 MW minimum bids, 1 MW increments (regelleistung.net).
Prices here are synthetic test inputs on the real day grid.
"""
from __future__ import annotations

import dataclasses
import math

import pandas as pd
import pytest

from src.model.build import FCR_CAP_FRACTION_OF_PMAX
from tests.dispatch_helpers import (
    BATTERY,
    FALL_BACK_DAY,
    NORMAL_DAY,
    PRICE_BASIS,
    SPRING_FORWARD_DAY,
    block_prices,
    grid_for,
    random_ancillary_case,
    solve_case,
)

TOL = 1e-6


def _assert_integer_mw(series: pd.Series) -> None:
    assert (series - series.round()).abs().max() < TOL


def _assert_all_close(series: pd.Series, value: float) -> None:
    assert (series - value).abs().max() < 1e-4


def _assert_on_cap_except_charging_block(series: pd.Series, cap: float) -> None:
    """With FCR counted 1.25 x (V8-2) the cap leaves no spare power: the battery must still charge
    somewhere to cover its standby load, and that costs FCR in one block. Every other block sits on the cap."""
    assert (series <= cap + 1e-4).all()
    assert ((series - cap).abs() < 1e-4).sum() >= len(series) - 1


@pytest.mark.parametrize("day", [NORMAL_DAY, SPRING_FORWARD_DAY, FALL_BACK_DAY])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_committed_reserve_is_integer_mw(day, seed):
    """market_prices.yaml's real bid_granularity_mw (1.0 for fcr/afrr_pos/afrr_neg)
    is on by default via PRICE_BASIS, so every random case already exercises it."""
    case = random_ancillary_case(day=day, seed=seed)
    _assert_integer_mw(case.result.c_fcr)
    _assert_integer_mw(case.result.c_pos)
    _assert_integer_mw(case.result.c_neg)


FCR_CAP_MW = FCR_CAP_FRACTION_OF_PMAX * BATTERY.power_mw  # 0.8 x 4.5 = 3.6: fractional by construction (C4)


def _fcr_only_case(price_basis) -> pd.Series:
    """Known answer: FCR priced high, aFRR+/- at zero, so C4's fractional cap
    (FCR_CAP_MW) is the only thing binding c_fcr -- every block should sit
    exactly on it."""
    grid = grid_for(NORMAL_DAY)
    day_ahead = pd.Series(0.0, index=grid.steps)
    fcr, pos, neg = block_prices(grid, 500.0), block_prices(grid, 0.0), block_prices(grid, 0.0)
    return solve_case(grid, day_ahead, fcr, pos, neg, price_basis=price_basis).result.c_fcr


def test_granularity_forces_an_otherwise_fractional_optimum_to_integer_mw():
    unconstrained = dataclasses.replace(PRICE_BASIS, bid_granularity_mw={})
    _assert_on_cap_except_charging_block(_fcr_only_case(unconstrained), FCR_CAP_MW)

    constrained = _fcr_only_case(PRICE_BASIS)  # granularity 1.0
    _assert_integer_mw(constrained)
    _assert_all_close(constrained, math.floor(FCR_CAP_MW))


def test_zero_granularity_reproduces_pre_v4_fractional_behaviour():
    zero = dataclasses.replace(PRICE_BASIS, bid_granularity_mw={m: 0.0 for m in PRICE_BASIS.bid_granularity_mw})
    for market in ("day_ahead", "fcr", "afrr_pos", "afrr_neg"):
        assert zero.granularity_for(market) == 0.0
    _assert_on_cap_except_charging_block(_fcr_only_case(zero), FCR_CAP_MW)
