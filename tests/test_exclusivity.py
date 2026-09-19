"""Charge/discharge exclusivity: never both in the same step, and the binary
u is consistent with which side is active."""
import pandas as pd
import pytest

from dispatch_helpers import (
    BATTERY, NORMAL_DAY, SPRING_FORWARD_DAY, block_prices, grid_for, random_ancillary_case,
    solve_case,
)

TOL = 1e-6


def _assert_exclusive(result) -> None:
    assert (result.p_ch * result.p_dis).abs().max() < TOL
    assert set(result.u.round().astype(int)) <= {0, 1}
    assert (result.u - result.u.round()).abs().max() < TOL
    # binary consistency: u=1 permits only charging, u=0 only discharging
    assert (result.p_ch <= BATTERY.power_mw * result.u + TOL).all()
    assert (result.p_dis <= BATTERY.power_mw * (1 - result.u) + TOL).all()


@pytest.mark.parametrize("day", [NORMAL_DAY, SPRING_FORWARD_DAY])
@pytest.mark.parametrize("seed", [0, 1])
def test_exclusive_on_random_days_with_ancillary(day, seed):
    _assert_exclusive(random_ancillary_case(day, seed=seed).result)


def test_exclusive_when_prices_are_negative_all_day():
    grid = grid_for(NORMAL_DAY)
    case = solve_case(grid, pd.Series(-50.0, index=grid.steps), with_ancillary=False)
    _assert_exclusive(case.result)


def test_exclusive_with_very_high_symmetric_fcr_price():
    grid = grid_for(NORMAL_DAY)
    case = solve_case(grid, pd.Series(80.0, index=grid.steps), block_prices(grid, 300.0))
    _assert_exclusive(case.result)
