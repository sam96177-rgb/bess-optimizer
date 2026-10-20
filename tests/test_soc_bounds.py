"""SOC band regression tests: e >= 0, e >= soc_min and e <= soc_max on every
step, for day-ahead-only and co-optimized models, on normal and DST days, and
with soc_min_fraction = 0 (which the config accepts but nothing else exercised).
"""
from __future__ import annotations

import dataclasses

import pytest

from tests.dispatch_helpers import (
    BATTERY,
    FALL_BACK_DAY,
    NORMAL_DAY,
    SPRING_FORWARD_DAY,
    grid_for,
    random_ancillary_case,
    random_day_ahead,
    solve_case,
)

TOL = 1e-6
DAYS = [NORMAL_DAY, SPRING_FORWARD_DAY, FALL_BACK_DAY]
BATTERIES = {
    "default": BATTERY,
    "soc_min_zero": dataclasses.replace(BATTERY, soc_min_fraction=0.0),
}


def _assert_in_band(e, battery):
    assert (e >= -TOL).all(), f"SOC went negative: min {e.min()}"
    assert (e >= battery.soc_min_mwh - TOL).all(), f"SOC below soc_min: min {e.min()}"
    assert (e <= battery.soc_max_mwh + TOL).all(), f"SOC above soc_max: max {e.max()}"


@pytest.mark.parametrize("battery_name", BATTERIES)
@pytest.mark.parametrize("day", DAYS)
def test_day_ahead_only_soc_stays_in_band(day, battery_name):
    battery = BATTERIES[battery_name]
    grid = grid_for(day)
    case = solve_case(
        grid, random_day_ahead(grid, seed=3, low=-300.0, high=900.0), with_ancillary=False, battery=battery
    )
    _assert_in_band(case.result.e, battery)


@pytest.mark.parametrize("battery_name", BATTERIES)
@pytest.mark.parametrize("day", DAYS)
def test_cooptimized_soc_stays_in_band(day, battery_name):
    battery = BATTERIES[battery_name]
    case = random_ancillary_case(day=day, seed=4, battery=battery)
    _assert_in_band(case.result.e, battery)


def test_soc_min_zero_battery_actually_uses_the_extra_band():
    """Guard against the zero-floor case being a no-op: with a huge spread the
    battery should dip below the default 1 MWh floor once it is allowed to."""
    grid = grid_for(NORMAL_DAY)
    prices = random_day_ahead(grid, seed=3, low=-300.0, high=900.0)
    default = solve_case(grid, prices, with_ancillary=False, battery=BATTERIES["default"])
    zero = solve_case(grid, prices, with_ancillary=False, battery=BATTERIES["soc_min_zero"])
    assert zero.result.e.min() < BATTERY.soc_min_mwh - 1e-3
    assert zero.result.objective_value >= default.result.objective_value - TOL
