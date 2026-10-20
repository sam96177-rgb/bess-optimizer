"""Reserve power rules and the hard p_ch / p_dis bounds.

V8-2: there is no total cap on fcr + pos + neg (the German TSOs' rules only need each
direction to keep its own power band), and FCR needs 1.25 MW of battery power per MW sold
(prequalification conditions v1.05, eq. 3.9). Prices here are synthetic test inputs on the
real day grid.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.data.grid import BlockBoundary
from src.model.build import AncillaryInputs, build_model
from src.model.solve import solve_model
from tests.dispatch_helpers import (
    ASSUMPTIONS,
    BATTERY,
    FALL_BACK_DAY,
    NORMAL_DAY,
    PRICE_BASIS,
    SOLVER,
    SPRING_FORWARD_DAY,
    block_prices,
    grid_for,
    random_day_ahead,
    solve_case,
)

TOL = 1e-6
FCR_POWER_FACTOR = 1.25  # written out here on purpose: the test must not follow the model's constant


@pytest.mark.parametrize("day", [NORMAL_DAY, SPRING_FORWARD_DAY, FALL_BACK_DAY])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_each_direction_keeps_its_own_power_band_with_fcr_at_1_25(day, seed):
    """Moderate day-ahead plus well-paid FCR and aFRR: trading power + 1.25 x FCR + aFRR must
    fit the converter rating in each direction, at every step. Without the 1.25 factor the
    optimizer packs FCR up to the rating and this fails."""
    grid = grid_for(day)
    case = solve_case(
        grid,
        random_day_ahead(grid, seed=seed, low=0.0, high=150.0),
        block_prices(grid, 200.0),
        block_prices(grid, 10.0),
        block_prices(grid, 10.0),
    )
    r = case.result
    block = pd.Series(grid.step_block_index, index=grid.steps)
    fcr, pos, neg = (s.reindex(block.values).set_axis(grid.steps) for s in (r.c_fcr, r.c_pos, r.c_neg))
    assert (r.p_dis + FCR_POWER_FACTOR * fcr + pos <= BATTERY.power_mw + TOL).all()
    assert (r.p_ch + FCR_POWER_FACTOR * fcr + neg <= BATTERY.power_mw + TOL).all()
    assert r.c_fcr.max() > 0


def test_aFRR_up_and_down_are_not_capped_together():
    """Known answer. FCR is worth nothing, aFRR+ and aFRR- are worth a lot. The old total cap
    held them to 5 MW together; now each takes its own band, floor(4.5) = 4 MW at the 1 MW
    bid granularity (the SOC headroom allows 4 MWh each way)."""
    grid = grid_for(NORMAL_DAY)
    case = solve_case(
        grid,
        pd.Series(0.0, index=grid.steps),
        block_prices(grid, 0.0),
        block_prices(grid, 500.0),
        block_prices(grid, 500.0),
    )
    granularity = PRICE_BASIS.granularity_for("afrr_pos")
    each = math.floor(BATTERY.power_mw / granularity) * granularity if granularity > 0 else BATTERY.power_mw
    assert (case.result.c_pos - each).abs().max() < 1e-4
    assert (case.result.c_neg - each).abs().max() < 1e-4
    assert (case.result.c_pos + case.result.c_neg).min() > BATTERY.power_nominal_mw


def test_block_with_no_steps_is_still_capped_per_direction():
    """A block with no steps in the day's grid skips C2/C3 but still earns capacity revenue,
    so only the variable's own upper bound (the converter rating) limits its c_pos / c_neg."""
    grid = grid_for(NORMAL_DAY)
    last = grid.blocks[-1]
    ghost = BlockBoundary(index=last.index + 1, start=last.end, end=last.end + (last.end - last.start))
    blocks = [*grid.blocks, ghost]
    assert ghost.index not in grid.step_block_index

    n = len(blocks)
    ancillary = AncillaryInputs(
        blocks=blocks,
        step_block_index=grid.step_block_index,
        fcr_prices=pd.Series(0.0, index=range(n)),
        afrr_pos_prices=pd.Series(500.0, index=range(n)),
        afrr_neg_prices=pd.Series(500.0, index=range(n)),
        price_basis=PRICE_BASIS,
        assumptions=ASSUMPTIONS,
    )
    model = build_model(
        steps=grid.steps,
        day_ahead_prices=pd.Series(0.0, index=grid.steps),
        battery=BATTERY,
        delta_t_hours=grid.delta_t_hours,
        ancillary=ancillary,
    )
    result = solve_model(model, SOLVER)
    assert result.c_pos[ghost.index] <= BATTERY.power_mw + TOL
    assert result.c_neg[ghost.index] <= BATTERY.power_mw + TOL
    assert result.c_pos[ghost.index] > 0


@pytest.mark.parametrize("day", [NORMAL_DAY, SPRING_FORWARD_DAY, FALL_BACK_DAY])
def test_dispatch_power_is_bounded_without_exclusivity_or_ancillary(day):
    """With C5 and C3 both off, only the SOC band used to cap power (~30 MW on
    a 4.5 MW battery). The variable upper bounds now do."""
    grid = grid_for(day)
    spiky = random_day_ahead(grid, seed=7, low=-300.0, high=900.0)
    case = solve_case(grid, spiky, with_ancillary=False, enforce_exclusivity=False)
    limit = BATTERY.power_mw
    assert case.result.p_ch.max() <= limit + TOL
    assert case.result.p_dis.max() <= limit + TOL
    assert case.result.p_ch.max() == pytest.approx(limit, abs=1e-4) or case.result.p_dis.max() == pytest.approx(
        limit, abs=1e-4
    )  # the spikes make it worth running flat out
