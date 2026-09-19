"""Reserve backing (C2): every step of a block keeps enough SOC room to
deliver the committed FCR and aFRR energy -- and reserves are actually sold
(guards against the spec's binary-gated form that forces FCR to zero)."""
import pandas as pd
import pytest

from dispatch_helpers import (
    ASSUMPTIONS, BATTERY, NORMAL_DAY, PRICE_BASIS, SPRING_FORWARD_DAY, block_prices, grid_for,
    random_ancillary_case, solve_case,
)
from src.valuation.revenue import compute_revenue

TOL = 1e-5


def _assert_backed(case) -> None:
    r = case.result
    for step, b in zip(case.grid.steps, case.grid.step_block_index):
        fcr, pos, neg = r.c_fcr.loc[b], r.c_pos.loc[b], r.c_neg.loc[b]
        low = BATTERY.soc_min_mwh + fcr * ASSUMPTIONS.fcr_reserve_hours + pos * ASSUMPTIONS.afrr_reserve_hours
        high = BATTERY.soc_max_mwh - (fcr * ASSUMPTIONS.fcr_reserve_hours + neg * ASSUMPTIONS.afrr_reserve_hours)
        assert r.e.loc[step] >= low - TOL, f"{step}: SOC {r.e.loc[step]:.4f} below backing floor {low:.4f}"
        assert r.e.loc[step] <= high + TOL, f"{step}: SOC {r.e.loc[step]:.4f} above backing ceiling {high:.4f}"


@pytest.mark.parametrize("day", [NORMAL_DAY, SPRING_FORWARD_DAY])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_soc_leaves_room_for_committed_reserves(day, seed):
    _assert_backed(random_ancillary_case(day, seed=seed))


def test_afrr_up_is_energy_backed_not_just_fcr():
    """aFRR-up alone, attractive price, flat DA: SOC must stay >= soc_min +
    c_pos * afrr_hours in every step of every block where it is sold."""
    grid = grid_for(NORMAL_DAY)
    case = solve_case(grid, pd.Series(50.0, index=grid.steps), block_prices(grid, 0.0), block_prices(grid, 40.0))
    assert case.result.c_pos.min() > 0.5  # incentive bit in every block
    _assert_backed(case)
    for step, b in zip(grid.steps, grid.step_block_index):
        floor = BATTERY.soc_min_mwh + case.result.c_pos.loc[b] * ASSUMPTIONS.afrr_reserve_hours
        assert case.result.e.loc[step] >= floor - TOL


def test_fcr_is_actually_sold_known_answer():
    """Flat DA (no arbitrage) and a high FCR price: the optimum is to commit
    the full FCR cap (0.8 * 4.5 = 3.6 MW) in every block. The spec's gated
    form (u_dis + u_ch <= 1 with reserves inside the gated constraints)
    would force this to 0."""
    grid = grid_for(NORMAL_DAY)
    case = solve_case(grid, pd.Series(50.0, index=grid.steps), block_prices(grid, 100.0))
    cap = 0.8 * BATTERY.power_mw
    assert (case.result.c_fcr - cap).abs().max() < 1e-4
    revenue = compute_revenue(
        case.result, case.day_ahead, grid.delta_t_hours, case.ancillary, battery=BATTERY,
    )
    assert revenue.fcr == pytest.approx(cap * 100.0 * len(grid.blocks), rel=1e-6)
    _assert_backed(case)


def test_fcr_sold_in_blocks_that_also_dispatch():
    """FCR headroom must coexist with energy dispatch (no per-step gating):
    with a moderate DA spread and a strong FCR price, FCR is committed in
    every block, including blocks that charge or discharge."""
    grid = grid_for(NORMAL_DAY)
    prices = pd.Series(50.0, index=grid.steps)
    prices.iloc[8:16] = 20.0
    prices.iloc[50:58] = 80.0
    case = solve_case(grid, prices, block_prices(grid, 100.0))
    r = case.result
    assert r.c_fcr.min() > 0.1
    dispatching_blocks = {
        b for i, b in enumerate(grid.step_block_index) if r.p_ch.iloc[i] > 1e-3 or r.p_dis.iloc[i] > 1e-3
    }
    assert dispatching_blocks, "scenario should still dispatch some energy"
    assert all(r.c_fcr.loc[b] > 0.1 for b in dispatching_blocks)
