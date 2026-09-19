"""Reserve commitments are one value per 4h block -- identical across every
step of that block -- including on DST days whose step count is not 96."""
import pytest

from dispatch_helpers import (
    BATTERY, NORMAL_DAY, SPRING_FORWARD_DAY, grid_for, random_ancillary_case,
)

TOL = 1e-6


@pytest.mark.parametrize(
    "day, n_steps",
    [(NORMAL_DAY, 96), (SPRING_FORWARD_DAY, 92)],
)
def test_reserve_constant_within_block_and_grid_size_matches_dst(day, n_steps):
    case = random_ancillary_case(day, seed=4)
    grid, r = case.grid, case.result
    assert len(grid.steps) == n_steps
    assert len(r.e) == n_steps

    # One variable per block, so the per-step expansion is constant by construction.
    assert len(r.c_fcr) == len(r.c_pos) == len(r.c_neg) == len(grid.blocks)
    steps_by_block: dict[int, list[int]] = {}
    for i, b in enumerate(grid.step_block_index):
        steps_by_block.setdefault(b, []).append(i)
    assert set(steps_by_block) == {b.index for b in grid.blocks}  # every block has steps, none orphaned
    assert sum(len(v) for v in steps_by_block.values()) == n_steps

    # And the model really uses that one value at every step: shared power
    # headroom (C3) evaluated per step against the block's commitment.
    for i, b in enumerate(grid.step_block_index):
        fcr, pos, neg = r.c_fcr.loc[b], r.c_pos.loc[b], r.c_neg.loc[b]
        assert r.p_ch.iloc[i] + fcr + neg <= BATTERY.power_mw + TOL
        assert r.p_dis.iloc[i] + fcr + pos <= BATTERY.power_mw + TOL


def test_reserves_are_committed_on_dst_day():
    case = random_ancillary_case(SPRING_FORWARD_DAY, seed=5)
    r = case.result
    assert (r.c_fcr.sum() + r.c_pos.sum() + r.c_neg.sum()) > 0.1
