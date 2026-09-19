"""SOC bookkeeping: at every step e[t] - e[t-1] must equal exactly what the
physics says (charge with eta, discharge with 1/eta, aux drain, and the
expected aFRR activation energy pushed through the same conversion losses).
"""
import pytest

from dispatch_helpers import (
    ASSUMPTIONS, BATTERY, NORMAL_DAY, SPRING_FORWARD_DAY, random_ancillary_case,
)

TOL = 1e-5


def _expected_delta(case, t: int) -> float:
    r = case.result
    dt_h = case.grid.delta_t_hours
    eta = BATTERY.eta
    delta = eta * r.p_ch.iloc[t] * dt_h - r.p_dis.iloc[t] * dt_h / eta - BATTERY.aux_load_mw * dt_h
    if case.ancillary is not None and case.ancillary.include_activation_in_soc:
        block = case.grid.step_block_index[t]
        delta -= (
            ASSUMPTIONS.alpha_pos / eta * r.c_pos.loc[block] - ASSUMPTIONS.alpha_neg * eta * r.c_neg.loc[block]
        ) * dt_h
    return delta


def _assert_conserved(case, initial_soc: float | None) -> None:
    e = case.result.e
    prev = e.iloc[-1] if initial_soc is None else initial_soc  # cyclic wrap vs given start
    for t in range(len(e)):
        assert e.iloc[t] - prev == pytest.approx(_expected_delta(case, t), abs=TOL), f"step {t}"
        prev = e.iloc[t]


@pytest.mark.parametrize("day", [NORMAL_DAY, SPRING_FORWARD_DAY])
def test_energy_conserved_every_step_cyclic(day):
    case = random_ancillary_case(day, seed=1)
    # Non-trivial: aFRR is actually sold, so the activation term is exercised.
    assert case.result.c_pos.sum() > 0.1 and case.result.c_neg.sum() > 0.1
    _assert_conserved(case, initial_soc=None)


def test_energy_conserved_every_step_non_cyclic_with_initial_soc():
    initial = 5.0
    case = random_ancillary_case(NORMAL_DAY, seed=2, cyclic_soc=False, initial_soc_mwh=initial)
    _assert_conserved(case, initial_soc=initial)


def test_energy_conserved_without_activation_in_soc():
    case = random_ancillary_case(NORMAL_DAY, seed=3, include_activation_in_soc=False)
    _assert_conserved(case, initial_soc=None)


def test_aux_load_alone_drains_soc_when_idle():
    """Hand check: a day so flat and cheap-to-hold that the battery only
    refills what aux drains -- total energy in (after eta) equals aux * 24h."""
    from dispatch_helpers import grid_for, solve_case
    import pandas as pd

    grid = grid_for(NORMAL_DAY)
    case = solve_case(grid, pd.Series(50.0, index=grid.steps), with_ancillary=False)
    stored_in = BATTERY.eta * case.result.p_ch.sum() * grid.delta_t_hours
    stored_out = case.result.p_dis.sum() * grid.delta_t_hours / BATTERY.eta
    assert stored_in - stored_out == pytest.approx(BATTERY.aux_load_mw * 24.0, abs=1e-4)
