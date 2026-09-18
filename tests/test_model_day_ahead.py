import math

import numpy as np
import pandas as pd
import pytest

from src.config import load_battery_config, load_solver_config
from src.model.battery import battery_from_config
from src.model.build import build_model
from src.model.solve import solve_model
from src.valuation.revenue import compute_revenue

TOL = 1e-4

BATTERY = battery_from_config(load_battery_config())
SOLVER = load_solver_config()
DELTA_T = 0.25

STEPS = pd.date_range("2026-06-01", periods=96, freq="15min", tz="Europe/Berlin")


def _flat_prices(value: float) -> pd.Series:
    return pd.Series(value, index=STEPS)


def _random_prices(seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(-20, 200, size=len(STEPS)), index=STEPS)


def _solve(prices: pd.Series, **kwargs):
    model = build_model(
        steps=STEPS,
        day_ahead_prices=prices,
        battery=BATTERY,
        delta_t_hours=DELTA_T,
        **kwargs,
    )
    return solve_model(model, SOLVER)


# ---- physical sanity checks, over random prices ----------------------------

def test_soc_never_leaves_bounds():
    result = _solve(_random_prices())
    assert (result.e >= BATTERY.soc_min_mwh - TOL).all()
    assert (result.e <= BATTERY.soc_max_mwh + TOL).all()


def test_power_never_exceeds_pmax():
    result = _solve(_random_prices())
    assert (result.p_ch <= BATTERY.power_mw + TOL).all()
    assert (result.p_dis <= BATTERY.power_mw + TOL).all()
    assert (result.p_ch >= -TOL).all()
    assert (result.p_dis >= -TOL).all()


def test_never_charges_and_discharges_same_step():
    result = _solve(_random_prices())
    both_active = (result.p_ch > 1e-6) & (result.p_dis > 1e-6)
    assert not both_active.any()


def test_energy_reconciles_with_round_trip_efficiency():
    result = _solve(_random_prices())
    # Rebuild the e trajectory from p_ch/p_dis independently of the solver
    # and check it matches the solver's own e series (cyclic wraparound).
    prev = result.e.iloc[-1]
    for t in range(len(STEPS)):
        expected = (
            prev
            + BATTERY.eta * DELTA_T * result.p_ch.iloc[t]
            - (DELTA_T / BATTERY.eta) * result.p_dis.iloc[t]
        )
        assert expected == pytest.approx(result.e.iloc[t], abs=1e-6)
        prev = result.e.iloc[t]


def test_revenue_recomputed_independently_matches_solver_objective():
    prices = _random_prices()
    model = build_model(steps=STEPS, day_ahead_prices=prices, battery=BATTERY, delta_t_hours=DELTA_T)
    result = solve_model(model, SOLVER)
    revenue = compute_revenue(result, prices, DELTA_T)
    assert revenue.total == pytest.approx(result.objective_value, rel=1e-6)


# ---- known-answer test -------------------------------------------------

def test_known_answer_two_price_day_full_cycle():
    """10 steps at 0 EUR/MWh (cheap window, plenty of time to reach
    soc_max), 10 steps at 1000 EUR/MWh (expensive window, plenty of time
    to reach soc_min), everything else at a flat 50 EUR/MWh that offers
    no profitable trade once the extreme windows have already used the
    battery's full cyclic swing. The obviously-optimal schedule: charge
    fully in the cheap window, discharge fully in the expensive window,
    do nothing else.
    """
    prices = _flat_prices(50.0)
    cheap_window = STEPS[10:20]
    expensive_window = STEPS[50:60]
    prices.loc[cheap_window] = 0.0
    prices.loc[expensive_window] = 1000.0

    result = _solve(prices)

    delta_e_max = BATTERY.soc_max_mwh - BATTERY.soc_min_mwh
    expected_objective = 1000.0 * delta_e_max * BATTERY.eta
    assert result.objective_value == pytest.approx(expected_objective, rel=1e-3)

    assert result.e.max() == pytest.approx(BATTERY.soc_max_mwh, abs=1e-3)
    assert result.e.min() == pytest.approx(BATTERY.soc_min_mwh, abs=1e-3)

    other_steps = STEPS.difference(cheap_window).difference(expensive_window)
    assert (result.p_ch.loc[other_steps].abs() < 1e-6).all()
    assert (result.p_dis.loc[other_steps].abs() < 1e-6).all()
    assert result.p_ch.loc[expensive_window].abs().sum() < 1e-6
    assert result.p_dis.loc[cheap_window].abs().sum() < 1e-6


# ---- config-flag behavior ------------------------------------------------

def test_cyclic_soc_false_requires_initial_soc():
    with pytest.raises(ValueError, match="initial_soc_mwh is required"):
        build_model(
            steps=STEPS, day_ahead_prices=_flat_prices(10.0), battery=BATTERY,
            delta_t_hours=DELTA_T, cyclic_soc=False,
        )


def test_cyclic_soc_true_rejects_initial_soc():
    with pytest.raises(ValueError, match="must not be set"):
        build_model(
            steps=STEPS, day_ahead_prices=_flat_prices(10.0), battery=BATTERY,
            delta_t_hours=DELTA_T, cyclic_soc=True, initial_soc_mwh=1.0,
        )


def test_non_cyclic_soc_starts_from_given_initial_soc():
    result = _solve(_flat_prices(50.0), cyclic_soc=False, initial_soc_mwh=BATTERY.soc_min_mwh)
    expected_e0 = (
        BATTERY.soc_min_mwh
        + BATTERY.eta * DELTA_T * result.p_ch.iloc[0]
        - (DELTA_T / BATTERY.eta) * result.p_dis.iloc[0]
    )
    assert result.e.iloc[0] == pytest.approx(expected_e0, abs=1e-6)
