"""Degradation cost and terminal credit, with hand-computed expectations."""
import dataclasses

import pandas as pd
import pytest

from tests.dispatch_helpers import BATTERY, NORMAL_DAY, grid_for, solve_case
from src.model.build import terminal_value_eur_mwh
from src.valuation.revenue import compute_revenue


def _two_level_prices(grid, low: float, high: float) -> pd.Series:
    prices = pd.Series(high, index=grid.steps)
    prices.iloc[:48] = low
    return prices


def test_flat_price_day_only_charges_for_aux_no_discharge():
    """Flat price: no arbitrage. With a cost per MWh of throughput the only
    energy moved is the minimum to cover aux drain over the day."""
    grid = grid_for(NORMAL_DAY)
    case = solve_case(grid, pd.Series(50.0, index=grid.steps), with_ancillary=False)
    r = case.result
    assert r.p_dis.sum() == pytest.approx(0.0, abs=1e-6)
    charged_mwh = r.p_ch.sum() * grid.delta_t_hours
    assert charged_mwh == pytest.approx(BATTERY.aux_load_mw * 24.0 / BATTERY.eta, abs=1e-4)


def test_degradation_larger_than_spread_means_no_cycling():
    """Spread 10 EUR/MWh; the round trip alone loses 10% of ~50 and each
    MWh through the battery costs 8 on each side -- cycling loses money."""
    grid = grid_for(NORMAL_DAY)
    case = solve_case(grid, _two_level_prices(grid, 45.0, 55.0), with_ancillary=False)
    assert case.result.p_dis.sum() == pytest.approx(0.0, abs=1e-6)


def test_same_spread_cycles_when_degradation_is_zero():
    """Contrast for the test above: 20 EUR/MWh spread, no degradation ->
    the optimizer cycles. Confirms the previous test is degradation-driven."""
    grid = grid_for(NORMAL_DAY)
    free = dataclasses.replace(BATTERY, degradation_cost_eur_mwh=0.0)
    case = solve_case(grid, _two_level_prices(grid, 40.0, 60.0), with_ancillary=False, battery=free)
    assert case.result.p_dis.sum() * grid.delta_t_hours > 1.0


def test_degradation_reported_equals_rate_times_throughput():
    grid = grid_for(NORMAL_DAY)
    free = dataclasses.replace(BATTERY, degradation_cost_eur_mwh=0.0)
    case = solve_case(grid, _two_level_prices(grid, 40.0, 60.0), with_ancillary=False, battery=free)
    revenue = compute_revenue(case.result, case.day_ahead, grid.delta_t_hours, battery=BATTERY)
    throughput = (case.result.p_ch.sum() + case.result.p_dis.sum()) * grid.delta_t_hours
    assert revenue.degradation == pytest.approx(8.0 * throughput)
    assert revenue.net == pytest.approx(revenue.gross - revenue.degradation)


# ---- terminal credit ------------------------------------------------------

def _spike_prices(grid) -> pd.Series:
    prices = pd.Series(20.0, index=grid.steps)
    prices.iloc[40:46] = 1000.0  # mid-day spike lifts the mean price well above 20
    return prices


def test_terminal_credit_zero_under_cyclic_soc():
    grid = grid_for(NORMAL_DAY)
    assert terminal_value_eur_mwh(BATTERY, _spike_prices(grid), cyclic_soc=True) == 0.0


def test_terminal_credit_is_mean_decision_price_when_not_cyclic():
    grid = grid_for(NORMAL_DAY)
    prices = _spike_prices(grid)
    assert terminal_value_eur_mwh(BATTERY, prices, cyclic_soc=False) == pytest.approx(prices.mean())


def test_terminal_credit_makes_battery_refill_after_last_sale():
    """After the spike the battery has no further sale, so without a credit
    it ends near soc_min; with credit lambda = mean (~81) it refills at 20."""
    grid = grid_for(NORMAL_DAY)
    ideal = dataclasses.replace(BATTERY, aux_load_mw=0.0, degradation_cost_eur_mwh=0.0)
    no_credit = dataclasses.replace(ideal, terminal_value="none")

    kwargs = dict(with_ancillary=False, cyclic_soc=False, initial_soc_mwh=ideal.soc_min_mwh)
    with_credit = solve_case(grid, _spike_prices(grid), battery=ideal, **kwargs)
    without = solve_case(grid, _spike_prices(grid), battery=no_credit, **kwargs)

    assert with_credit.result.e.iloc[-1] == pytest.approx(ideal.soc_max_mwh, abs=1e-3)
    assert without.result.e.iloc[-1] == pytest.approx(ideal.soc_min_mwh, abs=1e-3)


def test_net_plus_terminal_credit_equals_objective_non_cyclic():
    grid = grid_for(NORMAL_DAY)
    prices = _spike_prices(grid)
    kwargs = dict(with_ancillary=False, cyclic_soc=False, initial_soc_mwh=BATTERY.soc_min_mwh)
    case = solve_case(grid, prices, **kwargs)
    lam = terminal_value_eur_mwh(BATTERY, prices, cyclic_soc=False)
    revenue = compute_revenue(
        case.result, prices, grid.delta_t_hours, battery=BATTERY, terminal_value_eur_mwh=lam,
    )
    assert revenue.terminal_credit == pytest.approx(lam * case.result.e.iloc[-1])
    assert revenue.net + revenue.terminal_credit == pytest.approx(case.result.objective_value, rel=1e-6)
    # The credit is not revenue.
    assert revenue.net == pytest.approx(revenue.gross - revenue.degradation)
