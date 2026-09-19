"""Recomputes revenue independently of the solver's own objective value —
a deliberately separate implementation (SPEC.md Milestone 3) so a bug in
build.py's objective wouldn't be self-confirming.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.model.battery import Battery
from src.model.build import AncillaryInputs, nominal_block_hours
from src.model.solve import SolveResult


@dataclass(frozen=True)
class RevenueBreakdown:
    day_ahead: float
    fcr: float
    afrr_pos: float
    afrr_neg: float
    activation: float
    degradation: float = 0.0  # a cost, stored as a positive number
    terminal_credit: float = 0.0  # decision-side value of leftover energy; NOT revenue

    @property
    def gross(self) -> float:
        return self.day_ahead + self.fcr + self.afrr_pos + self.afrr_neg + self.activation

    @property
    def net(self) -> float:
        return self.gross - self.degradation

    @property
    def objective_equivalent(self) -> float:
        """What the solver's objective should equal when settlement prices
        equal decision prices."""
        return self.net + self.terminal_credit


def compute_revenue(
    result: SolveResult,
    day_ahead_prices: pd.Series,
    delta_t_hours: float,
    ancillary: AncillaryInputs | None = None,
    *,
    battery: Battery,
    terminal_value_eur_mwh: float = 0.0,
) -> RevenueBreakdown:
    """`terminal_value_eur_mwh` is the decision-side lambda from
    build.terminal_value_eur_mwh; the resulting credit is reported
    separately and never counted in gross/net."""
    prices = day_ahead_prices.reindex(result.p_dis.index)
    day_ahead = float((prices * (result.p_dis - result.p_ch) * delta_t_hours).sum())
    degradation = float(battery.degradation_cost_eur_mwh * (result.p_ch + result.p_dis).sum() * delta_t_hours)
    terminal_credit = float(terminal_value_eur_mwh * result.e.iloc[-1])

    if ancillary is None:
        return RevenueBreakdown(
            day_ahead=day_ahead, fcr=0.0, afrr_pos=0.0, afrr_neg=0.0, activation=0.0,
            degradation=degradation, terminal_credit=terminal_credit,
        )

    fcr = _capacity_revenue(result.c_fcr, ancillary.fcr_prices, "fcr", ancillary.price_basis, ancillary.blocks)
    afrr_pos = _capacity_revenue(
        result.c_pos, ancillary.afrr_pos_prices, "afrr_pos", ancillary.price_basis, ancillary.blocks
    )
    afrr_neg = _capacity_revenue(
        result.c_neg, ancillary.afrr_neg_prices, "afrr_neg", ancillary.price_basis, ancillary.blocks
    )

    activation = 0.0
    for block in ancillary.blocks:
        activation += (
            ancillary.assumptions.alpha_pos * result.c_pos.loc[block.index]
            + ancillary.assumptions.alpha_neg * result.c_neg.loc[block.index]
        ) * nominal_block_hours(block) * ancillary.assumptions.activation_energy_price_eur_mwh

    return RevenueBreakdown(
        day_ahead=day_ahead, fcr=fcr, afrr_pos=afrr_pos, afrr_neg=afrr_neg, activation=activation,
        degradation=degradation, terminal_credit=terminal_credit,
    )


def _capacity_revenue(
    capacity_solution: pd.Series,
    prices: pd.Series,
    market: str,
    price_basis,
    blocks,
) -> float:
    basis = price_basis.basis_for(market)
    total = 0.0
    for block in blocks:
        price = prices.loc[block.index]
        value = capacity_solution.loc[block.index]
        if basis == "per_block_total":
            total += price * value
        elif basis == "per_mw_hour":
            total += price * value * nominal_block_hours(block)
        else:
            raise ValueError(f"{market}: unsupported price_basis '{basis}'")
    return float(total)
