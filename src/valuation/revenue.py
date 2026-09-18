"""Recomputes revenue independently of the solver's own objective value —
a deliberately separate implementation (SPEC.md Milestone 3) so a bug in
build.py's objective wouldn't be self-confirming.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.model.build import AncillaryInputs, nominal_block_hours
from src.model.solve import SolveResult


@dataclass(frozen=True)
class RevenueBreakdown:
    day_ahead: float
    fcr: float
    afrr_pos: float
    afrr_neg: float
    activation: float

    @property
    def total(self) -> float:
        return self.day_ahead + self.fcr + self.afrr_pos + self.afrr_neg + self.activation


def compute_revenue(
    result: SolveResult,
    day_ahead_prices: pd.Series,
    delta_t_hours: float,
    ancillary: AncillaryInputs | None = None,
) -> RevenueBreakdown:
    prices = day_ahead_prices.reindex(result.p_dis.index)
    day_ahead = float((prices * (result.p_dis - result.p_ch) * delta_t_hours).sum())

    if ancillary is None:
        return RevenueBreakdown(day_ahead=day_ahead, fcr=0.0, afrr_pos=0.0, afrr_neg=0.0, activation=0.0)

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

    return RevenueBreakdown(day_ahead=day_ahead, fcr=fcr, afrr_pos=afrr_pos, afrr_neg=afrr_neg, activation=activation)


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
