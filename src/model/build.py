"""linopy model construction for the single-day BESS dispatch problem.

One function, `build_model`, serves both SPEC.md Milestone 3 (day-ahead
only: C1, C5, C6) and Milestone 4 (full co-optimization: adds C2, C3, C4,
C7) — passing `ancillary=None` gives the Milestone 3 model; passing an
`AncillaryInputs` gives the Milestone 4 model. Splitting these into two
near-duplicate builders would just duplicate C1/C5/C6; the flag keeps one
implementation instead of two constraint sets to keep in sync.

A note on C6 (terminal SOC) and the e[t] indexing convention: SPEC.md
defines `e[t]` as "state of charge at end of step t" and C6 as
`e[last step] == e[first step]`. Taken fully literally that would tie
e[0] (end of the FIRST step, i.e. after step 0's own dispatch already
happened) to e[T-1], which forces step 0's net energy change to exactly
zero -- clearly not the intended behavior, and not what "prevents the
optimizer from manufacturing revenue by draining the battery and never
refilling it" describes. The standard, physically-meaningful reading
(and the one implemented here) is that the day's SOC *before* step 0 has
happened equals e[T-1]: i.e. C1's dynamics for step 0 use e[T-1] as the
"previous" state whenever cyclic_soc=True, making the day a closed loop.
When cyclic_soc=False, step 0's previous state is instead the caller-
supplied `initial_soc_mwh` (e.g. the prior day's actual ending SOC, for
Milestone 5's rolling multi-day chaining).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import linopy
import numpy as np
import pandas as pd

from src.config import AssumptionsConfig, MarketPriceBasisConfig
from src.data.grid import BlockBoundary
from src.model.battery import Battery


@dataclass(frozen=True)
class AncillaryInputs:
    blocks: list[BlockBoundary]
    step_block_index: list[int]
    fcr_prices: pd.Series  # indexed by block index
    afrr_pos_prices: pd.Series
    afrr_neg_prices: pd.Series
    price_basis: MarketPriceBasisConfig
    assumptions: AssumptionsConfig
    include_activation_in_soc: bool = True

    def __post_init__(self) -> None:
        n_blocks = len(self.blocks)
        for name, series in (
            ("fcr_prices", self.fcr_prices),
            ("afrr_pos_prices", self.afrr_pos_prices),
            ("afrr_neg_prices", self.afrr_neg_prices),
        ):
            if len(series) != n_blocks:
                raise ValueError(
                    f"{name} has {len(series)} entries but there are {n_blocks} blocks"
                )
            if series.isna().any():
                raise ValueError(f"{name} contains NaN — a block with a missing price must be excluded upstream, not fed in")


@dataclass(frozen=True)
class FixedReserves:
    """Reserve commitments already made at an earlier gate. Any field left
    None stays a free decision variable. Each Series is indexed by block."""

    fcr: pd.Series | None = None
    pos: pd.Series | None = None
    neg: pd.Series | None = None


FCR_CAP_FRACTION_OF_PMAX = 0.8  # C4
# Battery power needed per MW of FCR sold: at least 1.25 x (German TSOs' FCR prequalification
# conditions v1.05, 5 July 2024, eq. 3.9).
FCR_POWER_FACTOR = 1.25
RESERVE_FIX_RESOLUTION = 1e-6  # committed MW are floored to this grid so a fixed value never sits above its own constraints


def _constrain_to_granularity(
    m: linopy.Model, var: linopy.Variable, name: str, block_ids: list[int], upper_bound_mw: float, granularity_mw: float,
) -> None:
    """1 MW minimum reserve bid, 1 MW increment (regelleistung.net tendering
    page; PLAN_V4_REVISION.md section 4): `var` must be an integer multiple
    of granularity_mw. linopy has no native "multiple of g" bound, only
    integer/binary, so this adds an auxiliary integer count variable and
    ties `var` to it -- 0 stays unconstrained (v3's fractional-MW behaviour).
    """
    if granularity_mw <= 0:
        return
    n_upper = math.floor(upper_bound_mw / granularity_mw)
    count = m.add_variables(lower=0, upper=n_upper, integer=True, coords=[block_ids], dims=["block"], name=f"n_{name}")
    m.add_constraints(var == granularity_mw * count, name=f"granularity_{name}")


def build_model(
    steps: pd.DatetimeIndex,
    day_ahead_prices: pd.Series,
    battery: Battery,
    delta_t_hours: float,
    cyclic_soc: bool = True,
    enforce_exclusivity: bool = True,
    initial_soc_mwh: float | None = None,
    ancillary: AncillaryInputs | None = None,
    fixed_reserves: FixedReserves | None = None,
) -> linopy.Model:
    if fixed_reserves is not None and ancillary is None:
        raise ValueError("fixed_reserves requires ancillary inputs")
    if day_ahead_prices.isna().any():
        raise ValueError("day_ahead_prices contains NaN — the caller must have already skipped this day")
    if not day_ahead_prices.index.equals(steps):
        raise ValueError("day_ahead_prices index must exactly match `steps`")
    if not cyclic_soc and initial_soc_mwh is None:
        raise ValueError("initial_soc_mwh is required when cyclic_soc=False")
    if cyclic_soc and initial_soc_mwh is not None:
        raise ValueError("initial_soc_mwh must not be set when cyclic_soc=True (it's derived from e[-1])")

    m = linopy.Model()
    n_steps = len(steps)

    # Hard bounds so power stays physical even with exclusivity (C5) and
    # ancillary (C3) both off; otherwise only the SOC band would cap it.
    p_ch = m.add_variables(lower=0, upper=battery.power_mw, coords=[steps], dims=["step"], name="p_ch")
    p_dis = m.add_variables(lower=0, upper=battery.power_mw, coords=[steps], dims=["step"], name="p_dis")
    u = m.add_variables(binary=True, coords=[steps], dims=["step"], name="u")
    e = m.add_variables(
        lower=battery.soc_min_mwh, upper=battery.soc_max_mwh, coords=[steps], dims=["step"], name="e"
    )

    c_fcr = c_pos = c_neg = None
    if ancillary is not None:
        n_blocks = len(ancillary.blocks)
        block_ids = [b.index for b in ancillary.blocks]
        fcr_cap_mw = FCR_CAP_FRACTION_OF_PMAX * battery.power_mw
        c_fcr = m.add_variables(lower=0, upper=fcr_cap_mw, coords=[block_ids], dims=["block"], name="c_fcr")
        # Upper bound = the converter rating per direction; also bounds a block with no steps
        # in the day's grid (DST), which skips C3 but still earns capacity revenue.
        c_pos = m.add_variables(lower=0, upper=battery.power_mw, coords=[block_ids], dims=["block"], name="c_pos")
        c_neg = m.add_variables(lower=0, upper=battery.power_mw, coords=[block_ids], dims=["block"], name="c_neg")
        # A fixed_reserves component already pins its variable to an exact
        # value (an earlier gate's own solve, sequential clearing) -- re-adding
        # the granularity equality on top would fight that fix whenever the
        # fixed value isn't itself on the grid, so it's skipped per market.
        fixed_component = {"fcr": None, "afrr_pos": None, "afrr_neg": None}
        if fixed_reserves is not None:
            fixed_component = {"fcr": fixed_reserves.fcr, "afrr_pos": fixed_reserves.pos, "afrr_neg": fixed_reserves.neg}
        for market, var, name, cap_mw in (
            ("fcr", c_fcr, "c_fcr", fcr_cap_mw),
            ("afrr_pos", c_pos, "c_pos", battery.power_mw),
            ("afrr_neg", c_neg, "c_neg", battery.power_mw),
        ):
            if fixed_component[market] is None:
                _constrain_to_granularity(m, var, name, block_ids, cap_mw, ancillary.price_basis.granularity_for(market))

    # ---- C5 — charge/discharge exclusivity ----
    if enforce_exclusivity:
        m.add_constraints(p_ch <= battery.power_mw * u, name="exclusivity_charge")
        m.add_constraints(p_dis <= battery.power_mw * (1 - u), name="exclusivity_discharge")

    # ---- C3 / C4 — shared power limits and FCR cap (Milestone 4 only) ----
    if ancillary is not None:
        for block in ancillary.blocks:
            step_mask = [bi == block.index for bi in ancillary.step_block_index]
            block_steps = steps[step_mask]
            fcr_b = c_fcr.sel(block=block.index)
            pos_b = c_pos.sel(block=block.index)
            neg_b = c_neg.sel(block=block.index)
            if len(block_steps) == 0:
                continue
            m.add_constraints(
                p_ch.sel(step=block_steps) + FCR_POWER_FACTOR * fcr_b + neg_b <= battery.power_mw,
                name=f"power_limit_charge_block{block.index}",
            )
            m.add_constraints(
                p_dis.sel(step=block_steps) + FCR_POWER_FACTOR * fcr_b + pos_b <= battery.power_mw,
                name=f"power_limit_discharge_block{block.index}",
            )
            # ---- C2 — SOC bounds with reserved headroom ----
            reserve_low = fcr_b * ancillary.assumptions.fcr_reserve_hours + pos_b * ancillary.assumptions.afrr_reserve_hours
            reserve_high = fcr_b * ancillary.assumptions.fcr_reserve_hours + neg_b * ancillary.assumptions.afrr_reserve_hours
            m.add_constraints(
                e.sel(step=block_steps) >= battery.soc_min_mwh + reserve_low,
                name=f"soc_reserve_low_block{block.index}",
            )
            m.add_constraints(
                e.sel(step=block_steps) <= battery.soc_max_mwh - reserve_high,
                name=f"soc_reserve_high_block{block.index}",
            )
    if fixed_reserves is not None:
        for var, committed in ((c_fcr, fixed_reserves.fcr), (c_pos, fixed_reserves.pos), (c_neg, fixed_reserves.neg)):
            if committed is None:
                continue
            for block in ancillary.blocks:
                value = math.floor(float(committed.loc[block.index]) / RESERVE_FIX_RESOLUTION) * RESERVE_FIX_RESOLUTION
                m.add_constraints(var.sel(block=block.index) == max(value, 0.0), name=f"fixed_{var.name}_block{block.index}")

    # C7 (block alignment) needs no explicit constraint: c_fcr/c_pos/c_neg
    # are single variables per block, structurally constant across all of
    # that block's steps.

    # ---- C1 — SOC dynamics ----
    # prev_e[t] = e[t-1]; step 0's previous state is e[-1] (cyclic_soc) or the
    # caller-supplied initial_soc_mwh (else). `.roll` wraps position 0 to the
    # last variable (cyclic); `.shift` leaves position 0 unassigned (linopy's
    # -1 "no variable" label), so adding a constant that's initial_soc_mwh at
    # position 0 and 0 elsewhere fills it in.
    if cyclic_soc:
        prev_e = e.roll(step=1, roll_coords=False)
    else:
        initial_soc_term = np.zeros(n_steps)
        initial_soc_term[0] = initial_soc_mwh
        prev_e = e.shift(step=1) + initial_soc_term

    if ancillary is not None and ancillary.include_activation_in_soc:
        alpha_pos = ancillary.assumptions.alpha_pos
        alpha_neg = ancillary.assumptions.alpha_neg
        # Gather each block's c_pos/c_neg onto every step in that block
        # (repeats, since a block spans several steps), then relabel the
        # "block" dim as "step" so it lines up with e/p_ch/p_dis positionally.
        pos_by_step = c_pos.sel(block=ancillary.step_block_index).rename(block="step").assign_coords(step=steps)
        neg_by_step = c_neg.sel(block=ancillary.step_block_index).rename(block="step").assign_coords(step=steps)
        # Activated aFRR energy passes through the conversion losses:
        # up-activation drains 1/eta per MWh delivered, down-activation
        # stores eta per MWh absorbed.
        activation = (alpha_pos / battery.eta * pos_by_step - alpha_neg * battery.eta * neg_by_step) * delta_t_hours
    else:
        activation = 0

    expr = (
        e
        - prev_e
        - battery.eta * delta_t_hours * p_ch
        + (delta_t_hours / battery.eta) * p_dis
        + battery.aux_load_mw * delta_t_hours
        + activation
    )
    m.add_constraints(expr == 0, name="soc_dynamics")

    # ---- Objective ----
    day_ahead_term = (day_ahead_prices.reindex(steps) * (p_dis - p_ch) * delta_t_hours).sum()
    degradation_term = battery.degradation_cost_eur_mwh * (p_ch + p_dis).sum() * delta_t_hours
    objective = day_ahead_term - degradation_term

    # Terminal credit: only meaningful when the day does not close its own
    # SOC loop. A credit (not a hard floor) so DST/extreme days stay feasible.
    terminal_lambda = terminal_value_eur_mwh(battery, day_ahead_prices, cyclic_soc)
    if terminal_lambda != 0.0:
        objective = objective + terminal_lambda * e.sel(step=steps[-1])

    if ancillary is not None:
        objective = objective + _capacity_revenue_term(
            c_fcr, ancillary.fcr_prices, "fcr", ancillary.price_basis, ancillary.blocks
        )
        objective = objective + _capacity_revenue_term(
            c_pos, ancillary.afrr_pos_prices, "afrr_pos", ancillary.price_basis, ancillary.blocks
        )
        objective = objective + _capacity_revenue_term(
            c_neg, ancillary.afrr_neg_prices, "afrr_neg", ancillary.price_basis, ancillary.blocks
        )
        activation_revenue = sum(
            (ancillary.assumptions.alpha_pos * c_pos.sel(block=b.index) + ancillary.assumptions.alpha_neg * c_neg.sel(block=b.index))
            * nominal_block_hours(b)
            * ancillary.assumptions.activation_energy_price_eur_mwh
            for b in ancillary.blocks
        )
        objective = objective + activation_revenue

    m.add_objective(objective, sense="max")
    return m


def terminal_value_eur_mwh(battery: Battery, decision_day_ahead_prices: pd.Series, cyclic_soc: bool) -> float:
    """EUR/MWh credit the optimizer assigns to energy left in the battery
    at the end of a non-cyclic day. Zero under cyclic SOC (the day already
    ends where it started) or when terminal_value is 'none'."""
    if cyclic_soc or battery.terminal_value == "none":
        return 0.0
    if battery.terminal_value == "mean_decision_da_price":
        return float(decision_day_ahead_prices.mean())
    raise ValueError(f"Unsupported terminal_value '{battery.terminal_value}'")


def nominal_block_hours(block: BlockBoundary) -> float:
    """The traded product's nominal duration (4h), NOT the count of steps
    actually present in this calendar day's grid for that block — a
    capacity payment is for the tendered product, and regelleistung.net
    settles it at the nominal duration regardless of a DST day's wall-clock
    stretching (see grid.py's DST spillover handling)."""
    return (block.end - block.start).total_seconds() / 3600.0


def _capacity_revenue_term(
    capacity_var,
    prices: pd.Series,
    market: str,
    price_basis: MarketPriceBasisConfig,
    blocks: list[BlockBoundary],
) -> linopy.LinearExpression:
    basis = price_basis.basis_for(market)
    terms = []
    for block in blocks:
        price = prices.loc[block.index]
        var_b = capacity_var.sel(block=block.index)
        if basis == "per_block_total":
            terms.append(price * var_b)
        elif basis == "per_mw_hour":
            terms.append(price * var_b * nominal_block_hours(block))
        else:
            raise ValueError(f"{market}: unsupported price_basis '{basis}' for a capacity market")
    return sum(terms)
