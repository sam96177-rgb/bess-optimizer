"""Shared scenario builders for the physics tests. Prices are synthetic and
generated on the real day grid (so DST days get their real 92/100 steps);
they are test inputs only and never feed a reported number.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    load_assumptions_config,
    load_battery_config,
    load_market_price_basis_config,
    load_market_timing_configs,
    load_solver_config,
)
from src.data.grid import DayGrid, build_day_grid
from src.model.battery import battery_from_config
from src.model.build import AncillaryInputs, build_model
from src.model.solve import SolveResult, solve_model

BATTERY = battery_from_config(load_battery_config())
SOLVER = load_solver_config()
ASSUMPTIONS = load_assumptions_config()
PRICE_BASIS = load_market_price_basis_config()
TIMINGS = load_market_timing_configs()

NORMAL_DAY = dt.date(2026, 6, 1)
SPRING_FORWARD_DAY = dt.date(2026, 3, 29)  # 92 quarter-hours
FALL_BACK_DAY = dt.date(2026, 10, 25)  # 100 quarter-hours


def grid_for(day: dt.date) -> DayGrid:
    return build_day_grid(day, TIMINGS["day_ahead"], TIMINGS["fcr"])


def block_prices(grid: DayGrid, value) -> pd.Series:
    n = len(grid.blocks)
    values = [value] * n if isinstance(value, (int, float)) else list(value)
    return pd.Series(values, index=[b.index for b in grid.blocks], dtype=float)


def random_day_ahead(grid: DayGrid, seed: int, low: float = -20.0, high: float = 200.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(low, high, size=len(grid.steps)), index=grid.steps)


@dataclass(frozen=True)
class Case:
    grid: DayGrid
    day_ahead: pd.Series
    ancillary: AncillaryInputs | None
    result: SolveResult


def solve_case(
    grid: DayGrid,
    day_ahead: pd.Series,
    fcr: pd.Series | None = None,
    afrr_pos: pd.Series | None = None,
    afrr_neg: pd.Series | None = None,
    *,
    battery=BATTERY,
    with_ancillary: bool = True,
    include_activation_in_soc: bool = True,
    price_basis=PRICE_BASIS,
    **build_kwargs,
) -> Case:
    ancillary = None
    if with_ancillary:
        zero = block_prices(grid, 0.0)
        ancillary = AncillaryInputs(
            blocks=grid.blocks,
            step_block_index=grid.step_block_index,
            fcr_prices=zero if fcr is None else fcr,
            afrr_pos_prices=zero if afrr_pos is None else afrr_pos,
            afrr_neg_prices=zero if afrr_neg is None else afrr_neg,
            price_basis=price_basis,
            assumptions=ASSUMPTIONS,
            include_activation_in_soc=include_activation_in_soc,
        )
    model = build_model(
        steps=grid.steps,
        day_ahead_prices=day_ahead,
        battery=battery,
        delta_t_hours=grid.delta_t_hours,
        ancillary=ancillary,
        **build_kwargs,
    )
    return Case(grid=grid, day_ahead=day_ahead, ancillary=ancillary, result=solve_model(model, SOLVER))


def random_ancillary_case(day: dt.date = NORMAL_DAY, seed: int = 0, **kwargs) -> Case:
    grid = grid_for(day)
    rng = np.random.default_rng(seed + 100)
    n = len(grid.blocks)
    return solve_case(
        grid,
        random_day_ahead(grid, seed),
        block_prices(grid, rng.uniform(20, 150, n)),
        block_prices(grid, rng.uniform(1, 30, n)),
        block_prices(grid, rng.uniform(1, 30, n)),
        **kwargs,
    )
