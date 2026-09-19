"""Three strategies evaluated over the same historical day set, so
they're directly comparable (SPEC.md Milestone 6):

1. day_ahead_only benchmark    -- pure day-ahead arbitrage, no ancillary
   markets. Same MILP core as the main run (build_model with
   ancillary=None), via rolling.run_rolling(include_ancillary=False).
2. naive benchmark              -- no optimization at all: charge fully
   during the day's single cheapest 4h block, discharge fully during its
   single most expensive 4h block, no ancillary. A simple rule an
   unsophisticated real operator might follow.
3. full_cooptimized benchmark   -- the main run from Milestone 5
   (rolling.run_rolling with ancillary enabled), included here as one of
   three so all three live behind one comparison entry point.
"""
from __future__ import annotations

import datetime as dt
import uuid

import pandas as pd

from src.config import (
    AssumptionsConfig,
    MarketPriceBasisConfig,
    MarketTimingConfig,
    PathsConfig,
    SolverConfig,
)
from src.data.forecast_loader import DayPrices, load_price_pair_range
from src.model.battery import Battery
from src.model.build import build_model
from src.model.rolling import LONG_FORMAT_COLUMNS, RollingRunResult, day_to_long_rows, revenue_to_row, run_rolling
from src.model.solve import SolveResult, solve_model
from src.valuation.revenue import compute_revenue


def run_day_ahead_only_benchmark(
    start_date: dt.date,
    end_date: dt.date,
    battery: Battery,
    solver_config: SolverConfig,
    day_ahead_timing: MarketTimingConfig,
    capacity_timing: MarketTimingConfig,
    cyclic_soc: bool = True,
    enforce_exclusivity: bool = True,
    paths_config: PathsConfig | None = None,
    decision_source: str = "realized",
    forecast_model: str | None = None,
    forecast_quantile: float | None = None,
) -> RollingRunResult:
    return run_rolling(
        start_date=start_date,
        end_date=end_date,
        battery=battery,
        solver_config=solver_config,
        day_ahead_timing=day_ahead_timing,
        capacity_timing=capacity_timing,
        cyclic_soc=cyclic_soc,
        enforce_exclusivity=enforce_exclusivity,
        include_ancillary=False,
        paths_config=paths_config,
        decision_source=decision_source,
        forecast_model=forecast_model,
        forecast_quantile=forecast_quantile,
    )


def run_full_cooptimized_benchmark(
    start_date: dt.date,
    end_date: dt.date,
    battery: Battery,
    solver_config: SolverConfig,
    assumptions: AssumptionsConfig,
    price_basis: MarketPriceBasisConfig,
    day_ahead_timing: MarketTimingConfig,
    capacity_timing: MarketTimingConfig,
    cyclic_soc: bool = True,
    enforce_exclusivity: bool = True,
    paths_config: PathsConfig | None = None,
    decision_source: str = "realized",
    forecast_model: str | None = None,
    forecast_quantile: float | None = None,
) -> RollingRunResult:
    return run_rolling(
        start_date=start_date,
        end_date=end_date,
        battery=battery,
        solver_config=solver_config,
        assumptions=assumptions,
        price_basis=price_basis,
        day_ahead_timing=day_ahead_timing,
        capacity_timing=capacity_timing,
        cyclic_soc=cyclic_soc,
        enforce_exclusivity=enforce_exclusivity,
        include_ancillary=True,
        paths_config=paths_config,
        decision_source=decision_source,
        forecast_model=forecast_model,
        forecast_quantile=forecast_quantile,
    )


STRICT_DA_PROXY_MODEL = "seasonal_naive"


def run_full_sequential_benchmark(
    start_date: dt.date,
    end_date: dt.date,
    battery: Battery,
    solver_config: SolverConfig,
    assumptions: AssumptionsConfig,
    price_basis: MarketPriceBasisConfig,
    day_ahead_timing: MarketTimingConfig,
    capacity_timing: MarketTimingConfig,
    cyclic_soc: bool = True,
    enforce_exclusivity: bool = True,
    paths_config: PathsConfig | None = None,
    decision_source: str = "realized",
    forecast_model: str | None = None,
    forecast_quantile: float | None = None,
    da_proxy_model: str | None = None,
) -> RollingRunResult:
    """Gate-ordered clearing (FCR -> aFRR -> DA); see src/model/sequential.py.

    da_proxy_model=None: every stage sees the same prices. da_proxy_model=X
    (forecast runs only): the FCR/aFRR stages see model X's day-ahead
    forecast instead -- STRICT_DA_PROXY_MODEL is the worst-case choice.
    With realized prices (perfect foresight) every stage sees the real
    prices, which by construction reproduces full_cooptimized.
    """
    return run_rolling(
        start_date=start_date,
        end_date=end_date,
        battery=battery,
        solver_config=solver_config,
        assumptions=assumptions,
        price_basis=price_basis,
        day_ahead_timing=day_ahead_timing,
        capacity_timing=capacity_timing,
        cyclic_soc=cyclic_soc,
        enforce_exclusivity=enforce_exclusivity,
        include_ancillary=True,
        paths_config=paths_config,
        decision_source=decision_source,
        forecast_model=forecast_model,
        forecast_quantile=forecast_quantile,
        da_proxy_model=da_proxy_model,
        sequential=True,
    )


def _cheapest_and_priciest_blocks(day_prices: DayPrices) -> tuple[int, int]:
    """'Block' here reuses this repo's existing 4-hour capacity-block
    structure (grid.py), rather than inventing a second, differently-sized
    notion of 'period' just for this heuristic. Ranked by each block's
    mean day-ahead price across its own steps.
    """
    avg_price_by_block = {}
    for block in day_prices.grid.blocks:
        block_steps = day_prices.grid.steps[
            [bi == block.index for bi in day_prices.grid.step_block_index]
        ]
        avg_price_by_block[block.index] = day_prices.day_ahead.loc[block_steps].mean()
    cheapest = min(avg_price_by_block, key=avg_price_by_block.get)
    priciest = max(avg_price_by_block, key=avg_price_by_block.get)
    return cheapest, priciest


def simulate_naive_day(day_prices: DayPrices, battery: Battery, starting_soc_mwh: float) -> SolveResult:
    """Charge at full power throughout the cheapest block (capped once
    soc_max is reached), discharge at full power throughout the most
    expensive block (capped once soc_min is reached), do nothing in every
    other step. No lookahead, no optimization -- a naive rule, not a
    solver result, but returned in the same SolveResult shape so it can
    reuse compute_revenue and rolling.day_to_long_rows unmodified.

    `day_prices` is the DECISION side (SPEC.md §6a) -- the cheapest/
    priciest block picks and objective_value both reflect what the rule
    expected to earn at decision time, not what it actually earned. Call
    this with pair.decision; settle separately via compute_revenue on
    pair.settlement.
    """
    cheapest_block, priciest_block = _cheapest_and_priciest_blocks(day_prices)
    dt_h = day_prices.grid.delta_t_hours
    steps = day_prices.grid.steps

    p_ch = pd.Series(0.0, index=steps)
    p_dis = pd.Series(0.0, index=steps)
    u = pd.Series(0, index=steps, dtype=int)
    e = pd.Series(index=steps, dtype=float)

    soc = starting_soc_mwh
    for step, block_idx in zip(steps, day_prices.grid.step_block_index):
        # Parasitic load drains the battery every step. Clamped at soc_min:
        # the heuristic has no lookahead to refill, so at the floor the aux
        # load is treated as grid-supplied rather than breaching the window.
        soc = max(battery.soc_min_mwh, soc - battery.aux_load_mw * dt_h)
        if block_idx == cheapest_block:
            headroom_mwh = max(0.0, battery.soc_max_mwh - soc)
            max_charge_energy_mwh = battery.power_mw * dt_h * battery.eta
            charge_energy_mwh = min(max_charge_energy_mwh, headroom_mwh)
            charge_power_mw = charge_energy_mwh / (battery.eta * dt_h)
            soc += battery.eta * charge_power_mw * dt_h
            p_ch.loc[step] = charge_power_mw
            u.loc[step] = 1
        elif block_idx == priciest_block:
            available_mwh = max(0.0, soc - battery.soc_min_mwh)
            max_discharge_power_from_soc_mw = available_mwh * battery.eta / dt_h
            discharge_power_mw = min(battery.power_mw, max_discharge_power_from_soc_mw)
            soc -= (dt_h / battery.eta) * discharge_power_mw
            p_dis.loc[step] = discharge_power_mw
        e.loc[step] = soc

    objective_value = float(
        (day_prices.day_ahead * (p_dis - p_ch) * dt_h).sum()
        - battery.degradation_cost_eur_mwh * (p_ch + p_dis).sum() * dt_h
    )
    return SolveResult(
        objective_value=objective_value, p_ch=p_ch, p_dis=p_dis, u=u, e=e, c_fcr=None, c_pos=None, c_neg=None,
    )


def run_naive_benchmark(
    start_date: dt.date,
    end_date: dt.date,
    battery: Battery,
    day_ahead_timing: MarketTimingConfig,
    capacity_timing: MarketTimingConfig,
    reset_daily_soc: bool = True,
    paths_config: PathsConfig | None = None,
    decision_source: str = "realized",
    forecast_model: str | None = None,
    forecast_quantile: float | None = None,
) -> RollingRunResult:
    """reset_daily_soc=True (default) starts every day at soc_min,
    matching the MILP strategies' cyclic_soc=True convention (SPEC.md
    Milestone 5: cyclic_soc means each day 'resets' rather than carrying
    SOC across the boundary) so all three benchmarks are comparable on
    equal terms. This heuristic has no terminal-SOC constraint of its own
    (it isn't an optimization), so 'reset' here just means each day's
    simulation starts fresh rather than continuing from the prior day's
    ending SOC.

    decision_source="forecast" picks the cheapest/priciest block from
    forecast_model's y_pred but always settles revenue on y_true (SPEC.md
    §6a) -- see simulate_naive_day.
    """
    pairs, skip_log = load_price_pair_range(
        start_date, end_date, day_ahead_timing, capacity_timing, paths_config,
        decision_source=decision_source, forecast_model=forecast_model, forecast_quantile=forecast_quantile,
    )

    solve_id = str(uuid.uuid4())
    long_rows: list[dict] = []
    daily_revenue_rows: list[dict] = []

    soc = battery.soc_min_mwh
    for pair in pairs:
        starting_soc = battery.soc_min_mwh if reset_daily_soc else soc
        result = simulate_naive_day(pair.decision, battery, starting_soc)
        soc = float(result.e.iloc[-1])

        revenue = compute_revenue(
            result, pair.settlement.day_ahead, pair.settlement.grid.delta_t_hours, ancillary=None, battery=battery,
        )
        long_rows.extend(day_to_long_rows(solve_id, pair.decision, result, e_is_assumption=False))
        daily_revenue_rows.append(revenue_to_row(pair.day, revenue, result, pair.settlement.grid.delta_t_hours))

    long_format = pd.DataFrame(long_rows, columns=LONG_FORMAT_COLUMNS)
    daily_revenue = pd.DataFrame(daily_revenue_rows)
    return RollingRunResult(
        solve_id=solve_id, long_format=long_format, daily_revenue=daily_revenue, skip_log=skip_log,
        decision_source=decision_source, forecast_model=forecast_model, forecast_quantile=forecast_quantile,
    )


def run_all_benchmarks(
    start_date: dt.date,
    end_date: dt.date,
    battery: Battery,
    solver_config: SolverConfig,
    assumptions: AssumptionsConfig,
    price_basis: MarketPriceBasisConfig,
    day_ahead_timing: MarketTimingConfig,
    capacity_timing: MarketTimingConfig,
    paths_config: PathsConfig | None = None,
    decision_source: str = "realized",
    forecast_model: str | None = None,
    forecast_quantile: float | None = None,
) -> dict[str, RollingRunResult]:
    strategies = {
        "day_ahead_only": run_day_ahead_only_benchmark(
            start_date, end_date, battery, solver_config, day_ahead_timing, capacity_timing, paths_config=paths_config,
            decision_source=decision_source, forecast_model=forecast_model, forecast_quantile=forecast_quantile,
        ),
        "naive": run_naive_benchmark(
            start_date, end_date, battery, day_ahead_timing, capacity_timing, paths_config=paths_config,
            decision_source=decision_source, forecast_model=forecast_model, forecast_quantile=forecast_quantile,
        ),
        "full_cooptimized": run_full_cooptimized_benchmark(
            start_date, end_date, battery, solver_config, assumptions, price_basis,
            day_ahead_timing, capacity_timing, paths_config=paths_config,
            decision_source=decision_source, forecast_model=forecast_model, forecast_quantile=forecast_quantile,
        ),
    }
    strategies["full_sequential"] = run_full_sequential_benchmark(
        start_date, end_date, battery, solver_config, assumptions, price_basis,
        day_ahead_timing, capacity_timing, paths_config=paths_config,
        decision_source=decision_source, forecast_model=forecast_model, forecast_quantile=forecast_quantile,
    )
    return strategies
