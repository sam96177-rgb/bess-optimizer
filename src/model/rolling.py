"""Rolling multi-day run (SPEC.md Milestone 5): solves each available day
independently, or chained by carrying SOC across day boundaries when
cyclic_soc=False, and assembles long-format results plus a per-day
revenue breakdown. Skipped days (from forecast_loader) are carried
through to a skip log rather than vanishing silently.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pickle
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT, AssumptionsConfig, MarketPriceBasisConfig, MarketTimingConfig, PathsConfig, SolverConfig
from src.data.forecast_loader import DayPricePair, DayPrices, SkipRecord, load_forecast_run_ids, load_price_pair_range
from src.model.build import AncillaryInputs, terminal_value_eur_mwh
from src.model.battery import Battery
from src.model.solve import SolveResult
from src.valuation.revenue import RevenueBreakdown, compute_revenue

LONG_FORMAT_COLUMNS = [
    "solve_id", "solve_date", "target_time", "block_id", "market", "variable", "value", "unit", "is_assumption",
]


@dataclass(frozen=True)
class RollingRunResult:
    solve_id: str
    long_format: pd.DataFrame
    daily_revenue: pd.DataFrame
    skip_log: list[SkipRecord]
    decision_source: str = "realized"
    forecast_model: str | None = None
    forecast_quantile: float | None = None


def day_to_long_rows(solve_id: str, day_prices: DayPrices, result: SolveResult, e_is_assumption: bool) -> list[dict]:
    """Shared by run_rolling and benchmarks.py (day-ahead-only and naive
    strategies also need this exact schema for a fair, uniform comparison).
    `e_is_assumption` should be True only when ancillary markets AND
    activation-in-SOC are both in play for this result (see SPEC.md Hard
    Rule 2 note on 'e' below).
    """
    rows = []
    for step in day_prices.grid.steps:
        for variable, unit in (("p_ch", "MW"), ("p_dis", "MW"), ("u", "dimensionless"), ("e", "MWh")):
            series = getattr(result, variable)
            rows.append(
                {
                    "solve_id": solve_id,
                    "solve_date": day_prices.day,
                    "target_time": step,
                    "block_id": None,
                    "market": "day_ahead" if variable in ("p_ch", "p_dis") else "battery",
                    "variable": variable,
                    "value": float(series.loc[step]),
                    "unit": unit,
                    # e's own value directly folds in the assumed aFRR
                    # activation energy term (C1) whenever that's enabled --
                    # SPEC.md Hard Rule 2: an assumption must never be mixed
                    # into a results file without being flagged there too.
                    "is_assumption": variable == "e" and e_is_assumption,
                }
            )

    if result.c_fcr is not None:
        for block in day_prices.grid.blocks:
            for variable, market, unit in (
                ("c_fcr", "fcr", "MW"),
                ("c_pos", "afrr_pos", "MW"),
                ("c_neg", "afrr_neg", "MW"),
            ):
                series = getattr(result, variable)
                rows.append(
                    {
                        "solve_id": solve_id,
                        "solve_date": day_prices.day,
                        "target_time": block.start,
                        "block_id": block.index,
                        "market": market,
                        "variable": variable,
                        "value": float(series.loc[block.index]),
                        "unit": unit,
                        "is_assumption": False,
                    }
                )
    return rows


def _solve_day_via_subprocess(
    day_prices: DayPrices,
    battery: Battery,
    solver_config: SolverConfig,
    cyclic_soc: bool,
    enforce_exclusivity: bool,
    initial_soc_mwh: float | None,
    ancillary: AncillaryInputs | None,
) -> SolveResult:
    """Runs build_model+solve_model for exactly one day in a fresh
    subprocess (src/model/_day_solve_worker.py) instead of in this
    process. See that module's docstring for why: a single long-running
    process solving 150+ days in a row was observed to progressively slow
    from ~4.5s/day to 50-120+s/day, consistent with a memory/state
    accumulation issue in linopy or HiGHS across many sequential Model()
    calls in one process. A fresh process per day sidesteps it.
    """
    task = {
        "day_prices": day_prices,
        "battery": battery,
        "solver_config": solver_config,
        "cyclic_soc": cyclic_soc,
        "enforce_exclusivity": enforce_exclusivity,
        "initial_soc_mwh": initial_soc_mwh,
        "ancillary": ancillary,
    }
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "task.pkl"
        output_path = Path(tmp) / "result.pkl"
        with open(input_path, "wb") as f:
            pickle.dump(task, f)

        proc = subprocess.run(
            [sys.executable, "-m", "src.model._day_solve_worker", str(input_path), str(output_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Worker subprocess failed solving {day_prices.day}:\n{proc.stderr}"
            )
        with open(output_path, "rb") as f:
            return pickle.load(f)


def run_rolling(
    start_date: dt.date,
    end_date: dt.date,
    battery: Battery,
    solver_config: SolverConfig,
    day_ahead_timing: MarketTimingConfig,
    capacity_timing: MarketTimingConfig,
    assumptions: AssumptionsConfig | None = None,
    price_basis: MarketPriceBasisConfig | None = None,
    cyclic_soc: bool = True,
    enforce_exclusivity: bool = True,
    include_activation_in_soc: bool = True,
    include_ancillary: bool = True,
    paths_config: PathsConfig | None = None,
    decision_source: str = "realized",
    forecast_model: str | None = None,
    forecast_quantile: float | None = None,
) -> RollingRunResult:
    """The MILP-based strategies (SPEC.md Milestone 5's main run, and
    Milestone 6's day-ahead-only benchmark) share this loop. Set
    include_ancillary=False for day-ahead-only arbitrage with no capacity
    markets at all (build_model's ancillary=None path) — assumptions and
    price_basis are then not needed and may be omitted.

    decision_source="forecast" makes the solver decide on forecast_model's
    y_pred (at forecast_quantile) while revenue is still settled on y_true
    (SPEC.md §6a: decide on forecast, settle on realized) — see
    src/data/forecast_loader.py's DayPricePair.
    """
    if include_ancillary and (assumptions is None or price_basis is None):
        raise ValueError("assumptions and price_basis are required when include_ancillary=True")

    pairs, skip_log = load_price_pair_range(
        start_date, end_date, day_ahead_timing, capacity_timing, paths_config,
        decision_source=decision_source, forecast_model=forecast_model, forecast_quantile=forecast_quantile,
    )

    solve_id = str(uuid.uuid4())
    e_is_assumption = include_ancillary and include_activation_in_soc

    def _ancillary_for(day_prices: DayPrices) -> AncillaryInputs | None:
        if not include_ancillary:
            return None
        return AncillaryInputs(
            blocks=day_prices.grid.blocks,
            step_block_index=day_prices.grid.step_block_index,
            fcr_prices=day_prices.fcr,
            afrr_pos_prices=day_prices.afrr_pos,
            afrr_neg_prices=day_prices.afrr_neg,
            price_basis=price_basis,
            assumptions=assumptions,
            include_activation_in_soc=include_activation_in_soc,
        )

    if cyclic_soc:
        # Each day is fully independent (no SOC carried across the
        # boundary), so they can be solved concurrently -- each still in
        # its own subprocess. The solver only ever sees pair.decision --
        # pair.settlement (y_true) is not part of the subprocess payload.
        def _task(pair: DayPricePair) -> tuple[DayPricePair, SolveResult]:
            result = _solve_day_via_subprocess(
                pair.decision, battery, solver_config, cyclic_soc, enforce_exclusivity, None, _ancillary_for(pair.decision),
            )
            return pair, result

        with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4))) as pool:
            solved = list(pool.map(_task, pairs))
    else:
        # Day N+1's initial SOC depends on day N's solved ending SOC, so
        # this chain must run in order.
        solved = []
        carried_soc_mwh: float | None = None
        for pair in pairs:
            initial_soc_mwh = carried_soc_mwh if carried_soc_mwh is not None else battery.soc_min_mwh
            result = _solve_day_via_subprocess(
                pair.decision, battery, solver_config, cyclic_soc, enforce_exclusivity, initial_soc_mwh, _ancillary_for(pair.decision),
            )
            carried_soc_mwh = float(result.e.iloc[-1])
            solved.append((pair, result))

    long_rows: list[dict] = []
    daily_revenue_rows: list[dict] = []
    for pair, result in solved:
        # Revenue is always settled on y_true, even for a forecast-driven
        # run (SPEC.md §6a) -- this is the only place settlement prices
        # are used, and they never reach the solver subprocess above.
        # The terminal credit lambda is decision-side (what the solver
        # valued leftover energy at), so it is derived from decision prices.
        revenue = compute_revenue(
            result, pair.settlement.day_ahead, pair.settlement.grid.delta_t_hours, _ancillary_for(pair.settlement),
            battery=battery,
            terminal_value_eur_mwh=terminal_value_eur_mwh(battery, pair.decision.day_ahead, cyclic_soc),
        )
        long_rows.extend(day_to_long_rows(solve_id, pair.decision, result, e_is_assumption))
        daily_revenue_rows.append(revenue_to_row(pair.day, revenue, result, pair.settlement.grid.delta_t_hours))

    long_format = pd.DataFrame(long_rows, columns=LONG_FORMAT_COLUMNS)
    daily_revenue = pd.DataFrame(daily_revenue_rows)
    return RollingRunResult(
        solve_id=solve_id, long_format=long_format, daily_revenue=daily_revenue, skip_log=skip_log,
        decision_source=decision_source, forecast_model=forecast_model, forecast_quantile=forecast_quantile,
    )


def revenue_to_row(day: dt.date, revenue: RevenueBreakdown, result: SolveResult, delta_t_hours: float) -> dict:
    """`total` is NET revenue (gross - degradation), kept under its old name
    so consumers of daily_revenue.parquet keep meaning "what the battery
    earned". `terminal_credit` is a decision-side value and is not in `net`."""
    return {
        "solve_date": day,
        "day_ahead": revenue.day_ahead,
        "fcr": revenue.fcr,
        "afrr_pos": revenue.afrr_pos,
        "afrr_neg": revenue.afrr_neg,
        "activation": revenue.activation,
        "activation_is_assumption": True,
        "gross": revenue.gross,
        "degradation": revenue.degradation,
        "net": revenue.net,
        "total": revenue.net,
        "terminal_credit": revenue.terminal_credit,
        "charged_mwh": float(result.p_ch.sum() * delta_t_hours),
        "discharged_mwh": float(result.p_dis.sum() * delta_t_hours),
    }


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_run_meta(
    result: RollingRunResult,
    strategy: str,
    start_date: dt.date,
    end_date: dt.date,
    battery: Battery,
    solver_config: SolverConfig,
    assumptions: AssumptionsConfig,
    price_basis: MarketPriceBasisConfig,
    paths_config: PathsConfig,
    cyclic_soc: bool = True,
    enforce_exclusivity: bool = True,
    include_activation_in_soc: bool = True,
) -> dict:
    """Assembles the run_meta.json sidecar (SPEC.md §6a / Milestone 8
    Part B) — the one record src/results_catalogue.py's scan_runs reads
    per run, and what batch.py fingerprints against for resumability.
    """
    dr = result.daily_revenue
    gross_eur = float(dr["gross"].sum()) if not dr.empty else 0.0
    degradation_eur = float(dr["degradation"].sum()) if not dr.empty else 0.0
    net_eur = float(dr["net"].sum()) if not dr.empty else 0.0
    usable_mwh = battery.soc_max_mwh - battery.soc_min_mwh
    equivalent_full_cycles = float(dr["discharged_mwh"].sum() / usable_mwh) if not dr.empty else 0.0
    forecast_run_ids = (
        load_forecast_run_ids(paths_config, result.forecast_model)
        if result.decision_source == "forecast" and result.forecast_model is not None
        else {}
    )
    return {
        "solve_id": result.solve_id,
        "strategy": strategy,
        "decision_source": result.decision_source,
        "forecast_model": result.forecast_model,
        "forecast_quantile": result.forecast_quantile,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "n_days_solved": len(result.daily_revenue),
        "n_days_skipped": len(result.skip_log),
        "battery": {
            "power_nominal_mw": battery.power_nominal_mw,
            "power_derating_factor": battery.power_derating_factor,
            "power_mw": battery.power_mw,
            "energy_mwh": battery.energy_mwh,
            "round_trip_efficiency": battery.round_trip_efficiency,
            "soc_min_fraction": battery.soc_min_fraction,
            "soc_max_fraction": battery.soc_max_fraction,
            "aux_load_mw": battery.aux_load_mw,
            "degradation_cost_eur_mwh": battery.degradation_cost_eur_mwh,
            "terminal_value": battery.terminal_value,
        },
        "solver": {
            "name": solver_config.name,
            "mip_gap": solver_config.mip_gap,
            "time_limit_seconds": solver_config.time_limit_seconds,
        },
        "model_flags": {
            "cyclic_soc": cyclic_soc,
            "enforce_exclusivity": enforce_exclusivity,
            "include_activation_in_soc": include_activation_in_soc,
        },
        "assumptions": {
            "alpha_pos": assumptions.alpha_pos,
            "alpha_neg": assumptions.alpha_neg,
            "activation_energy_price_eur_mwh": assumptions.activation_energy_price_eur_mwh,
            "fcr_reserve_hours": assumptions.fcr_reserve_hours,
            "afrr_reserve_hours": assumptions.afrr_reserve_hours,
        },
        "price_basis": dict(price_basis.basis_by_market),
        "gross_revenue_eur": gross_eur,
        "degradation_cost_eur": degradation_eur,
        "net_revenue_eur": net_eur,
        "total_revenue_eur": net_eur,
        "equivalent_full_cycles": equivalent_full_cycles,
        "forecast_repo_path": str(paths_config.forecast_repo_path),
        "forecast_run_ids": forecast_run_ids,
        "git_commit": _git_commit(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def write_rolling_run_results(
    result: RollingRunResult,
    results_dir,
    *,
    strategy: str,
    start_date: dt.date,
    end_date: dt.date,
    battery: Battery,
    solver_config: SolverConfig,
    assumptions: AssumptionsConfig,
    price_basis: MarketPriceBasisConfig,
    paths_config: PathsConfig,
    cyclic_soc: bool = True,
    enforce_exclusivity: bool = True,
    include_activation_in_soc: bool = True,
) -> None:
    """Persists a run's outputs under results/. Kept separate from
    run_rolling itself so the solve logic stays testable without disk I/O.

    run_meta.json is written LAST, after both parquets and the skip log,
    so a run that crashes mid-write leaves no sidecar -- it stays
    invisible to results_catalogue.scan_runs and gets retried by batch.py
    rather than being mistaken for a completed run.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_dir = results_dir / result.solve_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result.long_format.to_parquet(run_dir / "long_format.parquet", index=False)
    result.daily_revenue.to_parquet(run_dir / "daily_revenue.parquet", index=False)

    # Explicit columns even when skip_log is empty, so the file always has
    # a header row -- a column-less empty CSV can't be read back by
    # pd.read_csv (EmptyDataError), which would make a clean run's own
    # skip log unreadable.
    skip_df = pd.DataFrame([{"day": s.day, "reason": s.reason} for s in result.skip_log], columns=["day", "reason"])
    skip_df.to_csv(run_dir / "skip_log.csv", index=False)

    meta = build_run_meta(
        result, strategy, start_date, end_date, battery, solver_config, assumptions, price_basis, paths_config,
        cyclic_soc=cyclic_soc, enforce_exclusivity=enforce_exclusivity, include_activation_in_soc=include_activation_in_soc,
    )
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
