"""Batch runner (SPEC.md §6a, Milestone 8 Part B), `python -m src.batch`:
solves the full {perfect-foresight, model x quantile} x strategy matrix
over a date range and writes each run to results/, resuming already-solved
combinations via src.results_catalogue.scan_runs so re-running the command
after an interrupt or a config change doesn't re-solve everything.

Default matrix over 2026-03-21 to 2026-08-31: 3 perfect-foresight runs
(one per strategy) + 5 models x q=0.5 x 3 strategies = 18 runs, roughly
1.5-3h wall time (~4.5s/day/strategy; days within one run already solve
concurrently when cyclic_soc=True). Meant to run in the background.

A failed combination is logged loudly (full traceback) and the batch moves
on to the next one rather than aborting a multi-hour run over one bad
combination -- the failure is never swallowed, just not fatal to the rest
of the matrix.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback
from pathlib import Path

from src.config import (
    REPO_ROOT,
    AssumptionsConfig,
    MarketPriceBasisConfig,
    MarketTimingConfig,
    PathsConfig,
    SolverConfig,
    load_assumptions_config,
    load_battery_config,
    load_market_price_basis_config,
    load_market_timing_configs,
    load_paths_config,
    load_solver_config,
)
from src.model.battery import Battery, battery_from_config
from src.model.rolling import RollingRunResult, write_rolling_run_results
from src.results_catalogue import scan_runs
from src.valuation.benchmarks import (
    run_day_ahead_only_benchmark,
    DEFAULT_DA_PROXY_MODEL,
    run_full_cooptimized_benchmark,
    run_full_sequential_benchmark,
    run_naive_benchmark,
)

DEFAULT_START_DATE = dt.date(2026, 3, 21)
DEFAULT_END_DATE = dt.date(2026, 8, 31)
DEFAULT_MODELS = ["seasonal_naive", "linear_lags", "lgbm_point", "lgbm_quantile", "lgbm_quantile_tuned"]
DEFAULT_QUANTILES = [0.5]
ALL_STRATEGIES = ["day_ahead_only", "naive", "full_cooptimized", "full_sequential"]
# Forecast-driven only: with perfect foresight every stage sees identical prices.
FORECAST_ONLY_STRATEGIES = {"full_sequential"}
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def _run_strategy(
    strategy: str,
    start_date: dt.date,
    end_date: dt.date,
    battery: Battery,
    solver: SolverConfig,
    assumptions: AssumptionsConfig,
    price_basis: MarketPriceBasisConfig,
    timings: dict[str, MarketTimingConfig],
    paths: PathsConfig,
    decision_source: str,
    forecast_model: str | None,
    forecast_quantile: float | None,
) -> RollingRunResult:
    kwargs = dict(
        decision_source=decision_source, forecast_model=forecast_model, forecast_quantile=forecast_quantile,
    )
    if strategy == "day_ahead_only":
        return run_day_ahead_only_benchmark(
            start_date, end_date, battery, solver, timings["day_ahead"], timings["fcr"],
            paths_config=paths, **kwargs,
        )
    if strategy == "naive":
        return run_naive_benchmark(
            start_date, end_date, battery, timings["day_ahead"], timings["fcr"],
            paths_config=paths, **kwargs,
        )
    if strategy == "full_cooptimized":
        return run_full_cooptimized_benchmark(
            start_date, end_date, battery, solver, assumptions, price_basis,
            timings["day_ahead"], timings["fcr"], paths_config=paths, **kwargs,
        )
    if strategy == "full_sequential":
        return run_full_sequential_benchmark(
            start_date, end_date, battery, solver, assumptions, price_basis,
            timings["day_ahead"], timings["fcr"], paths_config=paths,
            forecast_model=forecast_model, forecast_quantile=forecast_quantile,
        )
    raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {ALL_STRATEGIES}")


def _fingerprint(battery: Battery, solver: SolverConfig, assumptions: AssumptionsConfig) -> dict:
    """The subset of config a matching prior run must agree on before
    it's treated as a valid resume point -- catches "config/*.yaml
    changed since that run solved" rather than silently reusing a stale
    result under new assumptions.
    """
    return {
        "battery.power_nominal_mw": battery.power_nominal_mw,
        "battery.power_derating_factor": battery.power_derating_factor,
        "battery.power_mw": battery.power_mw,
        "battery.energy_mwh": battery.energy_mwh,
        "battery.round_trip_efficiency": battery.round_trip_efficiency,
        "battery.soc_min_fraction": battery.soc_min_fraction,
        "battery.soc_max_fraction": battery.soc_max_fraction,
        "battery.aux_load_mw": battery.aux_load_mw,
        "battery.degradation_cost_eur_mwh": battery.degradation_cost_eur_mwh,
        "battery.terminal_value": battery.terminal_value,
        "solver.name": solver.name,
        "solver.mip_gap": solver.mip_gap,
        "solver.time_limit_seconds": solver.time_limit_seconds,
        "assumptions.alpha_pos": assumptions.alpha_pos,
        "assumptions.alpha_neg": assumptions.alpha_neg,
        "assumptions.activation_energy_price_eur_mwh": assumptions.activation_energy_price_eur_mwh,
        "assumptions.fcr_reserve_hours": assumptions.fcr_reserve_hours,
        "assumptions.afrr_reserve_hours": assumptions.afrr_reserve_hours,
    }


def _already_solved(
    catalogue,
    strategy: str,
    decision_source: str,
    forecast_model: str | None,
    forecast_quantile: float | None,
    start_date: dt.date,
    end_date: dt.date,
    fingerprint: dict,
) -> bool:
    if catalogue.empty:
        return False

    mask = (
        (catalogue["strategy"] == strategy)
        & (catalogue["decision_source"] == decision_source)
        & (catalogue["start_date"] == start_date.isoformat())
        & (catalogue["end_date"] == end_date.isoformat())
    )
    mask &= catalogue["forecast_model"].isna() if forecast_model is None else catalogue["forecast_model"] == forecast_model
    mask &= (
        catalogue["forecast_quantile"].isna()
        if forecast_quantile is None
        else catalogue["forecast_quantile"] == forecast_quantile
    )
    candidates = catalogue[mask]
    if candidates.empty:
        return False

    if strategy == "full_sequential":
        fingerprint = {**fingerprint, "da_proxy_model": DEFAULT_DA_PROXY_MODEL}

    for _, row in candidates.iterrows():
        if all(row.get(key) == value for key, value in fingerprint.items()):
            return True
    return False


def build_matrix(
    models: list[str], quantiles: list[float], strategies: list[str], include_perfect_foresight: bool,
) -> list[tuple[str, str, str | None, float | None]]:
    """Every (strategy, decision_source, forecast_model, forecast_quantile)
    combination the batch should attempt, in a stable order.
    """
    combos: list[tuple[str, str, str | None, float | None]] = []
    for strategy in strategies:
        if include_perfect_foresight and strategy not in FORECAST_ONLY_STRATEGIES:
            combos.append((strategy, "realized", None, None))
        for model in models:
            for quantile in quantiles:
                combos.append((strategy, "forecast", model, quantile))
    return combos


def run_batch(
    start_date: dt.date,
    end_date: dt.date,
    models: list[str],
    quantiles: list[float],
    strategies: list[str],
    include_perfect_foresight: bool,
    force: bool,
    dry_run: bool,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    paths_config: PathsConfig | None = None,
) -> None:
    battery = battery_from_config(load_battery_config())
    solver = load_solver_config()
    assumptions = load_assumptions_config()
    price_basis = load_market_price_basis_config()
    paths = paths_config or load_paths_config()
    timings = load_market_timing_configs(paths)
    fingerprint = _fingerprint(battery, solver, assumptions)

    combos = build_matrix(models, quantiles, strategies, include_perfect_foresight)
    catalogue = scan_runs(results_dir)

    solved, skipped, failed = 0, 0, 0
    for strategy, decision_source, forecast_model, forecast_quantile in combos:
        label = (
            f"{strategy} | {decision_source}"
            + (f" | {forecast_model} q{forecast_quantile}" if decision_source == "forecast" else "")
        )

        if not force and _already_solved(
            catalogue, strategy, decision_source, forecast_model, forecast_quantile, start_date, end_date, fingerprint,
        ):
            print(f"[skip]  {label} (already solved, matching config)")
            skipped += 1
            continue

        if dry_run:
            print(f"[would run] {label}")
            continue

        print(f"[run]   {label} ...", flush=True)
        try:
            result = _run_strategy(
                strategy, start_date, end_date, battery, solver, assumptions, price_basis, timings, paths,
                decision_source, forecast_model, forecast_quantile,
            )
            write_rolling_run_results(
                result, results_dir,
                strategy=strategy, start_date=start_date, end_date=end_date,
                battery=battery, solver_config=solver, assumptions=assumptions, price_basis=price_basis,
                paths_config=paths,
            )
            catalogue = scan_runs(results_dir)
            total = float(result.daily_revenue["total"].sum()) if not result.daily_revenue.empty else 0.0
            print(f"[done]  {label} -> solve_id={result.solve_id} net_revenue_eur={total:.2f}")
            solved += 1
        except Exception:
            print(f"[FAILED] {label}")
            traceback.print_exc()
            failed += 1

    if dry_run:
        print(f"\nDry run: {len(combos)} combinations in the matrix, {skipped} already solved, "
              f"{len(combos) - skipped} would run.")
    else:
        print(f"\nBatch complete: {solved} solved, {skipped} skipped (already solved), {failed} failed, "
              f"out of {len(combos)} combinations.")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Batch-solve the forecast-model x strategy results matrix")
    parser.add_argument("--start-date", type=dt.date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=dt.date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--quantiles", nargs="+", type=float, default=DEFAULT_QUANTILES)
    parser.add_argument("--strategies", nargs="+", choices=ALL_STRATEGIES, default=ALL_STRATEGIES)
    parser.add_argument("--include-perfect-foresight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Re-solve even if a matching run already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print the matrix without solving anything")
    args = parser.parse_args(argv)

    run_batch(
        start_date=args.start_date,
        end_date=args.end_date,
        models=args.models,
        quantiles=args.quantiles,
        strategies=args.strategies,
        include_perfect_foresight=args.include_perfect_foresight,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
