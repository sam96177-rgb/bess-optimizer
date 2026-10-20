"""Batch runner (SPEC.md §6a, Milestone 8 Part B), `python -m src.batch`:
solves the full {perfect-foresight, forecast model} x strategy matrix (median forecasts only)
over a date range and writes each run to results/, resuming already-solved
combinations via src.results_catalogue.scan_runs so re-running the command
after an interrupt or a config change doesn't re-solve everything.

Default matrix, models/strategies/window from config/study.yaml (V5-6): 2
perfect-foresight runs (one per strategy) + N models x q=0.5 x 2
strategies (N=5, 12 combos total, under the v5 study; N=4, 10 combos under
v6 -- PLAN_V6_REVISION.md V6-2). Measured ~15min wall time for the v5
study's 12-combo default after the V5-4 build vectorisation (days within
one run already solve concurrently when cyclic_soc=True). Meant to run in
the background. The
full_sequential strategy is not in the default matrix; request it with
`--strategies`.

A failed combination is logged loudly (full traceback) and the batch moves
on to the next one rather than aborting a long run over one bad
combination -- the failure is never swallowed, just not fatal to the rest
of the matrix.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

from src.config import (
    REPO_ROOT,
    AssumptionsConfig,
    MarketPriceBasisConfig,
    PathsConfig,
    SolverConfig,
    load_assumptions_config,
    load_battery_config,
    load_market_price_basis_config,
    load_market_timing_configs,
    load_paths_config,
    load_solver_config,
    load_study_config,
)
from src.data.forecast_loader import MEDIAN_QUANTILE, load_forecast_run_ids
from src.model.battery import Battery, battery_from_config
from src.model.build import FCR_POWER_FACTOR
from src.model.rolling import write_rolling_run_results
from src.results_catalogue import scan_runs
from src.valuation.benchmarks import STRATEGY_FLAGS, run_strategy

_STUDY = load_study_config()
DEFAULT_START_DATE, DEFAULT_END_DATE = _STUDY.optimization_window
DEFAULT_MODELS = _STUDY.optimizer_models
DEFAULT_STRATEGIES = _STUDY.strategies
ALL_STRATEGIES = list(STRATEGY_FLAGS)  # full_sequential is not in DEFAULT_STRATEGIES: opt in with --strategies

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def _fingerprint(
    battery: Battery, solver: SolverConfig, assumptions: AssumptionsConfig, price_basis: MarketPriceBasisConfig,
) -> dict:
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
        "model.fcr_power_factor": FCR_POWER_FACTOR,
        "solver.name": solver.name,
        "solver.mip_gap": solver.mip_gap,
        "solver.time_limit_seconds": solver.time_limit_seconds,
        "assumptions.alpha_pos": assumptions.alpha_pos,
        "assumptions.alpha_neg": assumptions.alpha_neg,
        "assumptions.activation_energy_price_eur_mwh": assumptions.activation_energy_price_eur_mwh,
        "assumptions.fcr_reserve_hours": assumptions.fcr_reserve_hours,
        "assumptions.afrr_reserve_hours": assumptions.afrr_reserve_hours,
        # Runs solved before v4 lack this key, so they never match and are re-solved.
        "market_prices.bid_granularity_mw.day_ahead": price_basis.granularity_for("day_ahead"),
        "market_prices.bid_granularity_mw.fcr": price_basis.granularity_for("fcr"),
        "market_prices.bid_granularity_mw.afrr_pos": price_basis.granularity_for("afrr_pos"),
        "market_prices.bid_granularity_mw.afrr_neg": price_basis.granularity_for("afrr_neg"),
    }


def _find_already_solved(
    catalogue,
    strategy: str,
    decision_source: str,
    forecast_model: str | None,
    forecast_quantile: float | None,
    start_date: dt.date,
    end_date: dt.date,
    fingerprint: dict,
    da_proxy_model: str | None = None,
    paths_config: PathsConfig | None = None,
) -> str | None:
    """The solve_id of a prior run matching this exact request, or None.
    PLAN_V6_REVISION.md V6-7: a forecast-driven candidate also has to match
    today's forecast_run_ids (per market), not just the config fingerprint
    -- otherwise re-forecasting a model (same name, new backtest) left the
    batch thinking the old solve was still current, since nothing else
    about the request had changed.
    """
    if catalogue.empty:
        return None

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
        return None

    if strategy == "full_sequential":
        fingerprint = {**fingerprint, "da_proxy_model": da_proxy_model if decision_source == "forecast" else None}

    def _matches(row, key, value) -> bool:
        # A None expectation means "no proxy": the sidecar stores null, which
        # the catalogue frame surfaces as NaN.
        return pd.isna(row.get(key)) if value is None else row.get(key) == value

    expected_run_ids: dict[str, str] | None = None
    if decision_source == "forecast" and forecast_model is not None and paths_config is not None:
        try:
            expected_run_ids = load_forecast_run_ids(paths_config, forecast_model)
        except (FileNotFoundError, ValueError):
            expected_run_ids = None  # no forecast results to compare against yet; fingerprint match still applies

    for _, row in candidates.iterrows():
        if not all(_matches(row, key, value) for key, value in fingerprint.items()):
            continue
        if expected_run_ids is not None and not all(
            _matches(row, f"forecast_run_ids.{market}", run_id) for market, run_id in expected_run_ids.items()
        ):
            continue
        return row["solve_id"]
    return None


def _already_solved(
    catalogue,
    strategy: str,
    decision_source: str,
    forecast_model: str | None,
    forecast_quantile: float | None,
    start_date: dt.date,
    end_date: dt.date,
    fingerprint: dict,
    da_proxy_model: str | None = None,
    paths_config: PathsConfig | None = None,
) -> bool:
    return _find_already_solved(
        catalogue, strategy, decision_source, forecast_model, forecast_quantile, start_date, end_date,
        fingerprint, da_proxy_model, paths_config,
    ) is not None


def _restamp_study_if_resumed(results_dir: Path, solve_id: str, study_name: str) -> None:
    """A resumed CEILING ('realized') run doesn't depend on the forecast
    model or study -- the same battery/solver/assumptions/window always
    solves to the same answer, so a later study's batch run legitimately
    reuses it instead of re-solving (PLAN_V6_REVISION.md V6-2's "study" run
    tag, introduced in V5-6, otherwise leaves it looking like it belongs
    only to whichever study first solved it -- invisible to every page
    that filters the catalogue by the CURRENT study). Stamping the latest
    study onto it keeps that filter correct without re-solving anything.
    """
    meta_path = results_dir / solve_id / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("study") != study_name:
        meta["study"] = study_name
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def build_matrix(
    models: list[str], quantiles: list[float], strategies: list[str], include_perfect_foresight: bool,
) -> list[tuple[str, str, str | None, float | None]]:
    """Every (strategy, decision_source, forecast_model, forecast_quantile)
    combination the batch should attempt, in a stable order.
    """
    combos: list[tuple[str, str, str | None, float | None]] = []
    for strategy in strategies:
        if include_perfect_foresight:
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
    sequential_da_proxy: str | None = None,
) -> None:
    battery = battery_from_config(load_battery_config())
    solver = load_solver_config()
    assumptions = load_assumptions_config()
    price_basis = load_market_price_basis_config()
    paths = paths_config or load_paths_config()
    timings = load_market_timing_configs(paths)
    fingerprint = _fingerprint(battery, solver, assumptions, price_basis)

    combos = build_matrix(models, quantiles, strategies, include_perfect_foresight)
    catalogue = scan_runs(results_dir)

    solved, skipped, failed = 0, 0, 0
    for strategy, decision_source, forecast_model, forecast_quantile in combos:
        label = (
            f"{strategy} | {decision_source}"
            + (f" | {forecast_model} q{forecast_quantile}" if decision_source == "forecast" else "")
        )

        resumed_solve_id = None if force else _find_already_solved(
            catalogue, strategy, decision_source, forecast_model, forecast_quantile, start_date, end_date, fingerprint,
            sequential_da_proxy, paths,
        )
        if resumed_solve_id is not None:
            if decision_source == "realized" and not dry_run:
                _restamp_study_if_resumed(results_dir, resumed_solve_id, _STUDY.name)
            print(f"[skip]  {label} (already solved, matching config)")
            skipped += 1
            continue

        if dry_run:
            print(f"[would run] {label}")
            continue

        print(f"[run]   {label} ...", flush=True)
        try:
            result = run_strategy(
                strategy,
                start_date=start_date, end_date=end_date, battery=battery, solver_config=solver,
                assumptions=assumptions, price_basis=price_basis,
                day_ahead_timing=timings["day_ahead"], capacity_timing=timings["fcr"], paths_config=paths,
                decision_source=decision_source, forecast_model=forecast_model, forecast_quantile=forecast_quantile,
                **({"da_proxy_model": sequential_da_proxy} if strategy == "full_sequential" and decision_source == "forecast" else {}),
            )
            write_rolling_run_results(
                result, results_dir,
                strategy=strategy, start_date=start_date, end_date=end_date,
                battery=battery, solver_config=solver, assumptions=assumptions, price_basis=price_basis,
                paths_config=paths, study_name=_STUDY.name,
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
    parser.add_argument("--strategies", nargs="+", choices=ALL_STRATEGIES, default=DEFAULT_STRATEGIES)
    parser.add_argument("--include-perfect-foresight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Re-solve even if a matching run already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print the matrix without solving anything")
    parser.add_argument(
        "--sequential-da-proxy", default=None,
        help="Model whose day-ahead forecast the FCR/aFRR stages of full_sequential see (e.g. seasonal_naive "
             "for the strict, worst-case variant). Default: the run's own model at every stage.",
    )
    args = parser.parse_args(argv)

    run_batch(
        start_date=args.start_date,
        end_date=args.end_date,
        models=args.models,
        quantiles=[MEDIAN_QUANTILE],
        strategies=args.strategies,
        include_perfect_foresight=args.include_perfect_foresight,
        force=args.force,
        dry_run=args.dry_run,
        sequential_da_proxy=args.sequential_da_proxy,
    )


if __name__ == "__main__":
    main()
