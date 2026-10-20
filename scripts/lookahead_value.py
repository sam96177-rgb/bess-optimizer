"""PLAN_V4_REVISION.md section 5c / PLAN_V5_REVISION.md V5-1: is the missing
cross-midnight lookahead worth building? Compares three day-ahead-only
configurations over the same realized (perfect-foresight) prices -- no new
forecasts needed, and with ancillary=None there are no capacity blocks, so
the block-index namespacing a real multi-day model would need never arises
here:

  1. Baseline           -- independent cyclic days (the existing
                            day_ahead_only, decision_source=realized run in
                            results/).
  2. Chained             -- cyclic_soc=False, SOC carried across midnight:
                            continuity, no lookahead.
  3. today + 12h lookahead -- one model per day over today plus the next
                            LOOKAHEAD_HOURS of tomorrow, solved via the same
                            subprocess-isolated path run_rolling uses
                            (rolling._solve_day_via_subprocess), committing
                            only today before rolling forward one day.

V4 used a 48h (full next day) window; its ~3h runtime wasn't worth it for
the value it added, so V5 narrowed it to LOOKAHEAD_HOURS: at the noon D-1
gate you already know day D's prices, so the real unknown horizon is only
the ~12h from midnight into the following day.

    python scripts/lookahead_value.py --start-date 2026-03-21 --end-date 2026-09-17
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    REPO_ROOT,
    SolverConfig,
    load_battery_config,
    load_market_timing_configs,
    load_paths_config,
    load_solver_config,
    load_study_config,
)
from src.data.forecast_loader import DayPrices, load_price_range  # noqa: E402
from src.data.grid import DayGrid  # noqa: E402
from src.model.battery import Battery, battery_from_config  # noqa: E402
from src.model.rolling import _solve_day_via_subprocess, run_rolling  # noqa: E402
from src.model.solve import SolveResult  # noqa: E402
from src.results_catalogue import scan_runs  # noqa: E402
from src.valuation.revenue import compute_revenue  # noqa: E402

EMPTY_BLOCK_PRICES = pd.Series(dtype=float)
LOOKAHEAD_HOURS = 12  # noon D-1 gate already knows day D; ~12h past midnight is the real unknown horizon


def _baseline_net_eur(results_dir: Path, start: dt.date, end: dt.date) -> float | None:
    catalogue = scan_runs(results_dir)
    if catalogue.empty:
        return None
    mask = (
        (catalogue["strategy"] == "day_ahead_only")
        & (catalogue["decision_source"] == "realized")
        & (catalogue["start_date"] == start.isoformat())
        & (catalogue["end_date"] == end.isoformat())
    )
    rows = catalogue[mask].sort_values("created_at", ascending=False)
    if rows.empty:
        return None
    daily = pd.read_parquet(results_dir / rows.iloc[0]["solve_id"] / "daily_revenue.parquet")
    return float(daily["net"].sum())


def _chained_net_eur(start, end, battery, solver, day_ahead_timing, capacity_timing) -> float:
    result = run_rolling(
        start, end, battery, solver, day_ahead_timing, capacity_timing,
        cyclic_soc=False, include_ancillary=False, decision_source="realized",
    )
    return float(result.daily_revenue["net"].sum())


def _lookahead_window_net_eur(
    start: dt.date, end: dt.date, battery: Battery, solver: SolverConfig, day_ahead_timing, capacity_timing, paths,
) -> float:
    days, skipped = load_price_range(start, end, day_ahead_timing, capacity_timing, paths)
    if skipped:
        print(f"[lookahead window] skipping {len(skipped)} day(s) with missing prices: {[s.day for s in skipped[:5]]}")
    by_day = {d.day: d for d in days}
    ordered = sorted(by_day)

    net_eur = 0.0
    carried_soc_mwh: float | None = None
    for i, day in enumerate(ordered):
        today = by_day[day]
        tomorrow = by_day.get(ordered[i + 1]) if i + 1 < len(ordered) and ordered[i + 1] == day + dt.timedelta(days=1) else None
        initial_soc = carried_soc_mwh if carried_soc_mwh is not None else battery.soc_min_mwh

        if tomorrow is None:
            window = today
        else:
            # Absolute-time arithmetic (src/data/grid.py convention), not
            # wall-clock, so this slices correctly across a DST day too.
            lookahead_cutoff = tomorrow.grid.steps[0] + pd.Timedelta(hours=LOOKAHEAD_HOURS)
            lookahead_steps = tomorrow.grid.steps[tomorrow.grid.steps < lookahead_cutoff]
            window = DayPrices(
                day=today.day,
                grid=DayGrid(
                    day=today.day,
                    steps=today.grid.steps.append(lookahead_steps),
                    blocks=[],
                    step_block_index=[],
                    delta_t_hours=today.grid.delta_t_hours,
                ),
                day_ahead=pd.concat([today.day_ahead, tomorrow.day_ahead.loc[lookahead_steps]]),
                fcr=EMPTY_BLOCK_PRICES, afrr_pos=EMPTY_BLOCK_PRICES, afrr_neg=EMPTY_BLOCK_PRICES,
            )

        full_result = _solve_day_via_subprocess(
            window, battery, solver, cyclic_soc=False, enforce_exclusivity=True,
            initial_soc_mwh=initial_soc, ancillary=None,
        )
        committed = SolveResult(
            objective_value=full_result.objective_value,
            p_ch=full_result.p_ch.loc[today.grid.steps],
            p_dis=full_result.p_dis.loc[today.grid.steps],
            u=full_result.u.loc[today.grid.steps],
            e=full_result.e.loc[today.grid.steps],
            c_fcr=None, c_pos=None, c_neg=None,
        )
        revenue = compute_revenue(committed, today.day_ahead, today.grid.delta_t_hours, battery=battery)
        net_eur += revenue.net
        carried_soc_mwh = float(committed.e.iloc[-1])

    return net_eur


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-date", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument(
        "--skip-chained", action="store_true",
        help="Skip config 2 (chained, no lookahead) -- its value doesn't depend on the lookahead window length",
    )
    args = parser.parse_args(argv)

    battery = battery_from_config(load_battery_config())
    solver = load_solver_config()
    paths = load_paths_config()
    timings = load_market_timing_configs(paths)
    results_dir = Path(args.results_dir)

    baseline = _baseline_net_eur(results_dir, args.start_date, args.end_date)
    if args.skip_chained:
        chained = None
        print("chained: skipped")
    else:
        chained = _chained_net_eur(args.start_date, args.end_date, battery, solver, timings["day_ahead"], timings["fcr"])
    windowed = _lookahead_window_net_eur(
        args.start_date, args.end_date, battery, solver, timings["day_ahead"], timings["fcr"], paths,
    )

    print(f"\n===== Today + {LOOKAHEAD_HOURS}h lookahead value | {args.start_date}..{args.end_date} =====")
    print(f"1. Baseline (independent cyclic days):  {baseline:.2f} EUR" if baseline is not None else "1. Baseline: no matching run in results/")
    print(f"2. Chained (SOC carried, no lookahead):  {chained:.2f} EUR" if chained is not None else "2. Chained: skipped")
    print(f"3. today + {LOOKAHEAD_HOURS}h lookahead (SOC carried + lookahead): {windowed:.2f} EUR")
    if baseline is not None and chained is not None:
        print(f"\nContinuity alone: {chained - baseline:+.2f} EUR")
        print(f"Continuity + lookahead: {windowed - baseline:+.2f} EUR")
        print(f"Lookahead's own contribution (3 - 2): {windowed - chained:+.2f} EUR")
    elif baseline is not None:
        print(f"\nContinuity + lookahead: {windowed - baseline:+.2f} EUR")

    # PLAN_V5_1_REVISION.md section 3: output-only, so the dashboard's
    # overview page can show these numbers instead of "not run yet" -- no
    # change to the calculation above.
    study = load_study_config()
    out_path = results_dir / f"lookahead_{study.name}.json"
    out_path.write_text(
        json.dumps(
            {
                "study": study.name,
                "start_date": args.start_date.isoformat(),
                "end_date": args.end_date.isoformat(),
                "lookahead_hours": LOOKAHEAD_HOURS,
                "baseline_net_eur": baseline,
                "chained_net_eur": chained,
                "windowed_net_eur": windowed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
