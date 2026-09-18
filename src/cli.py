"""CLI entry point (SPEC.md Milestone 6): runs all three benchmark
strategies over a date range and prints total revenue, revenue split by
market, cycles per day, average SOC utilisation, and the assumptions
banner -- plus an explicit call-out if the assumed activation term is a
large share of total revenue, since that means the headline number is
mostly an assumption, not a measurement.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from src.config import (
    CONFIG_DIR,
    load_assumptions_config,
    load_battery_config,
    load_market_price_basis_config,
    load_market_timing_configs,
    load_paths_config,
    load_solver_config,
)
from src.model.battery import Battery, battery_from_config
from src.model.rolling import RollingRunResult
from src.valuation.benchmarks import run_all_benchmarks

ACTIVATION_FRACTION_WARNING_THRESHOLD = 0.2


def format_assumptions_banner() -> str:
    """Prints config/assumptions.yaml verbatim rather than reformatting
    its values, so the banner can never drift out of sync with the actual
    comments explaining what each assumption is and why (SPEC.md Hard
    Rule 2: assumptions must be printed in a banner at the top of every
    CLI run).
    """
    raw_text = (CONFIG_DIR / "assumptions.yaml").read_text(encoding="utf-8")
    border = "=" * 78
    return f"{border}\nASSUMPTIONS IN EFFECT FOR THIS RUN (config/assumptions.yaml)\n{border}\n{raw_text}{border}\n"


def summarize_revenue(result: RollingRunResult) -> dict[str, float]:
    dr = result.daily_revenue
    if dr.empty:
        return {k: 0.0 for k in ("day_ahead", "fcr", "afrr_pos", "afrr_neg", "activation", "total")}
    return {
        "day_ahead": float(dr["day_ahead"].sum()),
        "fcr": float(dr["fcr"].sum()),
        "afrr_pos": float(dr["afrr_pos"].sum()),
        "afrr_neg": float(dr["afrr_neg"].sum()),
        "activation": float(dr["activation"].sum()),
        "total": float(dr["total"].sum()),
    }


def compute_operational_stats(result: RollingRunResult, battery: Battery, delta_t_hours: float) -> dict[str, float]:
    """cycles_per_day: mean daily (energy discharged at the meter) /
    (usable SOC range) -- one full cycle == fully draining the usable
    range once. avg_soc_utilization: mean daily (max(e) - min(e)) /
    (usable SOC range) -- how much of the available headroom actually got
    used, averaged over days.
    """
    usable_range = battery.soc_max_mwh - battery.soc_min_mwh
    lf = result.long_format
    if lf.empty or usable_range <= 0:
        return {"cycles_per_day": 0.0, "avg_soc_utilization": 0.0}

    daily_cycles = []
    daily_utilization = []
    for _, group in lf.groupby("solve_date"):
        discharged_mwh = (group.loc[group["variable"] == "p_dis", "value"] * delta_t_hours).sum()
        daily_cycles.append(discharged_mwh / usable_range)

        e_values = group.loc[group["variable"] == "e", "value"]
        if not e_values.empty:
            daily_utilization.append((e_values.max() - e_values.min()) / usable_range)

    return {
        "cycles_per_day": sum(daily_cycles) / len(daily_cycles) if daily_cycles else 0.0,
        "avg_soc_utilization": sum(daily_utilization) / len(daily_utilization) if daily_utilization else 0.0,
    }


def print_summary(name: str, result: RollingRunResult, battery: Battery, delta_t_hours: float) -> None:
    revenue = summarize_revenue(result)
    stats = compute_operational_stats(result, battery, delta_t_hours)

    print(f"--- {name} ---")
    print(f"  days solved: {len(result.daily_revenue)}  |  days skipped: {len(result.skip_log)}")
    print(f"  total revenue: EUR {revenue['total']:.2f}")
    print(
        "    split -> day_ahead: {day_ahead:.2f}  fcr: {fcr:.2f}  afrr_pos: {afrr_pos:.2f}  "
        "afrr_neg: {afrr_neg:.2f}  activation: {activation:.2f}".format(**revenue)
    )
    print(
        f"  cycles/day: {stats['cycles_per_day']:.2f}  "
        f"avg SOC utilisation: {stats['avg_soc_utilization'] * 100:.1f}%"
    )
    if revenue["total"] != 0:
        activation_fraction = revenue["activation"] / revenue["total"]
        if abs(activation_fraction) > ACTIVATION_FRACTION_WARNING_THRESHOLD:
            print(
                f"  !! {activation_fraction * 100:.1f}% of total revenue comes from the ASSUMED "
                "activation term -- this headline number is largely an assumption, not a measurement."
            )
    print()


def main(argv: list[str] | None = None) -> None:
    # config/assumptions.yaml's comments use non-ASCII punctuation (em
    # dashes); Windows consoles often default to a codepage that can't
    # render them, garbling Hard Rule 2's assumptions banner. Force UTF-8
    # on stdout so the banner is always legible regardless of console.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="BESS multi-market optimizer: benchmark comparison")
    parser.add_argument("--start-date", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=dt.date.fromisoformat, required=True)
    args = parser.parse_args(argv)

    battery = battery_from_config(load_battery_config())
    solver = load_solver_config()
    assumptions = load_assumptions_config()
    price_basis = load_market_price_basis_config()
    paths = load_paths_config()
    timings = load_market_timing_configs(paths)

    print(format_assumptions_banner())
    print(f"Run window: {args.start_date} to {args.end_date}\n")

    results = run_all_benchmarks(
        args.start_date, args.end_date, battery, solver, assumptions, price_basis,
        timings["day_ahead"], timings["fcr"], paths_config=paths,
    )
    delta_t_hours = timings["day_ahead"].product_length_hours
    for name, result in results.items():
        print_summary(name, result, battery, delta_t_hours)


if __name__ == "__main__":
    main()
