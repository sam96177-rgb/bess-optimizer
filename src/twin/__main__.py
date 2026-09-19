"""`python -m src.twin --solve-id <id>`: replay a stored run through the
digital twin and write twin_summary.parquet / twin_paths.parquet /
twin_meta.json next to the run's own files."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from src.config import REPO_ROOT, load_market_timing_configs, load_paths_config
from src.data.grid import build_day_grid
from src.twin.plot import plot_day
from src.twin.replay import (
    MODES, TwinConfig, battery_from_meta, replay_run, write_twin_outputs,
)

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Replay a solved run through the digital twin")
    parser.add_argument("--solve-id", required=True)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--n-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=MODES, default="stochastic")
    parser.add_argument("--call-degree-mean", type=float, default=TwinConfig.call_degree_mean,
                        help="PLACEHOLDER: mean fraction of committed capacity a called step delivers (1 = always full).")
    parser.add_argument("--call-degree-concentration", type=float, default=TwinConfig.call_degree_concentration)
    parser.add_argument("--plot-day", type=dt.date.fromisoformat, default=None, help="Also save a SOC plot for this day.")
    args = parser.parse_args(argv)

    run_dir = Path(args.results_dir) / args.solve_id
    if not (run_dir / "run_meta.json").exists():
        parser.error(f"No completed run at {run_dir}")

    timings = load_market_timing_configs(load_paths_config())
    cfg = TwinConfig(
        n_draws=args.n_draws, seed=args.seed, mode=args.mode,
        call_degree_mean=args.call_degree_mean, call_degree_concentration=args.call_degree_concentration,
    )
    summary, paths, twin_meta = replay_run(
        run_dir, lambda day: build_day_grid(day, timings["day_ahead"], timings["fcr"]), cfg,
    )
    write_twin_outputs(run_dir, summary, paths, twin_meta)

    print("=" * 78)
    print("DIGITAL TWIN -- aFRR activation is a PLACEHOLDER assumption "
          f"(alpha_pos={twin_meta['alpha_pos']}, alpha_neg={twin_meta['alpha_neg']}, "
          f"call size mean={cfg.call_degree_mean})")
    print("=" * 78)
    print(f"run {args.solve_id} | mode={cfg.mode} | {twin_meta['n_days']} days x {cfg.n_draws} draws | seed={cfg.seed}")
    if summary.empty:
        print("no solved days to replay")
        return
    any_violation = (summary["n_soc_violation_steps"] > 0) | (summary["n_power_violation_steps"] > 0)
    per_day = summary.groupby("solve_date").agg(
        shortfall=("dispatch_shortfall_mwh", "mean"),
        act_short_up=("activation_shortfall_up_mwh", "mean"),
        act_short_down=("activation_shortfall_down_mwh", "mean"),
    )
    per_day["total_shortfall"] = per_day.sum(axis=1)
    print(f"share of (day, draw) replays with any violation: {any_violation.mean() * 100:.1f}%")
    print(f"mean planned-dispatch/aux shortfall per day: {summary['dispatch_shortfall_mwh'].mean():.3f} MWh")
    print(f"mean activation shortfall per day: up {summary['activation_shortfall_up_mwh'].mean():.3f} MWh, "
          f"down {summary['activation_shortfall_down_mwh'].mean():.3f} MWh")
    print(f"activation delivered vs planned expectation: {summary['activation_delivered_mwh'].sum():.1f} vs "
          f"{summary['activation_expected_mwh'].sum():.1f} MWh (summed over all replays)")
    print(f"worst day (mean shortfall): {per_day['total_shortfall'].idxmax()} "
          f"({per_day['total_shortfall'].max():.3f} MWh)")
    print(f"max power-limit violation steps in any replay: {int(summary['n_power_violation_steps'].max())}")
    print(f"wrote {run_dir}/twin_summary.parquet, twin_paths.parquet, twin_meta.json")

    if args.plot_day is not None:
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        out = plot_day(paths, args.plot_day, battery_from_meta(meta), run_dir / f"twin_soc_{args.plot_day}.png")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
