"""Why does the forecast-driven day-ahead leg lose money?

For each forecast model and strategy (day_ahead_only and full_cooptimized),
over one solved date window, compares the forecast-driven run with its
perfect-foresight run on the days they share:

  - day-ahead revenue, degradation cost, and the two together;
  - traded volume and the REALIZED average buy/sell price of the schedule
    (a schedule that buys and sells at the right hours has sell > buy);
  - how good the day-ahead forecast's shape is (correlation of the
    intraday profile with the realized one) and its level bias.

    python scripts/da_leg_diagnosis.py --start-date 2026-03-21 --end-date 2026-09-17
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analysis import forecast_shape_quality, realized_prices  # noqa: E402
from src.config import REPO_ROOT, load_paths_config  # noqa: E402
from src.data.forecast_loader import load_raw_y_pred, load_raw_y_true  # noqa: E402
from src.results_catalogue import scan_runs  # noqa: E402

STRATEGIES = ("day_ahead_only", "full_cooptimized")


def _pick(catalogue: pd.DataFrame, strategy: str, source: str, model, window) -> pd.Series | None:
    mask = (
        (catalogue["strategy"] == strategy) & (catalogue["decision_source"] == source)
        & (catalogue["start_date"] == window[0]) & (catalogue["end_date"] == window[1])
        & catalogue["battery.power_nominal_mw"].notna()  # current (v2) battery only
    )
    mask &= catalogue["forecast_model"].isna() if model is None else catalogue["forecast_model"] == model
    rows = catalogue[mask].sort_values("created_at", ascending=False)
    return None if rows.empty else rows.iloc[0]


def _daily(results_dir: Path, solve_id: str) -> pd.DataFrame:
    df = pd.read_parquet(results_dir / solve_id / "daily_revenue.parquet")
    df["solve_date"] = pd.to_datetime(df["solve_date"]).dt.date
    return df.set_index("solve_date")


def _realized_prices(results_dir: Path, solve_id: str, y_true_da: pd.Series) -> tuple[float, float]:
    lf = pd.read_parquet(results_dir / solve_id / "long_format.parquet", columns=["variable", "target_time", "value"])
    return realized_prices(lf, y_true_da)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = parser.parse_args(argv)
    window = (args.start_date, args.end_date)
    results_dir = Path(args.results_dir)

    catalogue = scan_runs(results_dir)
    paths = load_paths_config()
    y_true = load_raw_y_true(paths).loc["day_ahead"]
    pd.set_option("display.width", 200)

    models = sorted(m for m in catalogue.loc[catalogue["decision_source"] == "forecast", "forecast_model"].dropna().unique())
    for strategy in STRATEGIES:
        perfect = _pick(catalogue, strategy, "realized", None, window)
        if perfect is None:
            print(f"\n[{strategy}] no perfect-foresight run for {window}; skipped")
            continue
        perfect_daily = _daily(results_dir, perfect["solve_id"])
        buy_p, sell_p = _realized_prices(results_dir, perfect["solve_id"], y_true)
        rows = []
        for model in models:
            run = _pick(catalogue, strategy, "forecast", model, window)
            if run is None or run["forecast_quantile"] != args.quantile:
                continue
            daily = _daily(results_dir, run["solve_id"])
            days = sorted(set(daily.index) & set(perfect_daily.index))
            f, p = daily.loc[days], perfect_daily.loc[days]
            buy, sell = _realized_prices(results_dir, run["solve_id"], y_true)
            shape = forecast_shape_quality(load_raw_y_pred(paths, model, args.quantile).loc["day_ahead"], y_true, days)
            rows.append(
                {
                    "model": model, "days": len(days),
                    "DA_eur": f["day_ahead"].sum(), "DA_perfect": p["day_ahead"].sum(),
                    "degr_eur": f["degradation"].sum(), "degr_perfect": p["degradation"].sum(),
                    "DA_minus_degr": f["day_ahead"].sum() - f["degradation"].sum(),
                    "DA_minus_degr_perfect": p["day_ahead"].sum() - p["degradation"].sum(),
                    "MWh_out": f["discharged_mwh"].sum(), "MWh_out_perfect": p["discharged_mwh"].sum(),
                    "real_buy": buy, "real_sell": sell, "margin": sell - buy,
                    **shape,
                }
            )
        print(f"\n===== {strategy} | window {window[0]}..{window[1]} | q{args.quantile} =====")
        print(f"perfect foresight schedule: realized buy {buy_p:.1f}, sell {sell_p:.1f}, margin {sell_p - buy_p:.1f} EUR/MWh")
        if rows:
            print(pd.DataFrame(rows).set_index("model").round(2).T.to_string())


if __name__ == "__main__":
    main()
