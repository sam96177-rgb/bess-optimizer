"""Forecast Comparison (PLAN_V6_REVISION.md V6-6): replaces forecast_error_cost.py,
oracle_capture.py and forecast_accuracy.py with one page per the three-level
story -- baseline (no forecasting model), with forecasting, and the ceiling
(the optimizer planning on the real prices, not achievable) -- and the single
capture definition from V6-1 (realized net over the ceiling's net).
"""
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (_APP_DIR, _APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import ui
from analysis import (
    BASELINE_MODEL, CAPTURE_ROWS, common_days, forecast_error_metrics, forecast_shape_quality, model_revenue_table, plan_agreement,
    oracle_capture, realized_prices,
)
from data_loader import load_daily_revenue, load_long_format, load_median_forecasts, load_run_catalogue, matching_runs

st.title("Forecast Comparison")

catalogue = ui.study_filter(load_run_catalogue())
if catalogue.empty:
    st.warning("No runs found in `results/`. Run `python -m src.batch` first.")
    st.stop()

perfect_rows = catalogue[catalogue["decision_source"] == "realized"]
forecast_rows = catalogue[catalogue["decision_source"] == "forecast"]
strategies = sorted(set(perfect_rows["strategy"]) & set(forecast_rows["strategy"]))
if not strategies:
    st.warning("No strategy has both a ceiling run and a forecast-driven run yet. Run `python -m src.batch`.")
    st.stop()

windows = sorted(set(zip(perfect_rows["start_date"], perfect_rows["end_date"])))
window = st.sidebar.selectbox("Date window", windows, format_func=lambda w: f"{w[0]} to {w[1]}")
start, end = dt.date.fromisoformat(window[0]), dt.date.fromisoformat(window[1])


def _in_window(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[(rows["start_date"] == window[0]) & (rows["end_date"] == window[1])]


models = sorted(forecast_rows["forecast_model"].dropna().unique())
if not models:
    st.warning("No forecast-driven runs found yet.")
    st.stop()

_LEVEL = {BASELINE_MODEL: "Baseline (no forecasting model)"}
_level_of = lambda model: _LEVEL.get(model, "With forecasting")  # noqa: E731

st.sidebar.caption(
    f"Levels: **{BASELINE_MODEL}** is the baseline (no forecasting model); every other model is "
    "\"with forecasting\"; the ceiling (ran on real prices) is reference only -- not achievable."
)
ui.model_picker(models, key="forecast_comparison_model_picker")

forecasts_all = load_median_forecasts()

for strategy in strategies:
    st.header(strategy)

    perfect_match = _in_window(matching_runs(catalogue, strategy, "realized"))
    if perfect_match.empty:
        st.info(f"No ceiling run for `{strategy}` in this window.")
        continue
    perfect_daily = load_daily_revenue(perfect_match.iloc[0]["solve_id"])

    model_daily: dict[str, pd.DataFrame] = {}
    model_long: dict[str, pd.DataFrame] = {}
    for model in models:
        match = _in_window(matching_runs(catalogue, strategy, "forecast", model))
        if not match.empty:
            model_daily[model] = load_daily_revenue(match.iloc[0]["solve_id"])
            model_long[model] = load_long_format(match.iloc[0]["solve_id"])
    if not model_daily:
        st.info(f"No forecast-driven run for `{strategy}` in this window.")
        continue

    revenue, perfect_net, n_days = model_revenue_table(
        perfect_daily, model_daily, (load_long_format(perfect_match.iloc[0]["solve_id"]), model_long)
    )

    forecasts = forecasts_all
    if not forecasts.empty:
        local_day = forecasts["target_time"].dt.date
        forecasts = forecasts[forecasts["model"].isin(model_daily) & (local_day >= start) & (local_day <= end)]
    errors = forecast_error_metrics(forecasts) if not forecasts.empty else forecast_error_metrics(
        pd.DataFrame(columns=["market", "model", "target_time", "y_pred", "y_true"])
    )
    day_ahead_mae = (
        errors.xs("day_ahead", level="market")["mae"] if "day_ahead" in errors.index.get_level_values("market")
        else pd.Series(dtype=float)
    )

    table = revenue.copy()
    table.insert(0, "level", [_level_of(m) for m in table.index])
    table["day_ahead_mae"] = day_ahead_mae.reindex(table.index)

    st.caption(
        f"{n_days} day(s) common to the ceiling and every model shown | ceiling net EUR {perfect_net:,.0f}. "
        f"{ui.AGREEMENT_CAPTION}"
    )
    highlight = table["charge_timing_pct"].idxmax() if table["charge_timing_pct"].notna().any() else None
    ui.styled_table(
        table,
        formats={**ui.AGREEMENT_FORMATS, "day_ahead_mae": "{:.2f}"},
        highlight_row=highlight,
        align_right=[*ui.AGREEMENT_FORMATS, "day_ahead_mae"],
    )

    with st.expander("Capture per market"):
        per_market = pd.DataFrame(index=list(model_daily), columns=CAPTURE_ROWS, dtype=float)
        for model, daily in model_daily.items():
            detail, _ = oracle_capture(perfect_daily, daily)
            per_market.loc[model] = detail["capture_pct"]
        ui.styled_table(per_market, formats={c: "{:.1f}%" for c in CAPTURE_ROWS})
        st.caption(
            "Per-market capture can exceed 100% or go negative: the markets share one battery, so a "
            "forecast-driven run can over-sell one market while under-trading another."
        )

    if not errors.empty and revenue["net_eur"].notna().any():
        markets = sorted(errors.index.get_level_values("market").unique())
        market = st.selectbox(
            "Market for the error axis", markets, index=markets.index("day_ahead") if "day_ahead" in markets else 0,
            key=f"scatter_market_{strategy}",
        )
        points = pd.DataFrame({"mae": errors.xs(market, level="market")["mae"], "net": revenue["net_eur"]}).dropna()
        if not points.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.scatter(points["mae"], points["net"], color="tab:blue")
            for model, row in points.iterrows():
                ax.annotate(model, (row["mae"], row["net"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
            ax.axhline(perfect_net, color="grey", linestyle="--", linewidth=1, label="ceiling (real prices)")
            ax.set_xlabel(f"{market} MAE (lower = more accurate)")
            ax.set_ylabel("net revenue, realized (EUR)")
            ax.set_title("Does a lower forecast error mean more money?")
            ax.legend(loc="lower left", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)

    da_forecasts = forecasts[forecasts["market"] == "day_ahead"] if not forecasts.empty else forecasts
    y_true_da = (
        da_forecasts.drop_duplicates("target_time").set_index("target_time")["y_true"].sort_index()
        if not da_forecasts.empty else pd.Series(dtype=float)
    )
    if not y_true_da.empty:
        perfect_lf = load_long_format(perfect_match.iloc[0]["solve_id"])
        perfect_buy, perfect_sell = realized_prices(perfect_lf, y_true_da)
        diag_rows = {}
        for model in model_daily:
            match = _in_window(matching_runs(catalogue, strategy, "forecast", model))
            if match.empty:
                continue
            lf = load_long_format(match.iloc[0]["solve_id"])
            buy, sell = realized_prices(lf, y_true_da)
            days = common_days(perfect_daily, model_daily[model])
            y_pred_da = da_forecasts.loc[da_forecasts["model"] == model].drop_duplicates("target_time").set_index("target_time")["y_pred"].sort_index()
            shape = forecast_shape_quality(y_pred_da, y_true_da, days)
            exact = plan_agreement(lf, perfect_lf, common_days(perfect_daily, *model_daily.values()), tolerance_quarters=0)
            diag_rows[model] = {
                "realized_buy": buy, "realized_sell": sell, "margin": sell - buy, **shape,
                "charge_exact_pct": exact["charge_timing_pct"], "discharge_exact_pct": exact["discharge_timing_pct"],
            }
        if diag_rows:
            diag = pd.DataFrame.from_dict(diag_rows, orient="index")
            diag.loc["ceiling (real prices)"] = {
                "realized_buy": perfect_buy, "realized_sell": perfect_sell, "margin": perfect_sell - perfect_buy,
                "shape_corr": 1.0, "bias_eur_mwh": 0.0, "peak_timing_gap_h": 0.0,
                "charge_exact_pct": 100.0, "discharge_exact_pct": 100.0,
            }
            with st.expander("Why accuracy matters: shape, timing, and the price it traded at"):
                ui.styled_table(
                    diag.rename(columns={
                        "realized_buy": "realized buy (EUR/MWh)", "realized_sell": "realized sell (EUR/MWh)",
                        "margin": "margin (EUR/MWh)", "shape_corr": "shape correlation",
                        "bias_eur_mwh": "level bias (EUR/MWh)", "peak_timing_gap_h": "peak timing gap (h)",
                        "charge_exact_pct": "charge in the exact quarter-hour", "discharge_exact_pct": "discharge in the exact quarter-hour",
                    }),
                    formats={c: "{:.2f}" for c in ["realized buy (EUR/MWh)", "realized sell (EUR/MWh)", "margin (EUR/MWh)",
                                                    "shape correlation", "level bias (EUR/MWh)", "peak timing gap (h)"]}
                    | {c: "{:.1f}%" for c in ["charge in the exact quarter-hour", "discharge in the exact quarter-hour"]},
                )
                st.caption(
                    "Shape correlation compares each day's mean-centred intraday profile with the realized one; "
                    "peak timing gap is the median distance between the forecast's and the real day's peak hour. "
                    "The last two columns are the headline timing score with no tolerance: the share of the model's "
                    "active quarter-hours where the ceiling is active in that very quarter-hour."
                )
