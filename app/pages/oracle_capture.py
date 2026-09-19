"""Oracle Capture: how much of what perfect foresight earned each forecast
model captured, per revenue component, for one strategy over one date window.

Reported per market on purpose. Most of a reserve-heavy strategy's revenue
is capacity payments (and a chunk is the ASSUMED activation term), so the
headline net capture looks high for every model; the day-ahead leg is where
forecast quality actually shows and would be hidden inside a single total.
"""
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (_APP_DIR, _APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import streamlit as st

from analysis import CAPTURE_ROWS, oracle_capture, proxy_label
from data_loader import filter_proxy, load_daily_revenue, load_run_catalogue, matching_runs

st.title("Oracle Capture")

catalogue = load_run_catalogue()
if catalogue.empty:
    st.warning("No runs found in `results/`. Run `python -m src.batch` first.")
    st.stop()

perfect_rows = catalogue[catalogue["decision_source"] == "realized"]
forecast_rows = catalogue[catalogue["decision_source"] == "forecast"]
if perfect_rows.empty or forecast_rows.empty:
    st.warning("Need at least one perfect-foresight and one forecast-driven run. Run `python -m src.batch`.")
    st.stop()

strategies = sorted(set(perfect_rows["strategy"]) & set(forecast_rows["strategy"]))
if not strategies:
    st.warning("No strategy has both a perfect-foresight and a forecast-driven run yet.")
    st.stop()
strategy = st.sidebar.selectbox("Strategy", strategies, index=strategies.index("full_cooptimized") if "full_cooptimized" in strategies else 0)

windows = sorted(set(zip(perfect_rows.loc[perfect_rows["strategy"] == strategy, "start_date"],
                         perfect_rows.loc[perfect_rows["strategy"] == strategy, "end_date"])))
window = st.sidebar.selectbox("Date window", windows, format_func=lambda w: f"{w[0]} to {w[1]}")
quantiles = sorted(forecast_rows["forecast_quantile"].dropna().unique())
quantile = st.sidebar.selectbox("Quantile", quantiles, index=quantiles.index(0.5) if 0.5 in quantiles else 0)

proxy = None
if strategy == "full_sequential":
    labels = sorted({proxy_label(v) for v in forecast_rows.loc[forecast_rows["strategy"] == strategy, "da_proxy_model"]}) \
        if "da_proxy_model" in forecast_rows.columns else [proxy_label(None)]
    proxy = st.sidebar.selectbox("Reserve-stage day-ahead view", labels)


def _one(rows: pd.DataFrame):
    rows = rows[(rows["start_date"] == window[0]) & (rows["end_date"] == window[1])]
    return None if rows.empty else rows.iloc[0]


perfect_run = _one(matching_runs(catalogue, strategy, "realized"))
if perfect_run is None:
    st.info("No perfect-foresight run for this strategy and window.")
    st.stop()
perfect_daily = load_daily_revenue(perfect_run["solve_id"])

models = sorted(forecast_rows["forecast_model"].dropna().unique())
capture = pd.DataFrame(index=models, columns=CAPTURE_ROWS, dtype=float)
detail: dict[str, tuple[pd.DataFrame, int]] = {}
for model in models:
    runs = matching_runs(catalogue, strategy, "forecast", model, quantile)
    if proxy is not None:
        runs = filter_proxy(runs, proxy)
    run = _one(runs)
    if run is None:
        continue
    table, n_days = oracle_capture(perfect_daily, load_daily_revenue(run["solve_id"]))
    detail[model] = (table, n_days)
    capture.loc[model] = table["capture_pct"]

st.caption(
    f"{strategy} | window {window[0]} to {window[1]} | quantile {quantile}"
    + (f" | reserve-stage view: {proxy}" if proxy else "")
    + " | capture = forecast-driven / perfect-foresight, both settled on realized prices, "
    "summed over the days both runs solved. `activation` uses an ASSUMED activation rate and price."
)

st.subheader("Capture % by market (higher = closer to perfect foresight)")
if capture.isna().all(axis=None):
    st.info("No forecast-driven run for this selection yet.")
else:
    st.dataframe(
        capture.style.background_gradient(cmap="RdYlGn", axis=None, vmin=0, vmax=100).format("{:.1f}%", na_rep="n/a"),
        width="stretch",
    )
    st.caption(
        "n/a = perfect foresight earned nothing (or lost money) in that market, so a share is undefined. "
        "Per-market capture can exceed 100% or go negative: the markets share one battery, so a forecast-driven "
        "run can over-sell one market while under-trading another. Only the net column is bounded by 100%."
    )

    st.subheader("Detail for one model (EUR)")
    picked = st.selectbox("Model", list(detail))
    table, n_days = detail[picked]
    st.caption(f"{n_days} day(s) common to both runs.")
    st.dataframe(
        table.style.format({"perfect_eur": "{:,.0f}", "forecast_eur": "{:,.0f}", "gap_eur": "{:,.0f}", "capture_pct": "{:.1f}%"}, na_rep="n/a"),
        width="stretch",
    )
