"""Forecast Error Cost: models x strategies matrix of
perfect-foresight total - forecast-driven total, both settled on y_true
over the same day set (SPEC.md §6a). Day sets are intersected before
differencing -- seasonal_naive skips 2 extra days (2026-03-29, 04-05) that
every other model solves, so raw totals aren't directly comparable (see
IMPLEMENTATION_PLAN.md's Risks section) -- and the number of dropped days
is shown alongside the cost so a large intersection gap isn't hidden.
"""
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (_APP_DIR, _APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import streamlit as st

from data_loader import load_daily_revenue, load_run_catalogue, matching_runs

st.title("Forecast Error Cost")

catalogue = load_run_catalogue()
if catalogue.empty:
    st.warning("No runs found in `results/`. Run `python -m src.batch` first.")
    st.stop()

perfect_rows = catalogue[catalogue["decision_source"] == "realized"]
forecast_rows = catalogue[catalogue["decision_source"] == "forecast"]
if perfect_rows.empty or forecast_rows.empty:
    st.warning(
        "Need at least one perfect-foresight run and one forecast-driven run to compute a "
        "cost of forecast error. Run `python -m src.batch` to solve the full matrix."
    )
    st.stop()

windows = sorted(set(zip(perfect_rows["start_date"], perfect_rows["end_date"])), key=lambda w: (w[0], w[1]))
window = st.sidebar.selectbox("Date window", windows, format_func=lambda w: f"{w[0]} to {w[1]}")
start_date, end_date = window

quantiles = sorted(forecast_rows["forecast_quantile"].dropna().unique())
default_idx = quantiles.index(0.5) if 0.5 in quantiles else 0
quantile = st.sidebar.selectbox("Quantile", quantiles, index=default_idx)

models = sorted(forecast_rows["forecast_model"].dropna().unique())
strategies = sorted(catalogue["strategy"].unique())

cost = pd.DataFrame(index=models, columns=strategies, dtype=float)
dropped = pd.DataFrame(index=models, columns=strategies, dtype=float)

for strategy in strategies:
    perfect_match = matching_runs(catalogue, strategy, "realized")
    perfect_match = perfect_match[(perfect_match["start_date"] == start_date) & (perfect_match["end_date"] == end_date)]
    if perfect_match.empty:
        continue
    perfect_daily = load_daily_revenue(perfect_match.iloc[0]["solve_id"])
    perfect_by_day = perfect_daily.set_index("solve_date")["total"]
    perfect_days = set(perfect_by_day.index)

    for model in models:
        forecast_match = matching_runs(catalogue, strategy, "forecast", model, quantile)
        forecast_match = forecast_match[
            (forecast_match["start_date"] == start_date) & (forecast_match["end_date"] == end_date)
        ]
        if forecast_match.empty:
            continue
        forecast_daily = load_daily_revenue(forecast_match.iloc[0]["solve_id"])
        forecast_by_day = forecast_daily.set_index("solve_date")["total"]
        forecast_days = set(forecast_by_day.index)

        common_days = perfect_days & forecast_days
        dropped.loc[model, strategy] = len(perfect_days ^ forecast_days)
        if not common_days:
            continue
        common_days = sorted(common_days)
        cost.loc[model, strategy] = perfect_by_day.loc[common_days].sum() - forecast_by_day.loc[common_days].sum()

st.caption(
    f"Window: {start_date} to {end_date} | quantile {quantile} | "
    "cost = perfect-foresight total - forecast-driven total, both summed over the intersected day set."
)

st.subheader("Cost of forecast error (EUR, higher = forecast error cost more)")
if cost.isna().all(axis=None):
    st.info("No (model, strategy) pair has both a perfect-foresight and a forecast-driven run for this window/quantile yet.")
else:
    styled = cost.style.background_gradient(cmap="RdYlGn_r", axis=None).format("{:,.2f}", na_rep="—")
    st.dataframe(styled, width="stretch")

st.subheader("Dropped days (in exactly one of the two runs, excluded from the cost above)")
st.dataframe(dropped.style.format("{:.0f}", na_rep="—"), width="stretch")
