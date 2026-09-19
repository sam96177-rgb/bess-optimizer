"""Sequential vs Simultaneous: net revenue of perfect foresight, simultaneous
co-optimization (`full_cooptimized`) and gate-ordered clearing
(`full_sequential`: FCR -> aFRR -> DA) on the same days, plus the reserve
commitments each made for a chosen day.

With the same model at every stage the two are essentially identical
(gate ordering costs nothing); a weaker day-ahead view for the FCR/aFRR
stages (e.g. seasonal_naive) is the strict variant. See README
"Sequential clearing".
"""
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (_APP_DIR, _APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis import NO_PROXY_LABEL, proxy_label, sequential_comparison
from data_loader import (
    filter_proxy, load_daily_revenue, load_long_format, load_run_catalogue, load_timings, matching_runs,
)

st.title("Sequential vs Simultaneous Clearing")

catalogue = load_run_catalogue()
if catalogue.empty or "full_sequential" not in set(catalogue["strategy"]):
    st.warning(
        "No `full_sequential` runs found. Solve some with "
        "`python -m src.batch --strategies full_sequential` (add `--sequential-da-proxy seasonal_naive` "
        "for the strict variant)."
    )
    st.stop()

seq_rows = catalogue[(catalogue["strategy"] == "full_sequential") & (catalogue["decision_source"] == "forecast")]
labels = sorted({proxy_label(v) for v in seq_rows["da_proxy_model"]}) if "da_proxy_model" in seq_rows.columns else [NO_PROXY_LABEL]
proxy = st.sidebar.selectbox("Reserve-stage day-ahead view", labels)
windows = sorted(set(zip(seq_rows["start_date"], seq_rows["end_date"])))
window = st.sidebar.selectbox("Date window", windows, format_func=lambda w: f"{w[0]} to {w[1]}")
quantiles = sorted(seq_rows["forecast_quantile"].dropna().unique())
quantile = st.sidebar.selectbox("Quantile", quantiles, index=quantiles.index(0.5) if 0.5 in quantiles else 0)


def _one(rows: pd.DataFrame):
    rows = rows[(rows["start_date"] == window[0]) & (rows["end_date"] == window[1])]
    return None if rows.empty else rows.iloc[0]


perfect_run = _one(matching_runs(catalogue, "full_cooptimized", "realized"))
if perfect_run is None:
    st.info("No perfect-foresight `full_cooptimized` run for this window.")
    st.stop()
perfect_daily = load_daily_revenue(perfect_run["solve_id"])

rows, runs_by_model = [], {}
for model in sorted(seq_rows["forecast_model"].dropna().unique()):
    sim_run = _one(matching_runs(catalogue, "full_cooptimized", "forecast", model, quantile))
    seq_run = _one(filter_proxy(matching_runs(catalogue, "full_sequential", "forecast", model, quantile), proxy))
    if sim_run is None or seq_run is None:
        continue
    comparison = sequential_comparison(
        perfect_daily, load_daily_revenue(sim_run["solve_id"]), load_daily_revenue(seq_run["solve_id"]),
    )
    if comparison is None:
        continue
    rows.append({"model": model, **comparison})
    runs_by_model[model] = (sim_run["solve_id"], seq_run["solve_id"])

st.caption(
    f"Window {window[0]} to {window[1]} | quantile {quantile} | reserve-stage view: {proxy} | net revenue "
    "settled on realized prices, summed over the days all three runs solved. "
    "On the optimizer's own prices sequential can never beat simultaneous; on realized prices either can "
    "come out ahead because forecasts are wrong."
)

if not rows:
    st.info("No model has perfect-foresight, simultaneous and sequential runs for this selection.")
    st.stop()

table = pd.DataFrame(rows).set_index("model")
st.dataframe(
    table.style.format(
        {
            "n_days": "{:.0f}", "perfect_eur": "{:,.0f}", "simultaneous_eur": "{:,.0f}", "sequential_eur": "{:,.0f}",
            "sequential_minus_simultaneous_eur": "{:+,.0f}", "simultaneous_capture_pct": "{:.2f}%",
            "sequential_capture_pct": "{:.2f}%",
        },
        na_rep="n/a",
    ),
    width="stretch",
)

st.subheader("Reserve commitments per 4h block")
model = st.selectbox("Model", list(runs_by_model))
sim_id, seq_id = runs_by_model[model]
sim_lf, seq_lf = load_long_format(sim_id), load_long_format(seq_id)
days = sorted(set(sim_lf["solve_date"]) & set(seq_lf["solve_date"]))
if not days:
    st.info("The two runs share no solved day.")
    st.stop()
day = st.selectbox("Day", days)

fcr_timing = load_timings()["fcr"]
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(14, 3.6))
for ax, (variable, title) in zip(axes, (("c_fcr", "FCR"), ("c_pos", "aFRR+"), ("c_neg", "aFRR-"))):
    for lf, color, label in ((sim_lf, "tab:blue", "simultaneous"), (seq_lf, "tab:orange", "sequential")):
        rows_ = lf[(lf["solve_date"] == day) & (lf["variable"] == variable)].sort_values("target_time")
        if rows_.empty:
            continue
        starts = list(rows_["target_time"])
        edges = starts + [starts[-1] + pd.Timedelta(hours=fcr_timing.product_length_hours)]
        ax.stairs(rows_["value"].to_numpy(), edges, baseline=None, color=color, linewidth=2, label=label, alpha=0.85)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
axes[0].set_ylabel("MW")
axes[0].legend(fontsize=8)
fig.tight_layout()
st.pyplot(fig)
