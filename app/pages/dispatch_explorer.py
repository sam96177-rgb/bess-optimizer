"""Dispatch Explorer: pick a forecast model, quantile, strategy, and date
window; see the day-ahead forecast fan, the battery's dispatch/SOC, its
ancillary capacity commitments, and daily/cumulative revenue against the
matching perfect-foresight run over the same window (SPEC.md §6a: decide
on forecast, settle on realized).
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

from analysis import revenue_split
from data_loader import (
    load_daily_revenue,
    load_forecast_prices,
    load_long_format,
    load_run_catalogue,
    load_timings,
    matching_runs,
    pivot_long_format,
)

st.title("Dispatch Explorer")

catalogue = load_run_catalogue()
if catalogue.empty:
    st.warning(
        "No runs found in `results/`. Solve one first, e.g. `python -m src.batch --dry-run` "
        "to see the matrix, then drop `--dry-run` (or use `python -m src.cli ... --write-results`)."
    )
    st.stop()

forecast_rows = catalogue[catalogue["decision_source"] == "forecast"]
if forecast_rows.empty:
    st.warning("No forecast-driven runs found yet -- only perfect-foresight runs are on disk so far.")
    st.stop()

models = sorted(forecast_rows["forecast_model"].dropna().unique())
model = st.sidebar.selectbox("Model", models)

quantiles = sorted(forecast_rows.loc[forecast_rows["forecast_model"] == model, "forecast_quantile"].dropna().unique())
quantile = st.sidebar.selectbox("Quantile", quantiles)

strategies = sorted(
    forecast_rows.loc[
        (forecast_rows["forecast_model"] == model) & (forecast_rows["forecast_quantile"] == quantile), "strategy",
    ].unique()
)
strategy = st.sidebar.selectbox("Strategy", strategies)

runs = matching_runs(catalogue, strategy, "forecast", model, quantile)
if len(runs) > 1:
    labels = [f"{r.start_date} to {r.end_date}  (solve_id={r.solve_id[:8]}…, {r.created_at})" for r in runs.itertuples()]
    picked = st.sidebar.selectbox("Run (multiple solved for this combination)", range(len(runs)), format_func=lambda i: labels[i])
    run_row = runs.iloc[picked]
else:
    run_row = runs.iloc[0]

solve_id = run_row["solve_id"]
start_date, end_date = run_row["start_date"], run_row["end_date"]

long_format = load_long_format(solve_id)
daily_revenue = load_daily_revenue(solve_id)

if daily_revenue.empty:
    st.warning(f"Run {solve_id} solved zero days (every day in {start_date}..{end_date} was skipped).")
    st.stop()

available_dates = sorted(daily_revenue["solve_date"].unique())
min_date, max_date = available_dates[0], available_dates[-1]
if min_date == max_date:
    window_start, window_end = min_date, max_date
    st.sidebar.caption(f"Single solved day: {min_date}")
else:
    window_start, window_end = st.sidebar.slider(
        "Date window", min_value=min_date, max_value=max_date, value=(min_date, max_date),
    )

# The comparison perfect-foresight run: same strategy, same exact date
# window as the run_meta.json this forecast run was solved with -- not
# just any perfect-foresight run for this strategy, since totals from
# different windows aren't comparable.
perfect_runs = matching_runs(catalogue, strategy, "realized")
perfect_match = perfect_runs[(perfect_runs["start_date"] == start_date) & (perfect_runs["end_date"] == end_date)]
perfect_row = perfect_match.iloc[0] if not perfect_match.empty else None

if perfect_row is None:
    st.info(
        f"No perfect-foresight run found for {strategy} over {start_date}..{end_date} -- "
        "revenue panel will show the forecast-driven run only. Solve one with the same "
        "--start-date/--end-date and --decision-source=realized to enable the comparison."
    )
    perfect_daily = pd.DataFrame()
else:
    perfect_daily = load_daily_revenue(perfect_row["solve_id"])

lf_window = long_format[(long_format["solve_date"] >= window_start) & (long_format["solve_date"] <= window_end)]
dr_window = daily_revenue[(daily_revenue["solve_date"] >= window_start) & (daily_revenue["solve_date"] <= window_end)]
pf_window = (
    perfect_daily[(perfect_daily["solve_date"] >= window_start) & (perfect_daily["solve_date"] <= window_end)]
    if not perfect_daily.empty
    else perfect_daily
)

fan = load_forecast_prices(model)
if not fan.empty:
    fan_window = fan[(fan.index.date >= window_start) & (fan.index.date <= window_end)]
else:
    fan_window = fan

wide = pivot_long_format(lf_window)

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(14, 10))

# --- Panel 1: day-ahead price -- decision (forecast fan) vs settlement (realized) ---
ax = axes[0]
if fan_window.empty:
    ax.text(0.5, 0.5, "No forecast data for this model/window", ha="center", va="center", transform=ax.transAxes)
else:
    x = fan_window.index
    if "q_0.05" in fan_window.columns and "q_0.95" in fan_window.columns and fan_window["q_0.05"].notna().any():
        ax.fill_between(x, fan_window["q_0.05"], fan_window["q_0.95"], color="tab:blue", alpha=0.15, label="5-95% interval")
    if "q_0.25" in fan_window.columns and "q_0.75" in fan_window.columns and fan_window["q_0.25"].notna().any():
        ax.fill_between(x, fan_window["q_0.25"], fan_window["q_0.75"], color="tab:blue", alpha=0.3, label="25-75% interval")
    if "q_0.5" in fan_window.columns:
        ax.plot(x, fan_window["q_0.5"], color="tab:blue", linewidth=1.5, label="median forecast (decision)")
    ax.plot(x, fan_window["y_true"], color="black", linewidth=1.3, linestyle="--", label="realized (settlement)")
    ax.legend(loc="upper left", fontsize=8)
ax.set_ylabel("EUR/MWh")
ax.set_title(f"Day-ahead price — {model} q{quantile}")

# --- Panel 2: battery dispatch + SOC ---
ax = axes[1]
ax_soc = ax.twinx()
lines, labels = [], []
if "p_ch" in wide.columns:
    l1, = ax.step(wide.index, wide["p_ch"], where="post", color="tab:green", label="charge (MW)")
    lines.append(l1)
if "p_dis" in wide.columns:
    l2, = ax.step(wide.index, -wide["p_dis"], where="post", color="tab:red", label="discharge (MW, shown negative)")
    lines.append(l2)
if "e" in wide.columns:
    l3, = ax_soc.plot(wide.index, wide["e"], color="black", linewidth=1.5, label="SOC (e, MWh)")
    lines.append(l3)
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_ylabel("MW")
ax_soc.set_ylabel("MWh")
ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
ax.set_title("Battery dispatch and state of charge")

# --- Panel 3: ancillary capacity commitments ---
ax = axes[2]
capacity_rows = lf_window[lf_window["variable"].isin(["c_fcr", "c_pos", "c_neg"])]
if capacity_rows.empty:
    ax.text(0.5, 0.5, "No ancillary capacity commitments for this strategy", ha="center", va="center", transform=ax.transAxes)
    ax.set_yticks([])
else:
    block_starts = sorted(capacity_rows["target_time"].unique())
    capacity_timing = load_timings()["fcr"]
    edges = list(block_starts) + [block_starts[-1] + pd.Timedelta(hours=capacity_timing.product_length_hours)]
    for variable, color, label in (("c_fcr", "tab:purple", "FCR"), ("c_pos", "tab:orange", "aFRR+"), ("c_neg", "tab:cyan", "aFRR-")):
        series = capacity_rows[capacity_rows["variable"] == variable].set_index("target_time")["value"].reindex(block_starts)
        ax.stairs(series.values, edges, baseline=None, color=color, linewidth=2, label=label)
    ax.legend(loc="upper left", fontsize=8)
ax.set_ylabel("MW")
ax.set_title("Ancillary capacity commitments (per 4h block)")

# --- Panel 4: daily + cumulative revenue, forecast-driven vs perfect-foresight ---
ax = axes[3]
ax_cum = ax.twinx()
fc_x = pd.to_datetime(dr_window["solve_date"])
fc_y = dr_window["total"].to_numpy()
offset = pd.Timedelta(hours=6)
bar_lines, bar_labels = [], []
b1 = ax.bar(fc_x - offset if not pf_window.empty else fc_x, fc_y, width=0.35, color="tab:blue", label="forecast-driven (daily)")
c1, = ax_cum.plot(fc_x, fc_y.cumsum(), color="tab:blue", linewidth=2, label="forecast-driven (cumulative)")
bar_lines += [b1, c1]
if not pf_window.empty:
    pf_x = pd.to_datetime(pf_window["solve_date"])
    pf_y = pf_window["total"].to_numpy()
    b2 = ax.bar(pf_x + offset, pf_y, width=0.35, color="tab:gray", label="perfect foresight (daily)")
    c2, = ax_cum.plot(pf_x, pf_y.cumsum(), color="black", linewidth=2, linestyle="--", label="perfect foresight (cumulative)")
    bar_lines += [b2, c2]
ax.set_ylabel("EUR / day")
ax_cum.set_ylabel("EUR cumulative")
ax.legend(bar_lines, [h.get_label() for h in bar_lines], loc="upper left", fontsize=8)
ax.set_title("Net revenue (gross minus degradation)")
ax.set_xlabel("Delivery time")

fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig)

def _split_text(frame: pd.DataFrame) -> str:
    if "gross" not in frame.columns:  # run predates the gross/degradation columns
        return f"EUR {frame['total'].sum():,.2f} net"
    s = revenue_split(frame)
    return f"gross EUR {s['gross']:,.2f} - degradation EUR {s['degradation']:,.2f} = net EUR {s['net']:,.2f}"


st.caption(
    f"{len(dr_window)} day(s) shown | forecast-driven: {_split_text(dr_window)}"
    + (f" | perfect foresight: {_split_text(pf_window)}" if not pf_window.empty else "")
)
