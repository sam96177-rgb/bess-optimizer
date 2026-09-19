"""Digital Twin: a stored run's plan replayed under the battery's real
physics and random aFRR calls (`python -m src.twin`), planned vs twin SOC and
where the plan could not be delivered. Read-only: run the twin CLI first.
"""
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (_APP_DIR, _APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json

import matplotlib.pyplot as plt
import streamlit as st

from analysis import twin_headlines
from data_loader import RESULTS_DIR, load_run_catalogue, load_twin_meta, load_twin_paths, load_twin_summary, twin_mtime

st.title("Digital Twin")

catalogue = load_run_catalogue()
twin_runs = catalogue[catalogue["solve_id"].map(lambda s: twin_mtime(s) is not None)] if not catalogue.empty else catalogue
if twin_runs.empty:
    st.warning(
        "No run has twin output yet. Replay one with "
        "`python -m src.twin --solve-id <solve_id> --n-draws 500 --plot-day <YYYY-MM-DD>` "
        "(solve ids are in the Run Catalogue)."
    )
    st.stop()


def _label(r) -> str:
    who = f"{r.forecast_model} q{r.forecast_quantile}" if r.decision_source == "forecast" else "perfect foresight"
    return f"{r.strategy} | {who} | {r.start_date}..{r.end_date} | {r.solve_id[:8]}"


twin_runs = twin_runs.reset_index(drop=True)
idx = st.sidebar.selectbox("Run", range(len(twin_runs)), format_func=lambda i: _label(twin_runs.iloc[i]))
solve_id = twin_runs.iloc[idx]["solve_id"]
mtime = twin_mtime(solve_id)
meta, summary, paths = load_twin_meta(solve_id, mtime), load_twin_summary(solve_id, mtime), load_twin_paths(solve_id, mtime)

st.warning(
    "aFRR activation in the twin is an ASSUMPTION, not a measurement: "
    f"alpha_pos={meta['alpha_pos']}, alpha_neg={meta['alpha_neg']}"
    + (f", mean call size {meta['call_degree_mean']} of committed capacity" if "call_degree_mean" in meta else "")
    + f" (mode: {meta['mode']}, {meta['n_draws']} draws, seed {meta['seed']}). Calls are independent per step -- "
    "read violation rates as assumption-driven."
)

h = twin_headlines(summary)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Replays hitting a limit", f"{h['share_replays_with_violation'] * 100:.1f}%")
c2.metric("Dispatch/aux shortfall", f"{h['mean_dispatch_shortfall_mwh_per_day']:.2f} MWh/day")
c3.metric("Activation shortfall", f"{h['mean_activation_shortfall_mwh_per_day']:.3f} MWh/day")
c4.metric("Activation delivered vs planned", f"{h['activation_delivered_mwh'] / max(h['activation_expected_mwh'], 1e-9) * 100:.1f}%")

st.subheader("By day")
per_day = summary.groupby("solve_date").agg(
    share_with_violation=("n_soc_violation_steps", lambda s: float((s > 0).mean())),
    mean_dispatch_shortfall_mwh=("dispatch_shortfall_mwh", "mean"),
    mean_min_soc_mwh=("min_soc_mwh", "mean"),
    mean_max_soc_mwh=("max_soc_mwh", "mean"),
    mean_abs_soc_deviation_mwh=("max_abs_soc_deviation_mwh", "mean"),
)
st.dataframe(
    per_day.style.format(
        {"share_with_violation": "{:.0%}", "mean_dispatch_shortfall_mwh": "{:.3f}", "mean_min_soc_mwh": "{:.2f}",
         "mean_max_soc_mwh": "{:.2f}", "mean_abs_soc_deviation_mwh": "{:.2f}"}
    ),
    width="stretch",
)

st.subheader("Planned vs twin state of charge")
day = st.selectbox("Day", sorted(paths["solve_date"].unique()))
rows = paths[paths["solve_date"] == day]
run_meta_path = RESULTS_DIR / solve_id / "run_meta.json"
battery_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))["battery"] if run_meta_path.exists() else None

fig, ax = plt.subplots(figsize=(12, 4.2))
first = rows[rows["draw"] == rows["draw"].min()].sort_values("target_time")
ax.step(first["target_time"], first["planned_e_mwh"], where="post", color="black", lw=2, label="planned SOC")
for draw, group in rows.groupby("draw"):
    group = group.sort_values("target_time")
    ax.step(group["target_time"], group["twin_e_mwh"], where="post", lw=1, alpha=0.8, label=f"twin draw {draw}")
    hit = group[group["violation"]]
    if not hit.empty:
        ax.scatter(hit["target_time"], hit["twin_e_mwh"], color="red", s=14, zorder=5)
if battery_meta:
    lo = battery_meta["soc_min_fraction"] * battery_meta["energy_mwh"]
    hi = battery_meta["soc_max_fraction"] * battery_meta["energy_mwh"]
    ax.axhline(lo, color="grey", ls="--", lw=1)
    ax.axhline(hi, color="grey", ls="--", lw=1, label="SOC window")
ax.set_ylabel("SOC (MWh)")
ax.legend(loc="upper right", fontsize=8, ncol=3)
fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig)
st.caption("Red dots: steps where the plan could not be delivered (SOC clipped at a limit). Only the first draws' paths are stored.")

st.subheader("Shortfall distribution across draws")
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.hist(summary["dispatch_shortfall_mwh"], bins=30, color="tab:red", alpha=0.75)
ax2.set_xlabel("Planned dispatch/aux not delivered, MWh per (day, draw)")
ax2.set_ylabel("replays")
fig2.tight_layout()
st.pyplot(fig2)
