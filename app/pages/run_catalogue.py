"""Run Catalogue: every solved run (results/<solve_id>/run_meta.json) and,
for a selected run, its skip log -- the discovery surface backing the
other two pages' run pickers.
"""
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (_APP_DIR, _APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st

from data_loader import load_run_catalogue, load_skip_log

st.title("Run Catalogue")

catalogue = load_run_catalogue()

if catalogue.empty:
    st.warning(
        "No runs found in `results/`. Solve one first, e.g. "
        "`python -m src.cli --start-date ... --end-date ... --write-results` "
        "or `python -m src.batch`."
    )
    st.stop()

st.caption(f"{len(catalogue)} run(s) on disk.")

display_cols = [
    c for c in [
        "solve_id", "strategy", "decision_source", "forecast_model", "forecast_quantile",
        "start_date", "end_date", "n_days_solved", "n_days_skipped", "total_revenue_eur",
        "git_commit", "created_at",
    ]
    if c in catalogue.columns
]
st.dataframe(
    catalogue[display_cols].sort_values("created_at", ascending=False).reset_index(drop=True),
    width="stretch",
)

st.subheader("Skip log for a run")
solve_id = st.selectbox("solve_id", catalogue["solve_id"].tolist())
skip_log = load_skip_log(solve_id)

row = catalogue[catalogue["solve_id"] == solve_id].iloc[0]
st.caption(
    f"{row['strategy']} | {row['decision_source']}"
    + (f" | {row['forecast_model']} q{row['forecast_quantile']}" if row["decision_source"] == "forecast" else "")
    + f" | {row['start_date']} to {row['end_date']} | "
    f"{row['n_days_solved']} solved, {row['n_days_skipped']} skipped"
)

if skip_log.empty:
    st.info("No skipped days for this run.")
else:
    st.dataframe(skip_log.reset_index(drop=True), width="stretch")
