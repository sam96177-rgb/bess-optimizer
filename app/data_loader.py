"""Shared, cached data loading for the dashboard (SPEC.md §6a, Milestone 8
Part C). Reads only this repo's results/<solve_id>/ sidecars (via
src.results_catalogue.scan_runs) and the sibling bess-market-forecast
repo's results/*.parquet (via src.config.load_paths_config, read-only per
SPEC.md Hard Rule 3) -- nothing here builds or solves a model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run app/streamlit_app.py` puts app/ on sys.path, not the repo
# root -- add the root so `src.*` imports resolve regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import MarketTimingConfig, load_market_timing_configs, load_paths_config  # noqa: E402
from src.results_catalogue import scan_runs  # noqa: E402

RESULTS_DIR = _REPO_ROOT / "results"

# The plan's chosen fan widths: 90% interval (0.05/0.95) and 50% interval
# (0.25/0.75), plus the median -- only rendered where the model actually
# produced that quantile (point models only have q=0.5).
FAN_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@st.cache_data
def load_run_catalogue() -> pd.DataFrame:
    """One row per completed run (results/<solve_id>/run_meta.json), the
    single discovery path shared with the CLI and batch.py.
    """
    return scan_runs(RESULTS_DIR)


@st.cache_data
def load_long_format(solve_id: str) -> pd.DataFrame:
    path = RESULTS_DIR / solve_id / "long_format.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["target_time"] = pd.to_datetime(df["target_time"])
    df["solve_date"] = pd.to_datetime(df["solve_date"]).dt.date
    return df


@st.cache_data
def load_daily_revenue(solve_id: str) -> pd.DataFrame:
    path = RESULTS_DIR / solve_id / "daily_revenue.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["solve_date"] = pd.to_datetime(df["solve_date"]).dt.date
    return df


@st.cache_data
def load_skip_log(solve_id: str) -> pd.DataFrame:
    path = RESULTS_DIR / solve_id / "skip_log.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # A run written before rolling.py started giving skip_log.csv an
        # explicit header row leaves a column-less empty file when nothing
        # was skipped -- pd.read_csv can't parse that; treat it the same
        # as "no skipped days" rather than erroring the page.
        return pd.DataFrame(columns=["day", "reason"])


@st.cache_data
def pivot_long_format(long_format: pd.DataFrame) -> pd.DataFrame:
    """One row per target_time, one column per `variable` (SPEC.md
    long-format schema: solve_id, solve_date, target_time, block_id,
    market, variable, value, unit, is_assumption).

    p_ch/p_dis/u/e are recorded every 15-min step, so those columns are
    fully populated. c_fcr/c_pos/c_neg are recorded once per 4h block (at
    that block's start timestamp only), so those columns are sparse here
    by construction -- callers that need per-block capacity values should
    filter long_format's own market/block_id columns directly instead of
    reading them off this pivot.
    """
    if long_format.empty:
        return long_format
    wide = long_format.pivot_table(index="target_time", columns="variable", values="value", aggfunc="first")
    wide.columns.name = None
    return wide.sort_index()


@st.cache_data
def load_forecast_prices(model: str) -> pd.DataFrame:
    """Day-ahead y_pred quantile fan (columns q_<level>) plus y_true for
    `model`, indexed by target_time. Reads the sibling repo's
    results/*.parquet directly (read-only, SPEC.md Hard Rule 3) rather
    than through src.data.forecast_loader.load_raw_y_pred, which only
    returns one (model, quantile) slice at a time -- not enough to draw a
    fan across every quantile a model produced.
    """
    paths = load_paths_config()
    files = sorted(paths.results_dir.glob("*.parquet"))
    columns = ["market", "model", "quantile", "target_time", "y_pred", "y_true"]
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)

    subset = full[(full["market"] == "day_ahead") & (full["model"] == model)]
    if subset.empty:
        return pd.DataFrame()

    fan = subset.pivot_table(index="target_time", columns="quantile", values="y_pred")
    fan = fan.rename(columns={q: f"q_{q}" for q in fan.columns})
    fan.columns.name = None
    fan["y_true"] = subset.drop_duplicates("target_time").set_index("target_time")["y_true"]
    fan.index = pd.to_datetime(fan.index)
    return fan.sort_index()


@st.cache_data
def available_forecast_models() -> list[str]:
    paths = load_paths_config()
    files = sorted(paths.results_dir.glob("*.parquet"))
    if not files:
        return []
    models: set[str] = set()
    for f in files:
        models.update(pd.read_parquet(f, columns=["model"])["model"].unique())
    return sorted(models)


@st.cache_data
def load_timings() -> dict[str, MarketTimingConfig]:
    return load_market_timing_configs(load_paths_config())


def matching_runs(
    catalogue: pd.DataFrame,
    strategy: str,
    decision_source: str,
    forecast_model: str | None = None,
    forecast_quantile: float | None = None,
) -> pd.DataFrame:
    """Catalogue rows for one (strategy, decision_source, forecast_model,
    forecast_quantile) combination, newest first -- there can be more than
    one if the same combination was solved over different date windows
    (e.g. a smoke test alongside the full batch run).
    """
    if catalogue.empty:
        return catalogue
    mask = (catalogue["strategy"] == strategy) & (catalogue["decision_source"] == decision_source)
    if forecast_model is None:
        mask &= catalogue["forecast_model"].isna()
    else:
        mask &= catalogue["forecast_model"] == forecast_model
    if forecast_quantile is None:
        mask &= catalogue["forecast_quantile"].isna()
    else:
        mask &= catalogue["forecast_quantile"] == forecast_quantile
    return catalogue[mask].sort_values("created_at", ascending=False)
