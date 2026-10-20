"""Shared, cached data loading for the dashboard (SPEC.md §6a, Milestone 8
Part C). Reads only this repo's results/<solve_id>/ sidecars (via
src.results_catalogue.scan_runs) and the sibling bess-market-forecast
repo's results/*.parquet (via src.config.load_paths_config, read-only per
SPEC.md Hard Rule 3) -- nothing here builds or solves a model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# `streamlit run app/streamlit_app.py` puts app/ on sys.path, not the repo
# root -- add the root so `src.*` imports resolve regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import MarketTimingConfig, load_market_timing_configs, load_paths_config  # noqa: E402
from src.data.forecast_loader import MEDIAN_QUANTILE, resolve_per_market_models  # noqa: E402
from src.results_catalogue import scan_runs  # noqa: E402

RESULTS_DIR = _REPO_ROOT / "results"
MODEL_INFO_PATH = Path(__file__).resolve().parent / "model_info.yaml"

# Strategies no dashboard page shows. Old runs of them may still be on disk
# (they stay in results/ and in scripts/export_runs.py's CSV); the pages just
# never offer them. full_sequential is still runnable via `src.batch --strategies`.
HIDDEN_STRATEGIES = frozenset({"naive", "full_sequential"})


@st.cache_data
def load_run_catalogue() -> pd.DataFrame:
    """One row per completed run (results/<solve_id>/run_meta.json), the
    single discovery path shared with the CLI and batch.py, minus the
    strategies the dashboard hides.
    """
    catalogue = scan_runs(RESULTS_DIR)
    if catalogue.empty:
        return catalogue
    return catalogue[~catalogue["strategy"].isin(HIDDEN_STRATEGIES)].reset_index(drop=True)


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
def load_median_forecasts() -> pd.DataFrame:
    """Every model's median forecast and the realized price, long format:
    market, model, target_time, y_pred, y_true. Reads the sibling repo's
    results/*.parquet directly (read-only, SPEC.md Hard Rule 3), with no
    import from that repo. Empty if there are no result files.
    """
    paths = load_paths_config()
    files = sorted(paths.results_dir.glob("*.parquet"))
    columns = ["market", "model", "quantile", "target_time", "y_pred", "y_true"]
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    if not frames:
        return pd.DataFrame(columns=["market", "model", "target_time", "y_pred", "y_true"])
    full = pd.concat(frames, ignore_index=True)
    median = full[full["quantile"] == MEDIAN_QUANTILE].drop(columns="quantile")
    median["target_time"] = pd.to_datetime(median["target_time"])
    return median.reset_index(drop=True)


def load_forecast_prices(model: str, market: str = "day_ahead") -> pd.DataFrame:
    """`market`'s median forecast (`forecast`) and realized price (`y_true`)
    for `model`, indexed by target_time. A blend name (per_market_best) uses
    the real model it assigns to `market`."""
    median = load_median_forecasts()
    subset = median[(median["market"] == market) & (median["model"] == resolve_per_market_models(model)[market])]
    if subset.empty:
        return pd.DataFrame()
    out = subset.drop_duplicates("target_time").set_index("target_time")[["y_pred", "y_true"]]
    return out.rename(columns={"y_pred": "forecast"}).sort_index()


@st.cache_data
def load_timings() -> dict[str, MarketTimingConfig]:
    return load_market_timing_configs(load_paths_config())


def matching_runs(
    catalogue: pd.DataFrame,
    strategy: str,
    decision_source: str,
    forecast_model: str | None = None,
) -> pd.DataFrame:
    """Catalogue rows for one (strategy, decision_source, forecast_model)
    combination, newest first -- there can be more than one if the same
    combination was solved over different date windows (e.g. a smoke test
    alongside the full batch run). Forecast runs are always the median:
    it is the only quantile ever solved.
    """
    if catalogue.empty:
        return catalogue
    mask = (catalogue["strategy"] == strategy) & (catalogue["decision_source"] == decision_source)
    if forecast_model is None:
        mask &= catalogue["forecast_model"].isna()
    else:
        mask &= catalogue["forecast_model"] == forecast_model
    if decision_source == "forecast":
        mask &= catalogue["forecast_quantile"] == MEDIAN_QUANTILE
    return catalogue[mask].sort_values("created_at", ascending=False)


@st.cache_data
def load_forecast_run_meta(run_id: str) -> dict:
    """One forecast backtest's own run JSON (sibling repo's results/<run_id>.json
    -- NOT this repo's results/<solve_id>/run_meta.json), for fields the
    dashboard reads but never computes itself, e.g. first_train_days."""
    path = load_paths_config().results_dir / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_lookahead_summary(study_name: str) -> dict | None:
    """results/lookahead_<study>.json (scripts/lookahead_value.py), or None
    if that study hasn't been run through it yet."""
    path = RESULTS_DIR / f"lookahead_{study_name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_model_info() -> dict[str, dict]:
    """PLAN_V5_1_REVISION.md section 1: one card per forecast name the
    batch matrix can produce (app/model_info.yaml), keyed by model name."""
    return yaml.safe_load(MODEL_INFO_PATH.read_text(encoding="utf-8"))
