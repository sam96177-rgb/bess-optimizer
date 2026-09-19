"""Pure-pandas comparisons behind the dashboard pages (no Streamlit, so they
are unit-tested directly). Every comparison is made on the INTERSECTION of
the days present in all runs involved, because runs can skip different days
(e.g. seasonal_naive has no forecast for 2026-03-29 / 2026-04-05).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

REVENUE_COMPONENTS = ["day_ahead", "fcr", "afrr_pos", "afrr_neg", "activation"]
CAPTURE_ROWS = REVENUE_COMPONENTS + ["gross", "net"]
NO_PROXY_LABEL = "same model at every stage"


def common_days(*daily_frames: pd.DataFrame) -> list[dt.date]:
    day_sets = [set(frame["solve_date"]) for frame in daily_frames if not frame.empty]
    if len(day_sets) != len(daily_frames) or not day_sets:
        return []
    return sorted(set.intersection(*day_sets))


def summed(daily: pd.DataFrame, days: list[dt.date], columns: list[str]) -> pd.Series:
    rows = daily[daily["solve_date"].isin(days)]
    return rows[columns].sum()


def revenue_split(daily: pd.DataFrame, days: list[dt.date] | None = None) -> dict[str, float]:
    """Gross, degradation cost and net over `days` (default: every day)."""
    if daily.empty:
        return {"gross": 0.0, "degradation": 0.0, "net": 0.0}
    days = sorted(set(daily["solve_date"])) if days is None else days
    s = summed(daily, days, ["gross", "degradation", "net"])
    return {k: float(s[k]) for k in ("gross", "degradation", "net")}


def oracle_capture(perfect_daily: pd.DataFrame, forecast_daily: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Per revenue component: what the forecast-driven run earned as a share
    of what perfect foresight earned on the same days.

    capture_pct is undefined (NaN) when the perfect-foresight figure is not
    positive: a share of a zero/negative base is meaningless -- and the
    day-ahead leg of a reserve-heavy strategy can be near zero or negative,
    which is why it is reported on its own rather than hidden in the total.
    """
    days = common_days(perfect_daily, forecast_daily)
    columns = CAPTURE_ROWS
    if not days:
        empty = pd.DataFrame(index=columns, columns=["perfect_eur", "forecast_eur", "gap_eur", "capture_pct"], dtype=float)
        return empty, 0
    perfect = summed(perfect_daily, days, columns)
    forecast = summed(forecast_daily, days, columns)
    table = pd.DataFrame({"perfect_eur": perfect, "forecast_eur": forecast})
    table["gap_eur"] = table["perfect_eur"] - table["forecast_eur"]
    table["capture_pct"] = np.where(table["perfect_eur"] > 0, 100.0 * table["forecast_eur"] / table["perfect_eur"], np.nan)
    return table, len(days)


def sequential_comparison(
    perfect_daily: pd.DataFrame, simultaneous_daily: pd.DataFrame, sequential_daily: pd.DataFrame,
) -> dict[str, float] | None:
    """Net revenue of perfect foresight, simultaneous and sequential clearing
    over the days all three have. None if there are no common days."""
    days = common_days(perfect_daily, simultaneous_daily, sequential_daily)
    if not days:
        return None
    net = lambda frame: float(summed(frame, days, ["net"])["net"])
    perfect, simultaneous, sequential = net(perfect_daily), net(simultaneous_daily), net(sequential_daily)
    return {
        "n_days": len(days),
        "perfect_eur": perfect,
        "simultaneous_eur": simultaneous,
        "sequential_eur": sequential,
        "sequential_minus_simultaneous_eur": sequential - simultaneous,
        "simultaneous_capture_pct": 100.0 * simultaneous / perfect if perfect > 0 else float("nan"),
        "sequential_capture_pct": 100.0 * sequential / perfect if perfect > 0 else float("nan"),
    }


def proxy_label(value) -> str:
    return NO_PROXY_LABEL if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def twin_headlines(summary: pd.DataFrame) -> dict[str, float]:
    if summary.empty:
        return {}
    any_violation = (summary["n_soc_violation_steps"] > 0) | (summary["n_power_violation_steps"] > 0)
    return {
        "n_days": int(summary["solve_date"].nunique()),
        "n_draws": int(summary["draw"].nunique()),
        "share_replays_with_violation": float(any_violation.mean()),
        "mean_dispatch_shortfall_mwh_per_day": float(summary["dispatch_shortfall_mwh"].mean()),
        "mean_activation_shortfall_mwh_per_day": float(
            (summary["activation_shortfall_up_mwh"] + summary["activation_shortfall_down_mwh"]).mean()
        ),
        "activation_delivered_mwh": float(summary["activation_delivered_mwh"].sum()),
        "activation_expected_mwh": float(summary["activation_expected_mwh"].sum()),
    }
