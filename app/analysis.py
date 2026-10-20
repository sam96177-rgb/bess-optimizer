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


def common_days(*daily_frames: pd.DataFrame) -> list[dt.date]:
    day_sets = [set(frame["solve_date"]) for frame in daily_frames if not frame.empty]
    if len(day_sets) != len(daily_frames) or not day_sets:
        return []
    return sorted(set.intersection(*day_sets))


def stacked_bars(daily: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(bottom, height) per revenue component. Positive and negative parts stack
    from separate baselines: day_ahead is negative on net-buyer days, and a single
    running total would draw those bars on top of the wrong component.
    """
    above = np.zeros(len(daily))
    below = np.zeros(len(daily))
    bars = {}
    for component in REVENUE_COMPONENTS:
        height = daily[component].to_numpy(dtype=float)
        bars[component] = (np.where(height >= 0, above, below), height)
        above = above + np.where(height >= 0, height, 0.0)
        below = below + np.where(height < 0, height, 0.0)
    return bars


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


_CAPTURE_COLUMNS = ["perfect_eur", "realized_eur", "capture_pct"]


def _share(part: pd.Series, whole: pd.Series) -> pd.Series:
    return pd.Series(np.where(whole > 0, 100.0 * part / whole, np.nan), index=whole.index)


def oracle_capture(perfect_daily: pd.DataFrame, forecast_daily: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Per revenue component, how much a forecast-driven run earned relative to
    the ceiling (the optimizer planning on the real prices) on the same days.

    `capture_pct` = the plan paid at real prices (`realized_eur`) over the
    ceiling paid at real prices (`perfect_eur`) -- PLAN_V6_REVISION.md V6-1
    dropped the headline (forecast-price) capture figure that runs before v6
    also reported; `daily_revenue.parquet` still carries its `expected_*`
    columns (untouched, so rolling.py and its tests don't need to change),
    they are just no longer read here.

    Shares are undefined (NaN) when the ceiling figure is not positive: a
    share of a zero/negative base is meaningless -- and the day-ahead leg of
    a reserve-heavy strategy can be near zero or negative, which is why it is
    reported on its own rather than hidden in the total.
    """
    days = common_days(perfect_daily, forecast_daily)
    if not days:
        empty = pd.DataFrame(index=CAPTURE_ROWS, columns=_CAPTURE_COLUMNS, dtype=float)
        return empty, 0
    perfect = summed(perfect_daily, days, CAPTURE_ROWS)
    realized = summed(forecast_daily, days, CAPTURE_ROWS)
    table = pd.DataFrame({"perfect_eur": perfect, "realized_eur": realized})
    table["capture_pct"] = _share(table["realized_eur"], table["perfect_eur"])
    return table, len(days)


BASELINE_MODEL = "seasonal_naive"
_ERROR_COLUMNS = ["n", "mae", "rmse", "skill_vs_baseline"]


def forecast_error_metrics(forecasts: pd.DataFrame, baseline: str = BASELINE_MODEL) -> pd.DataFrame:
    """MAE, RMSE and skill per (market, model), from long-format median
    forecasts (columns market, model, target_time, y_pred, y_true).

    Within each market every model is scored on the SAME timestamps: only
    those where every model has a (non-NaN) prediction and the realized price
    is known. That is the same intersection rule the revenue comparisons use,
    so one model's missing days (seasonal_naive has none for 2026-03-29 and
    2026-04-05) cannot flatter or penalise it against the others.

    skill_vs_baseline = 1 - RMSE(model) / RMSE(baseline): 1 is perfect, 0 ties
    the baseline, negative is worse. NaN when the baseline is absent from the
    market or has zero RMSE.
    """
    rows = []
    for market, group in forecasts.groupby("market"):
        wide = group.pivot_table(index="target_time", columns="model", values="y_pred", aggfunc="first")
        truth = group.drop_duplicates("target_time").set_index("target_time")["y_true"].reindex(wide.index)
        usable = wide.notna().all(axis=1) & truth.notna()
        wide, truth = wide[usable], truth[usable]
        if wide.empty:
            continue
        error = wide.sub(truth, axis=0)
        mae, rmse = error.abs().mean(), np.sqrt((error**2).mean())
        base_rmse = rmse.get(baseline, np.nan)
        for model in wide.columns:
            skill = 1.0 - rmse[model] / base_rmse if base_rmse and not np.isnan(base_rmse) else np.nan
            rows.append(
                {"market": market, "model": model, "n": len(wide), "mae": mae[model], "rmse": rmse[model],
                 "skill_vs_baseline": skill}
            )
    if not rows:
        return pd.DataFrame(columns=_ERROR_COLUMNS, index=pd.MultiIndex.from_tuples([], names=["market", "model"]))
    return pd.DataFrame(rows).set_index(["market", "model"])[_ERROR_COLUMNS]


QUARTER_HOURS = 0.25
ACTIVE_MW = 1e-3  # below this a quarter-hour counts as idle
_AGREEMENT_COLUMNS = [
    f"{side}_{what}" for side in ("charge", "discharge") for what in ("timing_pct", "amount_pct", "mwh_per_day")
] + ["wrong_way_qh_per_day"]


def _active_by_quarter(long_format: pd.DataFrame, days: list[dt.date]) -> pd.DataFrame:
    """Boolean charge/discharge activity per quarter-hour (index solve_date, target_time)."""
    rows = long_format[long_format["variable"].isin(["p_ch", "p_dis"]) & long_format["solve_date"].isin(days)]
    wide = rows.pivot_table(index=["solve_date", "target_time"], columns="variable", values="value", aggfunc="first")
    return wide.sort_index().rename(columns={"p_ch": "charge", "p_dis": "discharge"})


def plan_agreement(
    model_long: pd.DataFrame, ceiling_long: pd.DataFrame, days: list[dt.date], tolerance_quarters: int = 2,
) -> dict[str, float]:
    """How closely a model's day-ahead plan follows the ceiling plan on `days`
    (long_format.parquet rows p_ch / p_dis, MW per quarter-hour).

    timing_pct: share of the model's active quarter-hours that have the ceiling
    active in the same direction within `tolerance_quarters` on the same day.
    amount_pct: 1 - sum|daily MWh model - ceiling| / sum daily MWh ceiling
    (ceiling 14 MWh, model 17 -> 79%; can go negative). wrong_way_qh_per_day:
    quarter-hours where the model charges while the ceiling discharges, or the reverse.
    """
    model, ceiling = _active_by_quarter(model_long, days), _active_by_quarter(ceiling_long, days)
    if model.empty or ceiling.empty or not days:
        return dict.fromkeys(_AGREEMENT_COLUMNS, float("nan"))
    ceiling = ceiling.reindex(model.index)
    out: dict[str, float] = {}
    for side, other in (("charge", "discharge"), ("discharge", "charge")):
        model_on, ceiling_on = model[side] > ACTIVE_MW, ceiling[side] > ACTIVE_MW
        near = ceiling_on.groupby(level="solve_date").transform(
            lambda s: s.astype(float).rolling(2 * tolerance_quarters + 1, center=True, min_periods=1).max() > 0
        )
        out[f"{side}_timing_pct"] = 100.0 * (model_on & near).sum() / model_on.sum() if model_on.any() else float("nan")
        daily_model = (model[side] * QUARTER_HOURS).groupby(level="solve_date").sum()
        daily_ceiling = (ceiling[side] * QUARTER_HOURS).groupby(level="solve_date").sum()
        out[f"{side}_amount_pct"] = (
            100.0 * (1 - (daily_model - daily_ceiling).abs().sum() / daily_ceiling.sum())
            if daily_ceiling.sum() > 0 else float("nan")
        )
        out[f"{side}_mwh_per_day"] = float(daily_model.mean())
        out[f"_wrong_{side}"] = float((model_on & (ceiling[other] > ACTIVE_MW)).sum())
    out["wrong_way_qh_per_day"] = (out.pop("_wrong_charge") + out.pop("_wrong_discharge")) / len(days)
    return {k: out[k] for k in _AGREEMENT_COLUMNS}


_REVENUE_TABLE_COLUMNS = _AGREEMENT_COLUMNS + ["net_eur", "capture_pct", "value_vs_baseline_eur"]


def model_revenue_table(
    perfect_daily: pd.DataFrame, model_daily: dict[str, pd.DataFrame],
    long_formats: tuple[pd.DataFrame, dict[str, pd.DataFrame]] | None = None,
) -> tuple[pd.DataFrame, float, int]:
    """Plan agreement, net revenue and capture per model over the WHOLE window,
    on the days that the ceiling and every model all solved.

    Returns (table indexed by model, ceiling net EUR, n_days). The first
    columns are plan_agreement (timing and amount vs the ceiling plan; NaN if
    `long_formats` = (ceiling long format, {model: long format}) is not given).
    net_eur is realized (decision settled on real prices); capture_pct is
    net_eur over the ceiling net (see oracle_capture); value_vs_baseline_eur =
    model net - BASELINE_MODEL's net on the SAME common days (NaN if
    BASELINE_MODEL isn't in `model_daily`), so a model is scored against
    "no forecasting model" as well as against the unreachable ceiling.
    """
    empty = pd.DataFrame(index=list(model_daily), columns=_REVENUE_TABLE_COLUMNS, dtype=float)
    days = common_days(perfect_daily, *model_daily.values())
    if not days:
        return empty, float("nan"), 0
    perfect_net = float(summed(perfect_daily, days, ["net"])["net"])
    perfect_kept = perfect_daily[perfect_daily["solve_date"].isin(days)]
    rows = {}
    for model, daily in model_daily.items():
        table, _ = oracle_capture(perfect_kept, daily[daily["solve_date"].isin(days)])
        agreement = (
            plan_agreement(long_formats[1][model], long_formats[0], days)
            if long_formats is not None and model in long_formats[1] else dict.fromkeys(_AGREEMENT_COLUMNS, float("nan"))
        )
        rows[model] = {
            **agreement,
            "net_eur": table.loc["net", "realized_eur"],
            "capture_pct": table.loc["net", "capture_pct"],
        }
    baseline_net = rows.get(BASELINE_MODEL, {}).get("net_eur", float("nan"))
    for row in rows.values():
        row["value_vs_baseline_eur"] = row["net_eur"] - baseline_net
    return pd.DataFrame.from_dict(rows, orient="index")[_REVENUE_TABLE_COLUMNS], perfect_net, len(days)


def forecast_shape_quality(y_pred: pd.Series, y_true: pd.Series, days: list[dt.date]) -> dict[str, float]:
    """Does a forecast put the day's local minima/maxima in the right
    quarter-hours? `shape_corr` compares the intraday PROFILE (mean-centred
    per day, so a level bias doesn't get credit or blame here -- that's
    `bias_eur_mwh`); `peak_timing_gap_h` is the median distance between the
    forecast's and the real day's peak hour.
    """
    corrs, biases, peak_gaps = [], [], []
    for day in days:
        t = y_true[[i for i in y_true.index if i.date() == day]]
        p = y_pred.reindex(t.index)
        if p.isna().any() or len(t) < 8:
            continue
        if p.std() > 0 and t.std() > 0:
            corrs.append(float(np.corrcoef(p - p.mean(), t - t.mean())[0, 1]))
        biases.append(float((p - t).mean()))
        peak_gaps.append(abs((p.values.argmax() - t.values.argmax()) * 0.25))
    return {
        "shape_corr": float(np.mean(corrs)) if corrs else float("nan"),
        "bias_eur_mwh": float(np.mean(biases)) if biases else float("nan"),
        "peak_timing_gap_h": float(np.median(peak_gaps)) if peak_gaps else float("nan"),
    }


_PERFECT_SUMMARY_COLUMNS = ["gross_eur", "wear_eur", "net_eur", "cycles_per_day"] + [f"{c}_eur" for c in REVENUE_COMPONENTS]


def perfect_foresight_summary(catalogue: pd.DataFrame, daily_by_id: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per perfect-foresight (decision_source == 'realized') strategy
    in `catalogue`: gross/wear/net straight from that run's run_meta columns
    (not recomputed) and cycles_per_day (equivalent_full_cycles / n_days_solved,
    matching the study report's "Cycles / day" column), plus each revenue
    component's EUR total from its daily_revenue (`daily_by_id[solve_id]`,
    summed over the days actually solved) -- the market split used to explain
    where the gross total comes from.
    """
    perfect = catalogue[catalogue["decision_source"] == "realized"]
    rows = {}
    for run in perfect.itertuples():
        row = {
            "gross_eur": run.gross_revenue_eur,
            "wear_eur": run.degradation_cost_eur,
            "net_eur": run.net_revenue_eur,
            "cycles_per_day": run.equivalent_full_cycles / run.n_days_solved if run.n_days_solved else float("nan"),
        }
        daily = daily_by_id.get(run.solve_id)
        if daily is not None and not daily.empty:
            days = sorted(set(daily["solve_date"]))
            row.update({f"{k}_eur": v for k, v in summed(daily, days, REVENUE_COMPONENTS).items()})
        rows[run.strategy] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=_PERFECT_SUMMARY_COLUMNS)


def capture_comparison(old_table: pd.DataFrame, new_table: pd.DataFrame, column: str = "capture_pct") -> tuple[dict[str, float], dict[str, float]]:
    """Restrict two per-model tables (model_revenue_table's first return
    value) to `column`, for the models both have a non-NaN value for -- the
    pairing a V4-vs-V5 bar chart needs (PLAN_V5_1_REVISION.md section 3)."""
    old = old_table[column].dropna()
    new = new_table[column].dropna()
    common = old.index.intersection(new.index)
    return old.loc[common].to_dict(), new.loc[common].to_dict()


def realized_prices(long_format: pd.DataFrame, y_true_da: pd.Series) -> tuple[float, float]:
    """Volume-weighted realized buy and sell price of a run's day-ahead
    schedule (long_format's p_ch/p_dis, settled on y_true_da)."""
    out = []
    for variable in ("p_ch", "p_dis"):
        s = long_format.loc[long_format["variable"] == variable].set_index("target_time")["value"]
        price = y_true_da.reindex(s.index)
        out.append(float((s * price).sum() / s.sum()) if s.sum() > 1e-9 else float("nan"))
    return out[0], out[1]
