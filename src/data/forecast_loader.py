"""Reads bess-market-forecast's results/*.parquet, read-only (SPEC.md Hard
Rule 3), and aligns each day's prices onto that day's grid
(src/data/grid.py). A day with any missing or NaN price in any of the four
markets is skipped and logged with a reason; it is never zero-filled or
interpolated (SPEC.md Hard Rule 1). This is also how the known 2026-09-13
hourly-instead-of-15-minute day gets handled: it isn't special-cased by
date, it's just a day where day_ahead's 15-min steps don't all have a
value, so the same general check skips it.

Two price columns are available and they mean different things (SPEC.md
§6a): `y_true` is the realized price and `y_pred` is what a given
forecasting model predicted ahead of the market's gate closure. A
forecast-driven run DECIDES on y_pred but must SETTLE on y_true — see
DayPricePair below.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from dataclasses import dataclass

import pandas as pd

from src.config import (
    _REQUIRED_MARKETS,
    MarketTimingConfig,
    PathsConfig,
    load_forecast_blends_config,
    load_paths_config,
)
from src.data.grid import DayGrid, build_day_grid

logger = logging.getLogger(__name__)

# Only the median is ever solved: every registered forecast model is a point model.
# `forecast_quantile` stays a parameter (and is recorded in run_meta.json) for provenance.
MEDIAN_QUANTILE = 0.5


@dataclass(frozen=True)
class DayPrices:
    day: dt.date
    grid: DayGrid
    day_ahead: pd.Series  # indexed by grid.steps, EUR/MWh
    fcr: pd.Series  # indexed by block index (grid.blocks[i].index)
    afrr_pos: pd.Series
    afrr_neg: pd.Series


@dataclass(frozen=True)
class DayPricePair:
    """The two price views of one delivery day. `decision` is what the
    optimizer is allowed to see when it commits (y_pred for a
    forecast-driven run, y_true for a perfect-foresight one); `settlement`
    is always y_true, because that is what the schedule actually earns.

    Kept as a pair of DayPrices rather than as extra fields on DayPrices
    itself: DayPrices is pickled into the per-day solver subprocess
    (src/model/rolling.py), and the settlement prices have no business
    being shipped to the solver — it must not be able to see them.
    """

    day: dt.date
    decision: DayPrices
    settlement: DayPrices
    # Sequential clearing only: what the FCR and aFRR stages (which close
    # before the day-ahead forecast exists) are allowed to see -- the
    # decision prices with day_ahead swapped for an earlier-available proxy.
    reserve_stage_decision: DayPrices | None = None


@dataclass(frozen=True)
class SkipRecord:
    day: dt.date
    reason: str


def load_raw_y_true(paths_config: PathsConfig) -> pd.Series:
    """One realized price per (market, target_time), read from every
    results/*.parquet file. y_true is identical across every
    model/quantile/horizon_step row for the same (market, target_time)
    (verified empirically — see Milestone 0/2 notes) so this collapses
    them; it fails loudly instead if it ever finds genuinely conflicting
    values, rather than silently picking one.
    """
    files = sorted(paths_config.results_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {paths_config.results_dir}")

    frames = [pd.read_parquet(f, columns=["market", "target_time", "y_true"]) for f in files]
    # Drop only EXACT duplicate rows (same market, target_time, AND
    # y_true) -- e.g. the same realized value repeated across every
    # model/quantile/horizon_step row. Deliberately not deduped on
    # (market, target_time) alone, which would silently keep an arbitrary
    # row if two disagreed on y_true instead of catching it below.
    full = pd.concat(frames, ignore_index=True).drop_duplicates()

    distinct_values = full.groupby(["market", "target_time"])["y_true"].nunique(dropna=False)
    if (distinct_values > 1).any():
        offending = distinct_values[distinct_values > 1].index.tolist()[:5]
        raise ValueError(
            "Found conflicting y_true values for the same (market, target_time) in "
            f"{paths_config.results_dir}, e.g. {offending}. This should never happen for a "
            "realized price — stop and investigate rather than picking one arbitrarily."
        )
    return full.set_index(["market", "target_time"])["y_true"].sort_index()


def load_raw_y_pred(paths_config: PathsConfig, model: str, quantile: float) -> pd.Series:
    """One predicted price per (market, target_time), for a single
    forecasting model at a single quantile -- or, if `model` names a
    config/forecast_blends.yaml blend, its per-market model instead (each
    market's slice from its own model, same guards, same MultiIndex shape).

    Unlike y_true, y_pred genuinely differs per model and per quantile, so
    this filters rather than collapses. Verified against the real results:
    (market, model, quantile, target_time) is unique across all 350,710
    rows, with exactly one cutoff per target_time, so no horizon
    disambiguation is needed — but it is asserted here anyway rather than
    assumed, since a future multi-horizon backtest would break it silently.

    NaN predictions are deliberately KEPT. A model can legitimately fail to
    predict some periods (seasonal_naive produces NaN for the 2026-03-29
    DST day and its one-week echo, because its 'same block last week'
    lookup lands on an irregular day). Dropping them here would make those
    periods look merely absent; keeping them lets load_day_prices skip the
    day with a legible reason, which is also what src/model/build.py's own
    NaN guards require.
    """
    blend = load_forecast_blends_config().resolve(model)
    if blend is not None:
        return _load_raw_y_pred_blend(paths_config, blend, quantile)

    files = sorted(paths_config.results_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {paths_config.results_dir}")

    columns = ["market", "model", "quantile", "target_time", "y_pred"]
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    full = pd.concat(frames, ignore_index=True)

    available_models = sorted(full["model"].unique())
    if model not in available_models:
        raise ValueError(f"Unknown forecast model '{model}'. Available: {available_models}")

    for_model = full[full["model"] == model]
    available_quantiles = sorted(for_model["quantile"].unique())
    selected = for_model[for_model["quantile"] == quantile]
    if selected.empty:
        raise ValueError(
            f"Forecast model '{model}' has no rows at quantile {quantile}. "
            f"Available quantiles for this model: {available_quantiles}. "
            "(Point models only produce the median, 0.5.)"
        )

    missing_markets = _REQUIRED_MARKETS - set(selected["market"].unique())
    if missing_markets:
        raise ValueError(
            f"Forecast model '{model}' at quantile {quantile} is missing predictions for "
            f"{sorted(missing_markets)}. All four markets are required to solve a day."
        )

    counts = selected.groupby(["market", "target_time"]).size()
    if (counts > 1).any():
        offending = counts[counts > 1].index.tolist()[:5]
        raise ValueError(
            f"Found multiple y_pred rows for the same (market, target_time) for model '{model}' "
            f"at quantile {quantile}, e.g. {offending}. This loader assumes one prediction per "
            "delivery period — stop and investigate (a multi-horizon backtest would cause this) "
            "rather than picking one arbitrarily."
        )
    return selected.set_index(["market", "target_time"])["y_pred"].sort_index()


def _load_raw_y_pred_blend(paths_config: PathsConfig, per_market: dict[str, str], quantile: float) -> pd.Series:
    """One load_raw_y_pred call per DISTINCT underlying model (not one per
    market, so a model used in three of the blend's four markets is only
    read once), then each market's rows come from its own model's result --
    load_raw_y_pred's own guards (all 4 markets present, one row per
    market/target_time, NaNs kept) already ran once per underlying model.
    """
    by_model = {name: load_raw_y_pred(paths_config, name, quantile) for name in set(per_market.values())}
    slices = [by_model[model].xs(market, level="market", drop_level=False) for market, model in per_market.items()]
    return pd.concat(slices).sort_index()


def resolve_per_market_models(model: str) -> dict[str, str]:
    """The real model backing each market for `model`: its own per-market
    dict if it names a config/forecast_blends.yaml blend, else itself
    repeated across all four markets. Provenance only (run_meta.json's
    forecast_models field) -- load_raw_y_pred resolves blends itself.
    """
    blend = load_forecast_blends_config().resolve(model)
    return dict(blend) if blend is not None else {m: model for m in _REQUIRED_MARKETS}


def load_forecast_run_ids(paths_config: PathsConfig, model: str) -> dict[str, str]:
    """One run_id per market for `model` — provenance only (run_meta.json's
    forecast_run_ids field), not used in any price calculation. If `model`
    names a blend, each market's run_id comes from its own underlying model.

    Verified against the real results: there are exactly 4 markets x 5
    models = 20 parquet files, one per (market, model) pair, and each
    file's run_id column is constant within it and equal to the filename.
    Asserted here rather than assumed, for the same reason as
    load_raw_y_pred's own uniqueness checks.
    """
    blend = load_forecast_blends_config().resolve(model)
    if blend is not None:
        return {market: load_forecast_run_ids(paths_config, m)[market] for market, m in blend.items()}

    files = sorted(paths_config.results_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {paths_config.results_dir}")

    columns = ["market", "model", "run_id"]
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    full = pd.concat(frames, ignore_index=True)

    for_model = full[full["model"] == model]
    if for_model.empty:
        available_models = sorted(full["model"].unique())
        raise ValueError(f"Unknown forecast model '{model}'. Available: {available_models}")

    run_ids: dict[str, str] = {}
    for market, group in for_model.groupby("market"):
        distinct = group["run_id"].unique()
        if len(distinct) != 1:
            raise ValueError(
                f"Expected exactly one run_id for model '{model}' in market '{market}', "
                f"found {sorted(distinct)}."
            )
        run_ids[str(market)] = str(distinct[0])
    return run_ids


def _extract_day_ahead(raw: pd.Series, grid: DayGrid) -> pd.Series | None:
    if "day_ahead" not in raw.index.get_level_values("market"):
        return None
    values = raw.loc["day_ahead"].reindex(grid.steps)
    if values.isna().any():
        return None
    return values


def _extract_capacity_market(raw: pd.Series, market: str, grid: DayGrid) -> pd.Series | None:
    if market not in raw.index.get_level_values("market"):
        return None
    block_starts = [b.start for b in grid.blocks]
    values = raw.loc[market].reindex(block_starts)
    if values.isna().any():
        return None
    return pd.Series(values.to_numpy(), index=[b.index for b in grid.blocks])


def load_day_prices(
    day: dt.date,
    raw: pd.Series,
    day_ahead_timing: MarketTimingConfig,
    capacity_timing: MarketTimingConfig,
) -> DayPrices | SkipRecord:
    grid = build_day_grid(day, day_ahead_timing, capacity_timing)

    day_ahead = _extract_day_ahead(raw, grid)
    if day_ahead is None:
        return SkipRecord(day, "day_ahead: missing or NaN price for one or more of this day's 15-min steps")

    fcr = _extract_capacity_market(raw, "fcr", grid)
    if fcr is None:
        return SkipRecord(day, "fcr: missing or NaN price for one or more of this day's blocks")

    afrr_pos = _extract_capacity_market(raw, "afrr_pos", grid)
    if afrr_pos is None:
        return SkipRecord(day, "afrr_pos: missing or NaN price for one or more of this day's blocks")

    afrr_neg = _extract_capacity_market(raw, "afrr_neg", grid)
    if afrr_neg is None:
        return SkipRecord(day, "afrr_neg: missing or NaN price for one or more of this day's blocks")

    return DayPrices(day=day, grid=grid, day_ahead=day_ahead, fcr=fcr, afrr_pos=afrr_pos, afrr_neg=afrr_neg)


def load_price_range(
    start_date: dt.date,
    end_date: dt.date,
    day_ahead_timing: MarketTimingConfig,
    capacity_timing: MarketTimingConfig,
    paths_config: PathsConfig | None = None,
) -> tuple[list[DayPrices], list[SkipRecord]]:
    """Realized (perfect-foresight) prices for each day in the range."""
    pairs, skipped = load_price_pair_range(
        start_date, end_date, day_ahead_timing, capacity_timing, paths_config
    )
    return [pair.settlement for pair in pairs], skipped


def load_price_pair_range(
    start_date: dt.date,
    end_date: dt.date,
    day_ahead_timing: MarketTimingConfig,
    capacity_timing: MarketTimingConfig,
    paths_config: PathsConfig | None = None,
    decision_source: str = "realized",
    forecast_model: str | None = None,
    forecast_quantile: float | None = None,
    da_proxy_model: str | None = None,
) -> tuple[list[DayPricePair], list[SkipRecord]]:
    """Decision and settlement prices for each day in the range.

    da_proxy_model (forecast runs only): also build `reserve_stage_decision`
    for each day, using that model's day-ahead y_pred (at the median)
    in place of forecast_model's. See DayPricePair.

    decision_source="realized" is the perfect-foresight case: both sides
    are the same y_true prices. decision_source="forecast" builds the
    decision side from forecast_model's y_pred at forecast_quantile, while
    settlement stays y_true.

    A day is skipped if EITHER side is incomplete, because a day the
    optimizer can decide on but not settle (or vice versa) cannot be
    honestly valued. For a forecast-driven run the reason is prefixed so
    the skip log says which side failed; for a realized run there is only
    one price source, so the reason is left unprefixed.
    """
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} must not be after end_date {end_date}")
    if decision_source not in ("realized", "forecast"):
        raise ValueError(f"decision_source must be 'realized' or 'forecast', got '{decision_source}'")
    if decision_source == "forecast" and (forecast_model is None or forecast_quantile is None):
        raise ValueError("forecast_model and forecast_quantile are required when decision_source='forecast'")
    if da_proxy_model is not None and decision_source != "forecast":
        raise ValueError("da_proxy_model only applies to forecast-driven runs (a realized run has no stage asymmetry)")

    paths_config = paths_config or load_paths_config()
    raw_true = load_raw_y_true(paths_config)
    raw_decision = (
        load_raw_y_pred(paths_config, forecast_model, forecast_quantile)
        if decision_source == "forecast"
        else raw_true
    )
    raw_proxy = load_raw_y_pred(paths_config, da_proxy_model, MEDIAN_QUANTILE) if da_proxy_model else None

    pairs: list[DayPricePair] = []
    skipped: list[SkipRecord] = []
    day = start_date
    while day <= end_date:
        settlement = load_day_prices(day, raw_true, day_ahead_timing, capacity_timing)
        if isinstance(settlement, SkipRecord):
            reason = settlement.reason if decision_source == "realized" else f"settlement: {settlement.reason}"
            record = SkipRecord(day, reason)
            logger.warning("Skipping %s: %s", record.day, record.reason)
            skipped.append(record)
            day += dt.timedelta(days=1)
            continue

        if decision_source == "realized":
            decision = settlement
        else:
            decision = load_day_prices(day, raw_decision, day_ahead_timing, capacity_timing)
            if isinstance(decision, SkipRecord):
                record = SkipRecord(day, f"decision ({forecast_model} q{forecast_quantile}): {decision.reason}")
                logger.warning("Skipping %s: %s", record.day, record.reason)
                skipped.append(record)
                day += dt.timedelta(days=1)
                continue

        reserve_stage = None
        if raw_proxy is not None:
            proxy_da = _extract_day_ahead(raw_proxy, decision.grid)
            if proxy_da is None:
                record = SkipRecord(
                    day, f"reserve-stage DA proxy ({da_proxy_model}): day_ahead missing or NaN for one or more steps",
                )
                logger.warning("Skipping %s: %s", record.day, record.reason)
                skipped.append(record)
                day += dt.timedelta(days=1)
                continue
            reserve_stage = dataclasses.replace(decision, day_ahead=proxy_da)

        pairs.append(
            DayPricePair(day=day, decision=decision, settlement=settlement, reserve_stage_decision=reserve_stage)
        )
        day += dt.timedelta(days=1)

    return pairs, skipped
