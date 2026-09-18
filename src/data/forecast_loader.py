"""Reads bess-market-forecast's results/*.parquet — y_true (realized
prices) only, read-only (SPEC.md Hard Rule 3) — and aligns each day's
prices onto that day's grid (src/data/grid.py). A day with any missing or
NaN price in any of the four markets is skipped and logged with a reason;
it is never zero-filled or interpolated (SPEC.md Hard Rule 1). This is
also how the known 2026-09-13 hourly-instead-of-15-minute day gets
handled: it isn't special-cased by date, it's just a day where day_ahead's
15-min steps don't all have a value, so the same general check skips it.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import pandas as pd

from src.config import MarketTimingConfig, PathsConfig, load_paths_config
from src.data.grid import DayGrid, build_day_grid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DayPrices:
    day: dt.date
    grid: DayGrid
    day_ahead: pd.Series  # indexed by grid.steps, EUR/MWh
    fcr: pd.Series  # indexed by block index (grid.blocks[i].index)
    afrr_pos: pd.Series
    afrr_neg: pd.Series


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
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} must not be after end_date {end_date}")

    paths_config = paths_config or load_paths_config()
    raw = load_raw_y_true(paths_config)

    days_ok: list[DayPrices] = []
    skipped: list[SkipRecord] = []
    day = start_date
    while day <= end_date:
        result = load_day_prices(day, raw, day_ahead_timing, capacity_timing)
        if isinstance(result, SkipRecord):
            logger.warning("Skipping %s: %s", result.day, result.reason)
            skipped.append(result)
        else:
            days_ok.append(result)
        day += dt.timedelta(days=1)

    return days_ok, skipped
