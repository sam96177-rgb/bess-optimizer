"""Builds, for a single calendar day, the 15-minute day-ahead-native time
grid and the 4-hour capacity-block boundaries, and maps each grid step to
its block. Mirrors bess-market-forecast's src/backtest/cutoffs.py
conventions exactly (SPEC.md §1: reuse the convention, don't invent a
second one) — day_ahead uses "variable_period_count" (the day genuinely
has 92/96/100 quarter-hours around a DST transition), fcr/afrr use
"fixed_period_count" (always 6 nominal blocks, boundaries placed by
absolute-time — not wall-clock — arithmetic from local midnight, so the
last block's end can spill past next calendar day's local midnight on a
short day).

All boundary arithmetic uses pd.Timestamp/pd.Timedelta (day_ahead's day
end also uses pd.DateOffset for the calendar-aware day boundary) — never
stdlib datetime/timedelta on a tz-attached value, which does wall-clock
arithmetic and silently mishandles the DST gap.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from src.config import MarketTimingConfig

FIFTEEN_MIN = pd.Timedelta(minutes=15)


def _localize_midnight(day: dt.date, timing: MarketTimingConfig) -> pd.Timestamp:
    if timing.delivery_start_local != dt.time(0, 0):
        raise ValueError(
            f"{timing.name}: grid.py assumes delivery_start_local == 00:00, "
            f"got {timing.delivery_start_local}. Stop and re-check rather than "
            "silently building on a changed convention."
        )
    naive = dt.datetime.combine(day, timing.delivery_start_local)
    return pd.Timestamp(naive).tz_localize(
        timing.timezone, ambiguous=False, nonexistent="shift_forward"
    )


def build_day_ahead_steps(day: dt.date, timing: MarketTimingConfig) -> pd.DatetimeIndex:
    """The day's own 15-minute step-start timestamps: 92 on a spring-forward
    DST day, 100 on a fall-back day, 96 otherwise.
    """
    if timing.dst_convention != "variable_period_count":
        raise ValueError(
            f"{timing.name}: build_day_ahead_steps requires dst_convention == "
            f"'variable_period_count', got '{timing.dst_convention}'"
        )
    day_start = _localize_midnight(day, timing)
    day_end = day_start + pd.DateOffset(days=1)
    product_freq = pd.Timedelta(hours=timing.product_length_hours)
    boundaries = pd.date_range(day_start, day_end, freq=product_freq, inclusive="left")
    return boundaries


@dataclass(frozen=True)
class BlockBoundary:
    index: int
    start: pd.Timestamp
    end: pd.Timestamp


def build_capacity_blocks(day: dt.date, timing: MarketTimingConfig) -> list[BlockBoundary]:
    """The day's 6 nominal 4-hour capacity blocks. Boundaries are absolute-
    time offsets from local midnight, so on a DST day the wall-clock
    boundaries shift (e.g. spring-forward's block 0 ends at local 05:00,
    not 04:00) and the final block's end can land in the next calendar day.
    """
    if timing.dst_convention != "fixed_period_count":
        raise ValueError(
            f"{timing.name}: build_capacity_blocks requires dst_convention == "
            f"'fixed_period_count', got '{timing.dst_convention}'"
        )
    raw_steps = 24 / timing.product_length_hours
    n_blocks = round(raw_steps)
    if abs(raw_steps - n_blocks) > 1e-9:
        raise ValueError(
            f"{timing.name}: product_length_hours={timing.product_length_hours} must evenly divide 24"
        )
    day_start = _localize_midnight(day, timing)
    product_freq = pd.Timedelta(hours=timing.product_length_hours)
    return [
        BlockBoundary(index=i, start=day_start + i * product_freq, end=day_start + (i + 1) * product_freq)
        for i in range(n_blocks)
    ]


def map_steps_to_blocks(
    steps: pd.DatetimeIndex,
    blocks: list[BlockBoundary],
    extend_last_block_over_long_day: bool = False,
) -> list[int]:
    """For each 15-min step, the index of the capacity block it falls in
    (block.start <= step < block.end). Raises if a step falls outside every
    block, except in one explicitly-enabled case:

    extend_last_block_over_long_day: on a DST fall-back day (25h) the 6
    nominal blocks, anchored to 24 absolute hours from local midnight, end
    an hour before the day's local end, leaving that trailing hour's steps
    uncovered. With this flag those steps (and only steps at/after the last
    block's end) join the LAST block, i.e. the last block's commitment is
    held through the day's end. ASSUMPTION, unverified: no real published
    fall-back-day data exists yet (data starts 2026-03-21; the day is
    2026-10-25), so how regelleistung actually cuts the 25h day is not
    confirmed. The block's own start/end (its nominal 4h, used for
    capacity-payment sizing) are left unchanged.
    """
    block_indices = []
    for step in steps:
        match = [b.index for b in blocks if b.start <= step < b.end]
        if not match and extend_last_block_over_long_day and blocks and step >= blocks[-1].end:
            match = [blocks[-1].index]
        if not match:
            raise ValueError(
                f"Step {step} does not fall within any of the {len(blocks)} capacity blocks "
                f"(last block ends {blocks[-1].end if blocks else 'n/a'}). Refusing to guess "
                "which block it belongs to."
            )
        block_indices.append(match[0])
    return block_indices


@dataclass(frozen=True)
class DayGrid:
    day: dt.date
    steps: pd.DatetimeIndex
    blocks: list[BlockBoundary]
    step_block_index: list[int]
    delta_t_hours: float


def build_day_grid(
    day: dt.date,
    day_ahead_timing: MarketTimingConfig,
    capacity_timing: MarketTimingConfig,
) -> DayGrid:
    """The full grid a single-day MILP needs: this day's 15-min steps, its
    6 nominal capacity blocks, and the step->block mapping.
    """
    steps = build_day_ahead_steps(day, day_ahead_timing)
    blocks = build_capacity_blocks(day, capacity_timing)
    step_block_index = map_steps_to_blocks(steps, blocks, extend_last_block_over_long_day=True)
    return DayGrid(
        day=day,
        steps=steps,
        blocks=blocks,
        step_block_index=step_block_index,
        delta_t_hours=day_ahead_timing.product_length_hours,
    )
