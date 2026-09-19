import datetime as dt

import pandas as pd
import pytest

from src.config import load_market_timing_configs
from src.data.grid import (
    BlockBoundary,
    build_capacity_blocks,
    build_day_ahead_steps,
    build_day_grid,
    map_steps_to_blocks,
)

TIMINGS = load_market_timing_configs()
DAY_AHEAD = TIMINGS["day_ahead"]
FCR = TIMINGS["fcr"]

NORMAL_DAY = dt.date(2026, 3, 4)  # confirmed present in results/, no DST nearby
DST_SPRING_FORWARD_DAY = dt.date(2026, 3, 29)  # confirmed 92 quarter-hours in Milestone 0
NEXT_DAY = dt.date(2026, 3, 30)
DST_FALL_BACK_DAY = dt.date(2026, 10, 25)  # 25h day; no real data exists yet


# ---- day-ahead 15-min steps --------------------------------------------

def test_normal_day_has_96_quarter_hour_steps():
    steps = build_day_ahead_steps(NORMAL_DAY, DAY_AHEAD)
    assert len(steps) == 96
    assert steps[0] == pd.Timestamp("2026-03-04 00:00", tz="Europe/Berlin")
    assert steps[-1] == pd.Timestamp("2026-03-04 23:45", tz="Europe/Berlin")
    assert list(pd.Series(steps).diff().dropna().unique()) == [pd.Timedelta(minutes=15)]


def test_dst_spring_forward_day_has_92_quarter_hour_steps():
    steps = build_day_ahead_steps(DST_SPRING_FORWARD_DAY, DAY_AHEAD)
    assert len(steps) == 92
    assert steps[0] == pd.Timestamp("2026-03-29 00:00", tz="Europe/Berlin")
    # 01:45 CET is the last step before the 02:00->03:00 gap; 02:00-02:45
    # local times don't exist that day, so no step should land there.
    assert pd.Timestamp("2026-03-29 01:45", tz="Europe/Berlin") in steps
    assert not any(s.hour == 2 for s in steps)
    assert pd.Timestamp("2026-03-29 03:00", tz="Europe/Berlin") in steps
    assert steps[-1] == pd.Timestamp("2026-03-29 23:45", tz="Europe/Berlin")


def test_day_after_dst_is_back_to_96_steps():
    steps = build_day_ahead_steps(NEXT_DAY, DAY_AHEAD)
    assert len(steps) == 96


# ---- capacity blocks -----------------------------------------------------

def test_normal_day_has_6_four_hour_blocks():
    blocks = build_capacity_blocks(NORMAL_DAY, FCR)
    assert len(blocks) == 6
    assert blocks[0].start == pd.Timestamp("2026-03-04 00:00", tz="Europe/Berlin")
    assert blocks[0].end == pd.Timestamp("2026-03-04 04:00", tz="Europe/Berlin")
    assert blocks[-1].end == pd.Timestamp("2026-03-05 00:00", tz="Europe/Berlin")


def test_dst_day_block_boundaries_shift_by_absolute_time():
    blocks = build_capacity_blocks(DST_SPRING_FORWARD_DAY, FCR)
    assert len(blocks) == 6
    # Matches markets.yaml's own documented example: last block's end lands
    # at 2026-03-30 01:00, not 00:00 -- and confirmed against the real
    # NaN y_pred target_times found in Milestone 0 (05,09,13,17,21:00).
    expected_ends = ["05:00", "09:00", "13:00", "17:00", "21:00"]
    for block, expected in zip(blocks[:-1], expected_ends):
        assert block.end == pd.Timestamp(f"2026-03-29 {expected}", tz="Europe/Berlin")
    assert blocks[-1].start == pd.Timestamp("2026-03-29 21:00", tz="Europe/Berlin")
    assert blocks[-1].end == pd.Timestamp("2026-03-30 01:00", tz="Europe/Berlin")


# ---- step -> block mapping -------------------------------------------------

def test_normal_day_maps_16_steps_per_block():
    steps = build_day_ahead_steps(NORMAL_DAY, DAY_AHEAD)
    blocks = build_capacity_blocks(NORMAL_DAY, FCR)
    mapping = map_steps_to_blocks(steps, blocks)
    assert len(mapping) == 96
    counts = pd.Series(mapping).value_counts().sort_index()
    assert counts.tolist() == [16] * 6


def test_dst_day_block_5_is_truncated_within_this_days_own_grid():
    steps = build_day_ahead_steps(DST_SPRING_FORWARD_DAY, DAY_AHEAD)
    blocks = build_capacity_blocks(DST_SPRING_FORWARD_DAY, FCR)
    mapping = map_steps_to_blocks(steps, blocks)
    assert len(mapping) == 92
    counts = pd.Series(mapping).value_counts().sort_index()
    # Block 0 absorbs the lost hour (its absolute 4h window still holds 16
    # real quarter-hour steps); blocks 1-4 are unaffected; block 5 only
    # gets this calendar day's own steps (21:00-23:45), the rest of its
    # absolute window belongs to the next day's grid.
    assert counts.tolist() == [16, 16, 16, 16, 16, 12]


def test_dst_fall_back_day_has_100_steps_and_trailing_hour_joins_last_block():
    steps = build_day_ahead_steps(DST_FALL_BACK_DAY, DAY_AHEAD)
    blocks = build_capacity_blocks(DST_FALL_BACK_DAY, FCR)
    assert len(steps) == 100
    # 6 nominal blocks anchored to 24 absolute hours end at 23:00 local.
    assert blocks[-1].end == pd.Timestamp("2026-10-25 23:00", tz="Europe/Berlin")
    with pytest.raises(ValueError, match="does not fall within any"):
        map_steps_to_blocks(steps, blocks)

    mapping = map_steps_to_blocks(steps, blocks, extend_last_block_over_long_day=True)
    counts = pd.Series(mapping).value_counts().sort_index()
    assert counts.tolist() == [16, 16, 16, 16, 16, 20]
    assert mapping == sorted(mapping)  # blocks stay contiguous in time


def test_extend_flag_does_not_swallow_steps_before_the_first_block():
    blocks = [
        BlockBoundary(
            index=0,
            start=pd.Timestamp("2026-03-04 04:00", tz="Europe/Berlin"),
            end=pd.Timestamp("2026-03-04 08:00", tz="Europe/Berlin"),
        )
    ]
    steps = pd.DatetimeIndex([pd.Timestamp("2026-03-04 03:00", tz="Europe/Berlin")])
    with pytest.raises(ValueError, match="does not fall within any"):
        map_steps_to_blocks(steps, blocks, extend_last_block_over_long_day=True)


def test_build_day_grid_fall_back_day_end_to_end():
    grid = build_day_grid(DST_FALL_BACK_DAY, DAY_AHEAD, FCR)
    assert len(grid.steps) == 100
    assert len(grid.step_block_index) == 100
    assert max(grid.step_block_index) == 5


def test_map_steps_to_blocks_raises_on_uncovered_step():
    blocks = [
        BlockBoundary(
            index=0,
            start=pd.Timestamp("2026-03-04 00:00", tz="Europe/Berlin"),
            end=pd.Timestamp("2026-03-04 04:00", tz="Europe/Berlin"),
        )
    ]
    steps = pd.DatetimeIndex([pd.Timestamp("2026-03-04 05:00", tz="Europe/Berlin")])
    with pytest.raises(ValueError, match="does not fall within any"):
        map_steps_to_blocks(steps, blocks)


# ---- combined day grid -----------------------------------------------------

def test_build_day_grid_normal_day():
    grid = build_day_grid(NORMAL_DAY, DAY_AHEAD, FCR)
    assert len(grid.steps) == 96
    assert len(grid.blocks) == 6
    assert len(grid.step_block_index) == 96
    assert grid.delta_t_hours == 0.25


def test_build_day_grid_dst_day_end_to_end():
    grid = build_day_grid(DST_SPRING_FORWARD_DAY, DAY_AHEAD, FCR)
    assert len(grid.steps) == 92
    assert len(grid.step_block_index) == 92
    assert max(grid.step_block_index) == 5
