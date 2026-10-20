import datetime as dt

import pandas as pd
import pytest

from src.cli import compute_operational_stats, format_assumptions_banner, main, summarize_revenue
from src.model.rolling import RollingRunResult
from tests.dispatch_helpers import BATTERY



def _fake_result() -> RollingRunResult:
    day1 = dt.date(2026, 6, 1)
    day2 = dt.date(2026, 6, 2)
    steps_day1 = pd.date_range("2026-06-01", periods=4, freq="6h", tz="Europe/Berlin")
    steps_day2 = pd.date_range("2026-06-02", periods=4, freq="6h", tz="Europe/Berlin")

    rows = []
    # Day 1: discharge 1 MWh total (at delta_t=1h for simplicity here),
    # SOC swings from soc_min to soc_min + 1.0
    for i, step in enumerate(steps_day1):
        rows.append({"solve_date": day1, "target_time": step, "variable": "p_dis", "value": 0.25 if i < 4 else 0.0})
        rows.append({"solve_date": day1, "target_time": step, "variable": "p_ch", "value": 0.0})
        rows.append({"solve_date": day1, "target_time": step, "variable": "e", "value": BATTERY.soc_min_mwh + i * 0.3})
    for i, step in enumerate(steps_day2):
        rows.append({"solve_date": day2, "target_time": step, "variable": "p_dis", "value": 0.0})
        rows.append({"solve_date": day2, "target_time": step, "variable": "p_ch", "value": 0.0})
        rows.append({"solve_date": day2, "target_time": step, "variable": "e", "value": BATTERY.soc_min_mwh})

    for r in rows:
        r.update({"solve_id": "fake", "block_id": None, "market": "day_ahead", "unit": "MW", "is_assumption": False})

    long_format = pd.DataFrame(rows)
    daily_revenue = pd.DataFrame(
        [
            {"solve_date": day1, "day_ahead": 100.0, "fcr": 20.0, "afrr_pos": 5.0, "afrr_neg": 5.0, "activation": 70.0,
             "gross": 200.0, "degradation": 0.0, "net": 200.0, "total": 200.0},
            {"solve_date": day2, "day_ahead": 50.0, "fcr": 0.0, "afrr_pos": 0.0, "afrr_neg": 0.0, "activation": 0.0,
             "gross": 50.0, "degradation": 0.0, "net": 50.0, "total": 50.0},
        ]
    )
    return RollingRunResult(solve_id="fake", long_format=long_format, daily_revenue=daily_revenue, skip_log=[])


def test_summarize_revenue_sums_across_days():
    result = _fake_result()
    revenue = summarize_revenue(result)
    assert revenue["total"] == pytest.approx(250.0)
    assert revenue["day_ahead"] == pytest.approx(150.0)
    assert revenue["activation"] == pytest.approx(70.0)


def test_summarize_revenue_empty_result_is_all_zero():
    empty = RollingRunResult(solve_id="empty", long_format=pd.DataFrame(), daily_revenue=pd.DataFrame(), skip_log=[])
    revenue = summarize_revenue(empty)
    assert all(v == 0.0 for v in revenue.values())


def test_compute_operational_stats_cycles_and_utilization():
    result = _fake_result()
    usable_range = BATTERY.soc_max_mwh - BATTERY.soc_min_mwh
    stats = compute_operational_stats(result, BATTERY, delta_t_hours=1.0)

    # Day 1: discharged 4*0.25*1.0 = 1.0 MWh -> cycles = 1.0/usable_range
    # Day 2: discharged 0 -> cycles = 0
    expected_cycles = ((1.0 / usable_range) + 0.0) / 2
    assert stats["cycles_per_day"] == pytest.approx(expected_cycles)

    # Day 1 SOC swing: soc_min+0.9 - soc_min = 0.9 -> utilization = 0.9/usable_range
    # Day 2 SOC swing: 0
    expected_utilization = ((0.9 / usable_range) + 0.0) / 2
    assert stats["avg_soc_utilization"] == pytest.approx(expected_utilization)


def test_format_assumptions_banner_contains_real_flagged_placeholders():
    banner = format_assumptions_banner()
    assert "PLACEHOLDER" in banner
    assert "fcr_hours" in banner
    assert "ASSUMPTIONS IN EFFECT" in banner


def test_main_runs_end_to_end_and_prints_expected_sections(capsys):
    main(["--start-date", "2026-06-15", "--end-date", "2026-06-15"])
    out = capsys.readouterr().out
    assert "ASSUMPTIONS IN EFFECT FOR THIS RUN" in out
    assert "day_ahead_only" in out
    assert "full_sequential" not in out
    assert "full_cooptimized" in out
    assert "BATTERY IN EFFECT FOR THIS RUN" in out
    assert "4.5 MW operational" in out
    assert "gross revenue" in out
    assert "degradation" in out
    assert "net revenue" in out
