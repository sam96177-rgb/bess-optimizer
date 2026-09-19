import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from analysis import (  # noqa: E402
    NO_PROXY_LABEL, common_days, oracle_capture, proxy_label, revenue_split, sequential_comparison, twin_headlines,
)

D1, D2, D3 = dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 3)


def _daily(rows: dict[dt.date, dict]) -> pd.DataFrame:
    out = []
    for day, v in rows.items():
        base = dict(day_ahead=0.0, fcr=0.0, afrr_pos=0.0, afrr_neg=0.0, activation=0.0, degradation=0.0)
        base.update(v)
        base["gross"] = base["day_ahead"] + base["fcr"] + base["afrr_pos"] + base["afrr_neg"] + base["activation"]
        base["net"] = base["gross"] - base["degradation"]
        base["solve_date"] = day
        out.append(base)
    return pd.DataFrame(out)


def test_common_days_is_the_intersection_and_empty_when_any_side_is_empty():
    a = _daily({D1: {}, D2: {}})
    b = _daily({D2: {}, D3: {}})
    assert common_days(a, b) == [D2]
    assert common_days(a, pd.DataFrame()) == []


def test_revenue_split_hand_computed():
    daily = _daily({D1: dict(day_ahead=100, fcr=50, degradation=10), D2: dict(day_ahead=40, activation=20, degradation=5)})
    assert revenue_split(daily) == {"gross": 210.0, "degradation": 15.0, "net": 195.0}
    assert revenue_split(daily, [D1]) == {"gross": 150.0, "degradation": 10.0, "net": 140.0}
    assert revenue_split(pd.DataFrame()) == {"gross": 0.0, "degradation": 0.0, "net": 0.0}


def test_oracle_capture_per_market_on_intersected_days_only():
    # D3 exists only on the perfect side and must be ignored.
    perfect = _daily({D1: dict(day_ahead=100, fcr=200), D2: dict(day_ahead=100, fcr=200), D3: dict(day_ahead=999, fcr=999)})
    forecast = _daily({D1: dict(day_ahead=50, fcr=190), D2: dict(day_ahead=30, fcr=210)})
    table, n_days = oracle_capture(perfect, forecast)

    assert n_days == 2
    assert (table.loc["day_ahead", "perfect_eur"], table.loc["day_ahead", "forecast_eur"]) == (200, 80)
    assert table.loc["day_ahead", "capture_pct"] == pytest.approx(40.0)
    assert table.loc["fcr", "capture_pct"] == pytest.approx(100.0)  # 400 of 400
    assert (table.loc["net", "perfect_eur"], table.loc["net", "forecast_eur"]) == (600, 480)
    assert table.loc["net", "gap_eur"] == pytest.approx(120.0)
    assert table.loc["net", "capture_pct"] == pytest.approx(80.0)


def test_capture_is_undefined_not_zero_when_the_perfect_base_is_not_positive():
    perfect = _daily({D1: dict(day_ahead=-10, fcr=100)})
    forecast = _daily({D1: dict(day_ahead=-30, fcr=90)})
    table, _ = oracle_capture(perfect, forecast)
    assert np.isnan(table.loc["day_ahead", "capture_pct"])
    assert np.isnan(table.loc["afrr_pos", "capture_pct"])  # 0 base
    assert table.loc["fcr", "capture_pct"] == pytest.approx(90.0)


def test_oracle_capture_without_common_days_returns_an_empty_table():
    table, n_days = oracle_capture(_daily({D1: {}}), _daily({D2: {}}))
    assert n_days == 0 and table["capture_pct"].isna().all()


def test_sequential_comparison_hand_computed():
    perfect = _daily({D1: dict(fcr=1000), D2: dict(fcr=1000), D3: dict(fcr=5000)})
    simultaneous = _daily({D1: dict(fcr=800), D2: dict(fcr=700)})
    sequential = _daily({D1: dict(fcr=790), D2: dict(fcr=720), D3: dict(fcr=1)})
    out = sequential_comparison(perfect, simultaneous, sequential)
    assert out["n_days"] == 2
    assert out["perfect_eur"] == 2000 and out["simultaneous_eur"] == 1500 and out["sequential_eur"] == 1510
    assert out["sequential_minus_simultaneous_eur"] == 10
    assert out["simultaneous_capture_pct"] == pytest.approx(75.0)
    assert out["sequential_capture_pct"] == pytest.approx(75.5)
    assert sequential_comparison(perfect, simultaneous, _daily({D3: {}})) is None


def test_proxy_label():
    assert proxy_label(np.nan) == NO_PROXY_LABEL and proxy_label(None) == NO_PROXY_LABEL
    assert proxy_label("seasonal_naive") == "seasonal_naive"


def test_twin_headlines_hand_computed():
    summary = pd.DataFrame(
        {
            "solve_date": [D1, D1, D2, D2],
            "draw": [0, 1, 0, 1],
            "n_soc_violation_steps": [0, 2, 0, 0],
            "n_power_violation_steps": [0, 0, 0, 0],
            "dispatch_shortfall_mwh": [0.0, 1.0, 0.0, 3.0],
            "activation_shortfall_up_mwh": [0.0, 0.5, 0.0, 0.0],
            "activation_shortfall_down_mwh": [0.0, 0.0, 0.5, 1.0],
            "activation_delivered_mwh": [10.0, 9.0, 10.0, 9.0],
            "activation_expected_mwh": [10.0, 10.0, 10.0, 10.0],
        }
    )
    h = twin_headlines(summary)
    assert h["n_days"] == 2 and h["n_draws"] == 2
    assert h["share_replays_with_violation"] == pytest.approx(0.25)
    assert h["mean_dispatch_shortfall_mwh_per_day"] == pytest.approx(1.0)
    assert h["mean_activation_shortfall_mwh_per_day"] == pytest.approx(0.5)
    assert (h["activation_delivered_mwh"], h["activation_expected_mwh"]) == (38.0, 40.0)
    assert twin_headlines(pd.DataFrame()) == {}
