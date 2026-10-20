import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from analysis import (  # noqa: E402
    BASELINE_MODEL, capture_comparison, common_days, forecast_error_metrics, model_revenue_table, oracle_capture,
    perfect_foresight_summary, plan_agreement, revenue_split, stacked_bars,
)

D1, D2, D3 = dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 3)


def _components(v: dict) -> dict:
    base = dict(day_ahead=0.0, fcr=0.0, afrr_pos=0.0, afrr_neg=0.0, activation=0.0, degradation=0.0)
    base.update(v)
    base["gross"] = base["day_ahead"] + base["fcr"] + base["afrr_pos"] + base["afrr_neg"] + base["activation"]
    base["net"] = base["gross"] - base["degradation"]
    return base


def _daily(rows: dict[dt.date, dict]) -> pd.DataFrame:
    out = []
    for day, v in rows.items():
        base = _components(v)
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


def test_oracle_capture_is_realized_over_ceiling_on_intersected_days_only():
    # D3 exists only on the ceiling side and must be ignored.
    perfect = _daily({D1: dict(day_ahead=100, fcr=200), D2: dict(day_ahead=100, fcr=200), D3: dict(day_ahead=999, fcr=999)})
    forecast = _daily({D1: dict(day_ahead=50, fcr=190), D2: dict(day_ahead=30, fcr=210)})
    table, n_days = oracle_capture(perfect, forecast)

    assert n_days == 2
    assert (table.loc["day_ahead", "perfect_eur"], table.loc["day_ahead", "realized_eur"]) == (200, 80)
    assert table.loc["day_ahead", "capture_pct"] == pytest.approx(40.0)  # 80 / 200
    assert table.loc["fcr", "capture_pct"] == pytest.approx(100.0)  # 400 of 400
    assert (table.loc["net", "perfect_eur"], table.loc["net", "realized_eur"]) == (600, 480)
    assert table.loc["net", "capture_pct"] == pytest.approx(80.0)


def test_capture_is_undefined_not_zero_when_the_ceiling_base_is_not_positive():
    perfect = _daily({D1: dict(day_ahead=-10, fcr=100)})
    forecast = _daily({D1: dict(day_ahead=-30, fcr=90)})
    table, _ = oracle_capture(perfect, forecast)
    assert np.isnan(table.loc["day_ahead", "capture_pct"])
    assert np.isnan(table.loc["afrr_pos", "capture_pct"])  # 0 base
    assert table.loc["fcr", "capture_pct"] == pytest.approx(90.0)


def test_oracle_capture_without_common_days_returns_an_empty_table():
    table, n_days = oracle_capture(_daily({D1: {}}), _daily({D2: {}}))
    assert n_days == 0 and table["capture_pct"].isna().all()


def test_stacked_bars_negative_day_ahead_does_not_shift_positive_components():
    daily = _daily({D1: dict(day_ahead=-100, fcr=50, afrr_pos=30, afrr_neg=-10, activation=20)})
    bars = stacked_bars(daily)
    bottoms = {c: float(b[0]) for c, (b, _) in bars.items()}
    # Positives stack up from 0 (fcr, afrr_pos, activation); negatives stack down from 0 (day_ahead, afrr_neg).
    assert bottoms == {"day_ahead": 0.0, "fcr": 0.0, "afrr_pos": 50.0, "afrr_neg": -100.0, "activation": 80.0}
    ends = [float(bottom[0] + height[0]) for bottom, height in bars.values()]
    assert max(ends) == 100.0 and min(ends) == -110.0




# ---- forecast_error_metrics: hand-computed ---------------------------------

def _forecasts(rows):
    """rows: (market, model, hour_offset, y_pred, y_true)."""
    t0 = pd.Timestamp("2026-06-01", tz="Europe/Berlin")
    return pd.DataFrame(
        [{"market": m, "model": mod, "target_time": t0 + pd.Timedelta(hours=h), "y_pred": p, "y_true": y}
         for m, mod, h, p, y in rows],
        columns=["market", "model", "target_time", "y_pred", "y_true"],
    )


def test_forecast_error_metrics_hand_computed():
    # truth 10, 20, 30, 40. naive errors: +2, -2, +2, -2 -> MAE 2, RMSE 2.
    # other errors: 0, 0, 0, +4 -> MAE 1, RMSE 2 (sqrt(16 / 4)).
    truth = [10, 20, 30, 40]
    naive = [12, 18, 32, 38]
    other = [10, 20, 30, 44]
    rows = [("day_ahead", "seasonal_naive", h, p, y) for h, (p, y) in enumerate(zip(naive, truth))]
    rows += [("day_ahead", "other", h, p, y) for h, (p, y) in enumerate(zip(other, truth))]
    out = forecast_error_metrics(_forecasts(rows))

    assert out.loc[("day_ahead", "seasonal_naive"), "mae"] == pytest.approx(2.0)
    assert out.loc[("day_ahead", "seasonal_naive"), "rmse"] == pytest.approx(2.0)
    assert out.loc[("day_ahead", "seasonal_naive"), "skill_vs_baseline"] == pytest.approx(0.0)
    assert out.loc[("day_ahead", "other"), "mae"] == pytest.approx(1.0)
    assert out.loc[("day_ahead", "other"), "rmse"] == pytest.approx(2.0)
    assert out.loc[("day_ahead", "other"), "skill_vs_baseline"] == pytest.approx(0.0)
    assert (out["n"] == 4).all()


def test_forecast_error_metrics_skill_is_one_minus_rmse_ratio():
    # baseline RMSE 2. perfect: RMSE 0 -> skill 1. bad: RMSE 4 -> skill -1.
    rows = [("fcr", "seasonal_naive", h, p, 0) for h, p in enumerate([2, -2, 2, -2])]
    rows += [("fcr", "perfect", h, 0, 0) for h in range(4)]
    rows += [("fcr", "bad", h, p, 0) for h, p in enumerate([4, -4, 4, -4])]
    out = forecast_error_metrics(_forecasts(rows))
    assert out.loc[("fcr", "perfect"), "skill_vs_baseline"] == pytest.approx(1.0)
    assert out.loc[("fcr", "bad"), "skill_vs_baseline"] == pytest.approx(-1.0)


def test_forecast_error_metrics_scores_every_model_on_the_common_timestamps_only():
    # "gappy" has no forecast at hour 3 (like seasonal_naive on a DST day); the
    # huge miss "full" makes there must not count against it.
    rows = [("day_ahead", "seasonal_naive", h, 10, 10) for h in range(4)]
    rows += [("day_ahead", "gappy", h, 10, 10) for h in range(3)] + [("day_ahead", "gappy", 3, float("nan"), 10)]
    rows += [("day_ahead", "full", h, 10, 10) for h in range(3)] + [("day_ahead", "full", 3, 1000, 10)]
    out = forecast_error_metrics(_forecasts(rows))
    assert (out["n"] == 3).all()
    assert out.loc[("day_ahead", "full"), "mae"] == pytest.approx(0.0)


def test_forecast_error_metrics_without_baseline_or_data():
    rows = [("day_ahead", "only", h, 11, 10) for h in range(2)]
    out = forecast_error_metrics(_forecasts(rows))
    assert out.loc[("day_ahead", "only"), "mae"] == pytest.approx(1.0)
    assert np.isnan(out.loc[("day_ahead", "only"), "skill_vs_baseline"])
    assert forecast_error_metrics(_forecasts([])).empty


# ---- model_revenue_table -----------------------------------------------------

def test_model_revenue_table_uses_days_common_to_perfect_and_every_model():
    perfect = _daily({D1: dict(day_ahead=100), D2: dict(day_ahead=100), D3: dict(day_ahead=100)})
    a = _daily({D1: dict(day_ahead=60), D2: dict(day_ahead=60)})
    # b lacks D2, so D2 must drop out for BOTH models.
    b = _daily({D1: dict(day_ahead=90), D3: dict(day_ahead=90)})
    table, perfect_net, n_days = model_revenue_table(perfect, {"a": a, "b": b})

    assert n_days == 1 and perfect_net == 100
    assert table.loc["a", "net_eur"] == 60
    assert table.loc["a", "capture_pct"] == pytest.approx(60.0)
    assert "cost_of_error_eur" not in table.columns
    assert table.loc["b", "capture_pct"] == pytest.approx(90.0)
    assert table.loc["b", "net_eur"] == 90


def _plan(days_plan: dict[dt.date, dict[str, dict[int, float]]]) -> pd.DataFrame:
    """Long format p_ch/p_dis rows: {day: {"p_ch": {quarter: MW}, "p_dis": {...}}}, 96 quarters a day."""
    rows = []
    for day, variables in days_plan.items():
        times = pd.date_range(pd.Timestamp(day), periods=96, freq="15min")
        for variable, by_quarter in variables.items():
            for q, t in enumerate(times):
                rows.append({"solve_date": day, "target_time": t, "variable": variable, "value": by_quarter.get(q, 0.0)})
    return pd.DataFrame(rows)


def test_plan_agreement_on_a_hand_made_two_day_case():
    ceiling = _plan({d: {"p_ch": dict.fromkeys(range(8, 12), 2.0), "p_dis": dict.fromkeys(range(60, 64), 2.0)} for d in (D1, D2)})
    model = _plan({
        # D1: charge shifted 2 quarters (inside the 30 min rule), discharge 10 quarters late (outside).
        D1: {"p_ch": dict.fromkeys(range(10, 14), 2.0), "p_dis": dict.fromkeys(range(70, 74), 2.0)},
        # D2: charge doubled to 4 MW (4 MWh vs 2), discharge exact.
        D2: {"p_ch": dict.fromkeys(range(8, 12), 4.0), "p_dis": dict.fromkeys(range(60, 64), 2.0)},
    })
    got = plan_agreement(model, ceiling, [D1, D2])

    assert got["charge_timing_pct"] == pytest.approx(100.0)
    assert got["discharge_timing_pct"] == pytest.approx(50.0)  # 4 of 8 quarters
    assert got["charge_amount_pct"] == pytest.approx(50.0)  # ceiling 2 + 2 MWh, gaps 0 + 2: 1 - 2/4
    assert got["discharge_amount_pct"] == pytest.approx(100.0)
    assert got["charge_mwh_per_day"] == pytest.approx(3.0)  # (2 + 4) / 2 days
    assert got["wrong_way_qh_per_day"] == 0.0


def test_plan_agreement_counts_quarters_that_act_against_the_ceiling():
    ceiling = _plan({D1: {"p_ch": {8: 2.0}, "p_dis": {60: 2.0}}})
    model = _plan({D1: {"p_ch": {60: 2.0}, "p_dis": {8: 2.0}}})
    assert plan_agreement(model, ceiling, [D1])["wrong_way_qh_per_day"] == 2.0


def test_model_revenue_table_value_vs_baseline_is_relative_to_the_baseline_models_net():
    perfect = _daily({D1: dict(day_ahead=100), D2: dict(day_ahead=100)})
    baseline = _daily({D1: dict(day_ahead=70), D2: dict(day_ahead=70)})
    better = _daily({D1: dict(day_ahead=90), D2: dict(day_ahead=90)})
    table, _, _ = model_revenue_table(perfect, {BASELINE_MODEL: baseline, "better": better})

    assert table.loc[BASELINE_MODEL, "value_vs_baseline_eur"] == pytest.approx(0.0)
    assert table.loc["better", "value_vs_baseline_eur"] == pytest.approx(40.0)  # 180 - 140


def test_model_revenue_table_value_vs_baseline_is_nan_without_the_baseline_model():
    perfect = _daily({D1: dict(day_ahead=100)})
    a = _daily({D1: dict(day_ahead=60)})
    table, _, _ = model_revenue_table(perfect, {"a": a})
    assert np.isnan(table.loc["a", "value_vs_baseline_eur"])


def test_model_revenue_table_with_no_common_days():
    table, perfect_net, n_days = model_revenue_table(_daily({D1: {}}), {"a": _daily({D2: {}})})
    assert n_days == 0 and np.isnan(perfect_net) and table["capture_pct"].isna().all()


# ---- perfect_foresight_summary ------------------------------------------------

def test_perfect_foresight_summary_reads_run_meta_columns_and_sums_the_market_split():
    catalogue = pd.DataFrame([
        {"solve_id": "p1", "strategy": "day_ahead_only", "decision_source": "realized", "n_days_solved": 2,
         "gross_revenue_eur": 200.0, "degradation_cost_eur": 20.0, "net_revenue_eur": 180.0, "equivalent_full_cycles": 3.0},
        {"solve_id": "f1", "strategy": "day_ahead_only", "decision_source": "forecast", "n_days_solved": 2,
         "gross_revenue_eur": 999.0, "degradation_cost_eur": 999.0, "net_revenue_eur": 999.0, "equivalent_full_cycles": 999.0},
    ])
    daily_by_id = {"p1": _daily({D1: dict(day_ahead=120, fcr=80), D2: dict(day_ahead=0, fcr=0)})}
    out = perfect_foresight_summary(catalogue, daily_by_id)

    assert list(out.index) == ["day_ahead_only"]  # the forecast row is excluded
    row = out.loc["day_ahead_only"]
    assert (row["gross_eur"], row["wear_eur"], row["net_eur"], row["cycles_per_day"]) == (200.0, 20.0, 180.0, 1.5)
    assert (row["day_ahead_eur"], row["fcr_eur"]) == (120.0, 80.0)
    assert row["activation_eur"] == 0.0


def test_perfect_foresight_summary_handles_a_missing_daily_frame():
    catalogue = pd.DataFrame([
        {"solve_id": "p1", "strategy": "full_cooptimized", "decision_source": "realized", "n_days_solved": 10,
         "gross_revenue_eur": 50.0, "degradation_cost_eur": 5.0, "net_revenue_eur": 45.0, "equivalent_full_cycles": 5.0},
    ])
    out = perfect_foresight_summary(catalogue, {})
    assert out.loc["full_cooptimized", "net_eur"] == 45.0
    assert out.loc["full_cooptimized", "cycles_per_day"] == pytest.approx(0.5)
    assert np.isnan(out.loc["full_cooptimized", "day_ahead_eur"])


# ---- capture_comparison -------------------------------------------------------

def test_capture_comparison_keeps_only_models_both_sides_have_a_value_for():
    old = pd.DataFrame({"capture_pct": [76.5, 87.5]}, index=["linear_lags", "lgbm_point"])
    new = pd.DataFrame({"capture_pct": [94.1, 93.1, 93.3]}, index=["linear_lags", "lgbm_point", "seasonal_naive"])
    old_out, new_out = capture_comparison(old, new)
    assert old_out == {"linear_lags": 76.5, "lgbm_point": 87.5}
    assert new_out == {"linear_lags": 94.1, "lgbm_point": 93.1}


def test_capture_comparison_drops_nan_before_intersecting():
    old = pd.DataFrame({"capture_pct": [76.5, np.nan]}, index=["linear_lags", "lgbm_point"])
    new = pd.DataFrame({"capture_pct": [94.1, 93.1]}, index=["linear_lags", "lgbm_point"])
    old_out, new_out = capture_comparison(old, new)
    assert old_out == {"linear_lags": 76.5}
    assert new_out == {"linear_lags": 94.1}
