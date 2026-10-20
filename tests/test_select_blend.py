"""pipeline/03_select_blend.py (PLAN_V6_REVISION.md V6-4): per_market_best is
picked by lowest MAE on each market's VALIDATION window only, never the test
window -- otherwise the blend choice would leak test-window information into
a comparison it then takes part in.
"""
import datetime as dt
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from src.config import StudyConfig, StudyWindow

_SCRIPT = Path(__file__).resolve().parents[1] / "pipeline" / "03_select_blend.py"
_spec = importlib.util.spec_from_file_location("select_blend_script", _SCRIPT)
select_blend_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(select_blend_script)

VALIDATION = StudyWindow(start=dt.date(2026, 1, 1), end=dt.date(2026, 1, 10))
TEST = StudyWindow(start=dt.date(2026, 3, 1), end=dt.date(2026, 3, 10))


def _study(models=("seasonal_naive", "linear_lags", "lgbm_point")) -> StudyConfig:
    return StudyConfig(
        name="v6",
        test_window={m: TEST for m in select_blend_script.MARKETS},
        models=list(models),
        strategies=["day_ahead_only"],
        blends=["per_market_best"],
        validation_window={m: VALIDATION for m in select_blend_script.MARKETS},
    )


def _forecast_rows(market: str, model: str, day: dt.date, y_pred: float, y_true: float = 100.0):
    return {
        "market": market, "model": model,
        "target_time": pd.Timestamp(day, tz="Europe/Berlin"),
        "y_pred": y_pred, "y_true": y_true,
    }


def test_select_blend_picks_the_lowest_mae_model_per_market_on_validation_dates_only():
    rows = []
    for market in select_blend_script.MARKETS:
        for day_offset in range(3):
            day = VALIDATION.start + dt.timedelta(days=day_offset)
            rows.append(_forecast_rows(market, "seasonal_naive", day, y_pred=90.0))  # |err|=10
            rows.append(_forecast_rows(market, "linear_lags", day, y_pred=95.0))  # |err|=5 -- best
            rows.append(_forecast_rows(market, "lgbm_point", day, y_pred=80.0))  # |err|=20
            # A test-window row where linear_lags is deliberately bad -- must be ignored.
            test_day = TEST.start + dt.timedelta(days=day_offset)
            rows.append(_forecast_rows(market, "linear_lags", test_day, y_pred=0.0))
    forecasts = pd.DataFrame(rows)

    per_market_best, mae_by_market = select_blend_script.select_blend(_study(), forecasts)

    assert per_market_best == {m: "linear_lags" for m in select_blend_script.MARKETS}
    for market in select_blend_script.MARKETS:
        assert mae_by_market[market]["linear_lags"] == pytest.approx(5.0)
        assert mae_by_market[market]["seasonal_naive"] == pytest.approx(10.0)
        assert mae_by_market[market]["lgbm_point"] == pytest.approx(20.0)


def test_select_blend_picks_afrr_pos_and_afrr_neg_independently():
    rows = []
    for day_offset in range(3):
        day = VALIDATION.start + dt.timedelta(days=day_offset)
        for market, winner, loser in [("afrr_pos", "lgbm_point", "linear_lags"), ("afrr_neg", "linear_lags", "lgbm_point")]:
            rows.append(_forecast_rows(market, winner, day, y_pred=99.0))  # |err|=1
            rows.append(_forecast_rows(market, loser, day, y_pred=50.0))  # |err|=50
            rows.append(_forecast_rows(market, "seasonal_naive", day, y_pred=70.0))
        for market in ("day_ahead", "fcr"):
            rows.append(_forecast_rows(market, "seasonal_naive", day, y_pred=99.0))
            rows.append(_forecast_rows(market, "linear_lags", day, y_pred=99.0))
            rows.append(_forecast_rows(market, "lgbm_point", day, y_pred=99.0))
    forecasts = pd.DataFrame(rows)

    per_market_best, _ = select_blend_script.select_blend(_study(), forecasts)
    assert per_market_best["afrr_pos"] == "lgbm_point"
    assert per_market_best["afrr_neg"] == "linear_lags"


def test_select_blend_raises_without_a_validation_window():
    study = StudyConfig(
        name="v6", test_window={"day_ahead": TEST}, models=["seasonal_naive"],
        strategies=[], blends=[], validation_window=None,
    )
    with pytest.raises(SystemExit, match="validation_window"):
        select_blend_script.select_blend(study, pd.DataFrame())


def test_select_blend_raises_when_a_market_has_no_validation_forecasts():
    forecasts = pd.DataFrame([_forecast_rows("day_ahead", "seasonal_naive", VALIDATION.start, 90.0)])
    with pytest.raises(SystemExit, match="fcr"):
        select_blend_script.select_blend(_study(), forecasts)


def test_render_forecast_blends_yaml_is_parseable_and_marks_the_pick():
    import yaml

    study = _study()
    per_market_best = {m: "linear_lags" for m in select_blend_script.MARKETS}
    mae_by_market = {m: {"seasonal_naive": 10.0, "linear_lags": 5.0, "lgbm_point": 20.0} for m in select_blend_script.MARKETS}
    text = select_blend_script.render_forecast_blends_yaml(study, per_market_best, mae_by_market)

    assert "<- picked" in text
    parsed = yaml.safe_load(text)
    assert parsed["per_market_best"] == per_market_best
