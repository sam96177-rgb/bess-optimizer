import datetime as dt

import pytest
import yaml

from src.config import (
    AssumptionsConfig,
    BatteryConfig,
    ForecastBlendsConfig,
    MarketPriceBasisConfig,
    PathsConfig,
    SolverConfig,
    StudyConfig,
    StudyWindow,
    load_assumptions_config,
    load_battery_config,
    load_forecast_blends_config,
    load_market_price_basis_config,
    load_paths_config,
    load_solver_config,
    load_study_config,
)


# ---- battery.yaml -----------------------------------------------------

def test_load_battery_config_reads_real_file():
    cfg = load_battery_config()
    assert isinstance(cfg, BatteryConfig)
    assert cfg.power_nominal_mw == 5.0
    assert cfg.power_derating_factor == 0.9
    assert cfg.power_mw == pytest.approx(4.5)
    assert cfg.energy_mwh == 10.0
    assert cfg.round_trip_efficiency == 0.90
    assert cfg.soc_min_fraction == 0.10
    assert cfg.soc_max_fraction == 0.90
    assert cfg.aux_load_mw == 0.05
    assert cfg.degradation_cost_eur_mwh == 8.0
    assert cfg.terminal_value == "mean_decision_da_price"


def _battery_kwargs(**overrides):
    kwargs = dict(
        power_nominal_mw=5.0, power_derating_factor=0.9, energy_mwh=10.0, round_trip_efficiency=0.90,
        soc_min_fraction=0.10, soc_max_fraction=0.90, aux_load_mw=0.05, degradation_cost_eur_mwh=8.0,
        terminal_value="mean_decision_da_price",
    )
    kwargs.update(overrides)
    return kwargs


def test_battery_config_rejects_invalid_soc_bounds():
    with pytest.raises(ValueError, match="soc_min_fraction"):
        BatteryConfig(**_battery_kwargs(soc_min_fraction=0.95, soc_max_fraction=0.05))


def test_battery_config_rejects_nonpositive_power():
    with pytest.raises(ValueError, match="power_nominal_mw"):
        BatteryConfig(**_battery_kwargs(power_nominal_mw=0.0))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"power_derating_factor": 0.0}, "power_derating_factor"),
        ({"power_derating_factor": 1.1}, "power_derating_factor"),
        ({"aux_load_mw": -0.01}, "aux_load_mw"),
        ({"degradation_cost_eur_mwh": -1.0}, "degradation_cost_eur_mwh"),
        ({"terminal_value": "bogus"}, "terminal_value"),
    ],
)
def test_battery_config_rejects_invalid_v2_fields(overrides, match):
    with pytest.raises(ValueError, match=match):
        BatteryConfig(**_battery_kwargs(**overrides))


def test_load_battery_config_rejects_old_v1_file(tmp_path):
    path = tmp_path / "battery.yaml"
    path.write_text(
        yaml.dump({
            "power_mw": 1.0, "energy_mwh": 2.0, "round_trip_efficiency": 0.88,
            "soc_min_fraction": 0.05, "soc_max_fraction": 0.95,
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required keys"):
        load_battery_config(path)


def test_load_battery_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_battery_config(tmp_path / "does_not_exist.yaml")


def test_load_battery_config_missing_key_raises(tmp_path):
    path = tmp_path / "battery.yaml"
    path.write_text(yaml.dump({"power_mw": 1.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_battery_config(path)


# ---- assumptions.yaml ---------------------------------------------------

def test_load_assumptions_config_reads_real_file():
    cfg = load_assumptions_config()
    assert isinstance(cfg, AssumptionsConfig)
    assert cfg.fcr_reserve_hours == 0.5
    assert cfg.afrr_reserve_hours == 1.0
    assert 0 <= cfg.alpha_pos <= 1
    assert 0 <= cfg.alpha_neg <= 1


def test_assumptions_config_rejects_out_of_range_alpha():
    with pytest.raises(ValueError, match="alpha_pos"):
        AssumptionsConfig(
            alpha_pos=1.5,
            alpha_neg=0.1,
            activation_energy_price_eur_mwh=100.0,
            fcr_reserve_hours=0.5,
            afrr_reserve_hours=1.0,
        )


def test_load_assumptions_config_missing_section_raises(tmp_path):
    path = tmp_path / "assumptions.yaml"
    path.write_text(yaml.dump({"afrr_activation": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="reserve_energy_duration"):
        load_assumptions_config(path)


# ---- solver.yaml --------------------------------------------------------

def test_load_solver_config_reads_real_file():
    cfg = load_solver_config()
    assert isinstance(cfg, SolverConfig)
    assert cfg.name == "highs"
    assert cfg.enforce_exclusivity is True
    assert cfg.cyclic_soc is True
    assert cfg.include_activation_in_soc is True


def test_load_solver_config_missing_flags_raises(tmp_path):
    path = tmp_path / "solver.yaml"
    path.write_text(
        yaml.dump({"solver": {"name": "highs", "time_limit_seconds": 1, "mip_gap": 0.01}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model_flags"):
        load_solver_config(path)


# ---- market_prices.yaml --------------------------------------------------

def test_load_market_price_basis_config_reads_real_file():
    cfg = load_market_price_basis_config()
    assert isinstance(cfg, MarketPriceBasisConfig)
    assert cfg.basis_for("fcr") == "per_block_total"
    assert cfg.basis_for("afrr_pos") == "per_mw_hour"
    assert cfg.basis_for("afrr_neg") == "per_mw_hour"
    assert cfg.basis_for("day_ahead") == "per_mwh_energy"


def test_market_price_basis_config_rejects_missing_market():
    with pytest.raises(ValueError, match="missing required markets"):
        MarketPriceBasisConfig(basis_by_market={"day_ahead": "per_mwh_energy"})


def test_market_price_basis_config_rejects_invalid_basis():
    with pytest.raises(ValueError, match="Invalid price_basis"):
        MarketPriceBasisConfig(
            basis_by_market={
                "day_ahead": "per_mwh_energy",
                "fcr": "not_a_real_basis",
                "afrr_pos": "per_mw_hour",
                "afrr_neg": "per_mw_hour",
            }
        )


def test_market_price_basis_config_unknown_market_lookup_raises():
    cfg = load_market_price_basis_config()
    with pytest.raises(KeyError):
        cfg.basis_for("not_a_market")


def test_market_price_basis_config_reads_real_bid_granularity():
    cfg = load_market_price_basis_config()
    assert cfg.granularity_for("fcr") == 1.0
    assert cfg.granularity_for("afrr_pos") == 1.0
    assert cfg.granularity_for("afrr_neg") == 1.0
    assert cfg.granularity_for("day_ahead") == 0


def test_market_price_basis_config_granularity_defaults_to_zero_when_absent():
    cfg = MarketPriceBasisConfig(basis_by_market={
        "day_ahead": "per_mwh_energy", "fcr": "per_block_total", "afrr_pos": "per_mw_hour", "afrr_neg": "per_mw_hour",
    })
    assert cfg.granularity_for("fcr") == 0.0


def test_market_price_basis_config_rejects_negative_granularity():
    with pytest.raises(ValueError, match="bid_granularity_mw must be >= 0"):
        MarketPriceBasisConfig(
            basis_by_market={
                "day_ahead": "per_mwh_energy", "fcr": "per_block_total",
                "afrr_pos": "per_mw_hour", "afrr_neg": "per_mw_hour",
            },
            bid_granularity_mw={"fcr": -1.0},
        )


# ---- forecast_blends.yaml --------------------------------------------------

def test_load_forecast_blends_config_reads_real_file():
    # pipeline/03_select_blend.py generates this file from validation-window MAE (PLAN_V6_REVISION.md
    # V6-4); the picks change with every study, so only the shape is pinned.
    cfg = load_forecast_blends_config()
    assert isinstance(cfg, ForecastBlendsConfig)
    blend = cfg.resolve("per_market_best")
    assert set(blend) == {"day_ahead", "fcr", "afrr_pos", "afrr_neg"}
    assert set(blend.values()) <= set(load_study_config().models)
    assert cfg.resolve("not_a_blend") is None


def test_forecast_blends_config_rejects_missing_market():
    with pytest.raises(ValueError, match="missing required markets"):
        ForecastBlendsConfig(blends={"bad": {"day_ahead": "seasonal_naive"}})


# ---- study.yaml (bess-market-forecast, read via forecast_repo_path) ---------

def test_load_study_config_reads_real_file():
    study = load_study_config()
    assert isinstance(study, StudyConfig)
    assert study.name == "v8"
    assert study.models == ["seasonal_naive", "lgbm_point", "linear_lags", "lear"]
    assert study.strategies == ["day_ahead_only", "full_cooptimized"]
    assert study.blends == ["per_market_best"]
    assert study.optimizer_models == study.models + study.blends
    assert study.history_start == dt.date(2025, 1, 1)
    assert study.min_training_history_days == 365
    assert study.validation_window["day_ahead"].start == dt.date(2026, 1, 20)
    assert study.validation_window["day_ahead"].end == dt.date(2026, 3, 20)


def test_study_config_optimization_window_is_the_intersection_of_market_windows():
    study = load_study_config()
    start, end = study.optimization_window
    assert start == max(w.start for w in study.test_window.values())
    assert end == min(w.end for w in study.test_window.values())
    assert start == dt.date(2026, 3, 21)
    assert end == dt.date(2026, 8, 31)


def test_study_config_optimization_window_rejects_non_overlapping_windows():
    study = StudyConfig(
        name="bad", models=["m"], strategies=[], blends=[],
        test_window={
            "day_ahead": StudyWindow(start=dt.date(2026, 1, 1), end=dt.date(2026, 1, 31)),
            "fcr": StudyWindow(start=dt.date(2026, 2, 1), end=dt.date(2026, 2, 28)),
        },
    )
    with pytest.raises(ValueError, match="don't overlap"):
        study.optimization_window


def test_load_study_config_missing_required_key_raises(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "study.yaml").write_text("name: v5\n", encoding="utf-8")
    bad_paths = PathsConfig(forecast_repo_path=tmp_path)
    with pytest.raises(ValueError, match="missing required keys"):
        load_study_config(bad_paths)


# ---- paths.yaml: forecast_venv_python (PLAN_V5_REVISION.md V5-7) -----------

def test_load_paths_config_reads_real_forecast_venv_python():
    paths = load_paths_config()
    assert paths.forecast_venv_python is not None
    assert paths.forecast_venv_python.exists()


def test_paths_config_forecast_venv_python_defaults_to_none_when_absent(tmp_path):
    paths = PathsConfig(forecast_repo_path=tmp_path)
    assert paths.forecast_venv_python is None


def test_paths_config_rejects_a_nonexistent_forecast_venv_python(tmp_path):
    with pytest.raises(FileNotFoundError, match="forecast_venv_python"):
        PathsConfig(forecast_repo_path=tmp_path, forecast_venv_python=tmp_path / "no_such_python.exe")
