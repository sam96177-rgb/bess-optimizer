import datetime as dt

import pytest
import yaml

from src.config import (
    MarketTimingConfig,
    PathsConfig,
    load_market_timing_configs,
    load_paths_config,
)


def test_load_paths_config_resolves_real_sibling_repo():
    cfg = load_paths_config()
    assert isinstance(cfg, PathsConfig)
    assert cfg.forecast_repo_path.exists()
    assert cfg.markets_yaml_path.exists()
    assert cfg.results_dir.exists()


def test_load_paths_config_missing_repo_raises(tmp_path):
    path = tmp_path / "paths.yaml"
    path.write_text(yaml.dump({"forecast_repo_path": str(tmp_path / "does_not_exist")}), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_paths_config(path)


def test_load_market_timing_configs_reads_real_sibling_file():
    configs = load_market_timing_configs()
    assert set(configs.keys()) == {"day_ahead", "fcr", "afrr_pos", "afrr_neg"}

    day_ahead = configs["day_ahead"]
    assert day_ahead.dst_convention == "variable_period_count"
    assert day_ahead.product_length_hours == 0.25
    assert day_ahead.timezone == "Europe/Berlin"
    assert day_ahead.delivery_start_local == dt.time(0, 0)
    assert day_ahead.gate_closure_local == dt.time(12, 0)
    assert day_ahead.gate_closure_offset_days == -1

    for market in ("fcr", "afrr_pos", "afrr_neg"):
        cfg = configs[market]
        assert cfg.dst_convention == "fixed_period_count"
        assert cfg.product_length_hours == 4.0
        assert cfg.gate_closure_offset_days == -1


def test_market_timing_config_rejects_invalid_dst_convention():
    with pytest.raises(ValueError, match="dst_convention"):
        MarketTimingConfig(
            name="day_ahead",
            timezone="Europe/Berlin",
            dst_convention="not_a_real_convention",
            product_length_hours=0.25,
            delivery_start_local=dt.time(0, 0),
        )


def test_load_market_timing_configs_rejects_mismatched_ancillary_block_structure(tmp_path):
    fake_repo = tmp_path / "bess-market-forecast"
    (fake_repo / "config").mkdir(parents=True)
    (fake_repo / "results").mkdir()
    markets = {
        "day_ahead": {
            "timezone": "Europe/Berlin",
            "dst_convention": "variable_period_count",
            "product_length_hours": 0.25,
            "delivery_start_local": "00:00",
        },
        "fcr": {
            "timezone": "Europe/Berlin",
            "dst_convention": "fixed_period_count",
            "product_length_hours": 4,
            "delivery_start_local": "00:00",
        },
        "afrr_pos": {
            "timezone": "Europe/Berlin",
            "dst_convention": "fixed_period_count",
            "product_length_hours": 2,  # deliberately mismatched vs fcr
            "delivery_start_local": "00:00",
        },
        "afrr_neg": {
            "timezone": "Europe/Berlin",
            "dst_convention": "fixed_period_count",
            "product_length_hours": 4,
            "delivery_start_local": "00:00",
        },
    }
    (fake_repo / "config" / "markets.yaml").write_text(yaml.dump(markets), encoding="utf-8")

    paths_path = tmp_path / "paths.yaml"
    paths_path.write_text(yaml.dump({"forecast_repo_path": str(fake_repo)}), encoding="utf-8")
    paths_config = load_paths_config(paths_path)

    with pytest.raises(ValueError, match="share one block structure"):
        load_market_timing_configs(paths_config)
