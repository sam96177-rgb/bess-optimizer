"""run_meta.json sidecar (SPEC.md §6a / Milestone 8 Part B): carries every
field results_catalogue.scan_runs and batch.py's resumability fingerprint
depend on, and is written LAST so a crashed run leaves no sidecar.
"""
import datetime as dt
import json

import pytest

import src.model.rolling as rolling
from src.config import (
    load_assumptions_config,
    load_battery_config,
    load_market_price_basis_config,
    load_market_timing_configs,
    load_paths_config,
    load_solver_config,
)
from src.model.battery import battery_from_config
from src.model.rolling import run_rolling, write_rolling_run_results

PATHS = load_paths_config()
TIMINGS = load_market_timing_configs(PATHS)
BATTERY = battery_from_config(load_battery_config())
SOLVER = load_solver_config()
ASSUMPTIONS = load_assumptions_config()
PRICE_BASIS = load_market_price_basis_config()

START = END = dt.date(2026, 6, 15)


def _write(tmp_path, **kwargs):
    result = run_rolling(
        start_date=START, end_date=END, battery=BATTERY, solver_config=SOLVER,
        assumptions=ASSUMPTIONS, price_basis=PRICE_BASIS,
        day_ahead_timing=TIMINGS["day_ahead"], capacity_timing=TIMINGS["fcr"],
        paths_config=PATHS, **kwargs,
    )
    write_rolling_run_results(
        result, tmp_path,
        strategy="full_cooptimized", start_date=START, end_date=END,
        battery=BATTERY, solver_config=SOLVER, assumptions=ASSUMPTIONS, price_basis=PRICE_BASIS,
        paths_config=PATHS,
    )
    meta_path = tmp_path / result.solve_id / "run_meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        return result, json.load(f)


def test_run_meta_carries_every_field_for_a_forecast_driven_run(tmp_path):
    result, meta = _write(
        tmp_path, decision_source="forecast", forecast_model="lgbm_point", forecast_quantile=0.5,
    )

    assert meta["solve_id"] == result.solve_id
    assert meta["strategy"] == "full_cooptimized"
    assert meta["decision_source"] == "forecast"
    assert meta["forecast_model"] == "lgbm_point"
    assert meta["forecast_quantile"] == 0.5
    assert meta["start_date"] == START.isoformat()
    assert meta["end_date"] == END.isoformat()
    assert meta["n_days_solved"] == len(result.daily_revenue)
    assert meta["n_days_skipped"] == len(result.skip_log)

    assert meta["battery"] == {
        "power_nominal_mw": BATTERY.power_nominal_mw,
        "power_derating_factor": BATTERY.power_derating_factor,
        "power_mw": BATTERY.power_mw,
        "energy_mwh": BATTERY.energy_mwh,
        "round_trip_efficiency": BATTERY.round_trip_efficiency,
        "soc_min_fraction": BATTERY.soc_min_fraction,
        "soc_max_fraction": BATTERY.soc_max_fraction,
        "aux_load_mw": BATTERY.aux_load_mw,
        "degradation_cost_eur_mwh": BATTERY.degradation_cost_eur_mwh,
        "terminal_value": BATTERY.terminal_value,
    }
    assert meta["solver"] == {
        "name": SOLVER.name, "mip_gap": SOLVER.mip_gap, "time_limit_seconds": SOLVER.time_limit_seconds,
    }
    assert meta["model_flags"] == {
        "cyclic_soc": True, "enforce_exclusivity": True, "include_activation_in_soc": True,
    }
    assert meta["assumptions"] == {
        "alpha_pos": ASSUMPTIONS.alpha_pos,
        "alpha_neg": ASSUMPTIONS.alpha_neg,
        "activation_energy_price_eur_mwh": ASSUMPTIONS.activation_energy_price_eur_mwh,
        "fcr_reserve_hours": ASSUMPTIONS.fcr_reserve_hours,
        "afrr_reserve_hours": ASSUMPTIONS.afrr_reserve_hours,
    }
    assert meta["price_basis"] == dict(PRICE_BASIS.basis_by_market)
    dr = result.daily_revenue
    assert meta["gross_revenue_eur"] == pytest.approx(float(dr["gross"].sum()))
    assert meta["degradation_cost_eur"] == pytest.approx(float(dr["degradation"].sum()))
    assert meta["net_revenue_eur"] == pytest.approx(meta["gross_revenue_eur"] - meta["degradation_cost_eur"])
    assert meta["total_revenue_eur"] == pytest.approx(meta["net_revenue_eur"])
    usable = BATTERY.soc_max_mwh - BATTERY.soc_min_mwh
    assert meta["equivalent_full_cycles"] == pytest.approx(float(dr["discharged_mwh"].sum()) / usable)
    assert meta["degradation_cost_eur"] > 0
    assert meta["forecast_repo_path"] == str(PATHS.forecast_repo_path)

    # One (market, model) pair per parquet file in the sibling repo (Verified facts).
    assert set(meta["forecast_run_ids"].keys()) == {"day_ahead", "fcr", "afrr_pos", "afrr_neg"}

    assert meta["created_at"]  # non-empty ISO timestamp; exact value isn't worth pinning


def test_run_meta_has_empty_forecast_run_ids_for_a_perfect_foresight_run(tmp_path):
    _, meta = _write(tmp_path)  # decision_source="realized" (default)
    assert meta["decision_source"] == "realized"
    assert meta["forecast_model"] is None
    assert meta["forecast_quantile"] is None
    assert meta["forecast_run_ids"] == {}


def test_run_meta_is_written_last(tmp_path, monkeypatch):
    """If assembling the metadata fails, the parquets and skip log must
    already be on disk (a partial-but-inspectable run) while run_meta.json
    must NOT exist (results_catalogue.scan_runs only discovers runs that
    have one -- a crashed run must stay invisible to it, not look done).
    """
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure while assembling run_meta")

    monkeypatch.setattr(rolling, "build_run_meta", _boom)

    result = run_rolling(
        start_date=START, end_date=END, battery=BATTERY, solver_config=SOLVER,
        assumptions=ASSUMPTIONS, price_basis=PRICE_BASIS,
        day_ahead_timing=TIMINGS["day_ahead"], capacity_timing=TIMINGS["fcr"], paths_config=PATHS,
    )
    with pytest.raises(RuntimeError, match="simulated failure"):
        write_rolling_run_results(
            result, tmp_path,
            strategy="full_cooptimized", start_date=START, end_date=END,
            battery=BATTERY, solver_config=SOLVER, assumptions=ASSUMPTIONS, price_basis=PRICE_BASIS,
            paths_config=PATHS,
        )

    run_dir = tmp_path / result.solve_id
    assert (run_dir / "long_format.parquet").exists()
    assert (run_dir / "daily_revenue.parquet").exists()
    assert (run_dir / "skip_log.csv").exists()
    assert not (run_dir / "run_meta.json").exists()
