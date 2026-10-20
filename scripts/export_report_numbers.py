"""PLAN_V5_1_REVISION.md section 5: every number the study report / slide
deck / dashboard overview show, computed once via the same app/analysis.py
functions app/pages/study_overview.py uses, and written to one JSON
snapshot. The report artifact and the slide deck are built FROM this file
(by hand, via the Artifact tool), so neither can disagree with the
dashboard or with each other.

    python scripts/export_report_numbers.py [--study v5]

Writes results/report_numbers_<study>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analysis import (  # noqa: E402
    BASELINE_MODEL, capture_comparison, forecast_error_metrics, model_revenue_table, perfect_foresight_summary,
)
from data_loader import (  # noqa: E402
    load_daily_revenue,
    load_forecast_run_meta,
    load_long_format,
    load_lookahead_summary,
    load_median_forecasts,
    load_model_info,
    load_run_catalogue,
    load_timings,
    matching_runs,
)

from src.config import REPO_ROOT, load_assumptions_config, load_battery_config, load_market_price_basis_config, load_study_config  # noqa: E402
from src.model.battery import battery_from_config  # noqa: E402

_MARKET_SELLS = {
    "day_ahead": "Energy: buy cheap, sell expensive (arbitrage)",
    "fcr": "Standby power for frequency support, both directions",
    "afrr_pos": "Standby power to push extra energy into the grid",
    "afrr_neg": "Standby power to absorb surplus energy",
}
_MARKET_LABELS = {"day_ahead": "Day-ahead", "fcr": "FCR", "afrr_pos": "aFRR+", "afrr_neg": "aFRR-"}


def _product_text(product_length_hours: float) -> str:
    n = round(24 / product_length_hours)
    return f"{n} quarter-hours" if product_length_hours < 1 else f"{n} blocks of {product_length_hours:g}h"


def _gate_text(cfg) -> str:
    if cfg.gate_closure_local is None:
        return "n/a"
    when = cfg.gate_closure_local.strftime("%H:%M")
    if cfg.gate_closure_offset_days == -1:
        return f"{when} (day before)"
    if cfg.gate_closure_offset_days in (None, 0):
        return when
    return f"{when} (D{cfg.gate_closure_offset_days:+d})"


def _study_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    study_col = frame["study"] if "study" in frame.columns else pd.Series(index=frame.index, dtype=object)
    return study_col.isna() if name == "pre-v5" else study_col == name


def _markets_section() -> list[dict]:
    timings = load_timings()
    return [
        {
            "market": market,
            "label": _MARKET_LABELS[market],
            "what_it_sells": _MARKET_SELLS[market],
            "product": _product_text(timings[market].product_length_hours),
            "bids_close": _gate_text(timings[market]),
        }
        for market in _MARKET_LABELS
    ]


def _forecast_models_section(study, catalogue: pd.DataFrame) -> dict:
    forecasts = load_median_forecasts()
    accuracy: dict[str, dict] = {}
    if not forecasts.empty:
        local_day = forecasts["target_time"].dt.date
        in_study_models = forecasts["model"].isin(study.models)
        keep = pd.Series(False, index=forecasts.index)
        for market, window in study.test_window.items():
            in_market = forecasts["market"] == market
            keep |= in_market & (local_day >= window.start) & (local_day <= window.end)
        errors = forecast_error_metrics(forecasts[keep & in_study_models], baseline=BASELINE_MODEL)
        if not errors.empty:
            skill = errors["skill_vs_baseline"].unstack("market")
            mae = errors["mae"].unstack("market")
            for model in skill.index:
                accuracy[model] = {
                    "skill": {m: skill.loc[model, m] for m in skill.columns},
                    "mae": {m: mae.loc[model, m] for m in mae.columns},
                }

    current = catalogue[_study_mask(catalogue, study.name)] if not catalogue.empty else catalogue
    run_id_cols = [c for c in current.columns if c.startswith("forecast_run_ids.")] if not current.empty else []
    forecast_rows = current[current["decision_source"] == "forecast"] if not current.empty else current
    train_days: list[int] = []
    retrain_every = None
    if not forecast_rows.empty and run_id_cols:
        row = forecast_rows.iloc[0]
        for col in run_id_cols:
            run_id = row[col]
            if pd.isna(run_id):
                continue
            meta = load_forecast_run_meta(run_id)
            if "first_train_days" in meta:
                train_days.append(meta["first_train_days"])
            retrain_every = meta.get("retrain_every_days", retrain_every)

    return {
        "models": study.models,
        "blends": study.blends,
        "accuracy": accuracy,
        "history_start": study.history_start.isoformat() if study.history_start else None,
        "first_train_days_min": min(train_days) if train_days else None,
        "first_train_days_max": max(train_days) if train_days else None,
        "retrain_every_days": retrain_every,
        "model_info": load_model_info(),
    }


def _optimizer_section(battery, study) -> dict:
    price_basis = load_market_price_basis_config()
    assumptions = load_assumptions_config()
    return {
        "battery": {
            "energy_mwh": battery.energy_mwh,
            "power_nominal_mw": battery.power_nominal_mw,
            "power_mw": battery.power_mw,
            "soc_min_mwh": battery.soc_min_mwh,
            "soc_max_mwh": battery.soc_max_mwh,
            "round_trip_efficiency": battery.round_trip_efficiency,
            "aux_load_mw": battery.aux_load_mw,
            "degradation_cost_eur_mwh": battery.degradation_cost_eur_mwh,
        },
        "fcr_bid_granularity_mw": price_basis.bid_granularity_mw.get("fcr", 1.0),
        "assumptions": {
            "alpha_pos": assumptions.alpha_pos,
            "alpha_neg": assumptions.alpha_neg,
            "activation_energy_price_eur_mwh": assumptions.activation_energy_price_eur_mwh,
            "fcr_reserve_hours": assumptions.fcr_reserve_hours,
            "afrr_reserve_hours": assumptions.afrr_reserve_hours,
        },
        "strategies": study.strategies,
    }


def _results_section(study, catalogue: pd.DataFrame) -> dict:
    current = catalogue[_study_mask(catalogue, study.name)] if not catalogue.empty else catalogue
    out: dict = {"perfect_foresight": {}, "revenue_by_strategy": {}, "capture_comparison": {}}
    if current.empty:
        return out

    perfect_rows = current[current["decision_source"] == "realized"]
    if not perfect_rows.empty:
        daily_by_id = {sid: load_daily_revenue(sid) for sid in perfect_rows["solve_id"]}
        summary = perfect_foresight_summary(current, daily_by_id)
        out["perfect_foresight"] = summary.to_dict(orient="index")

    pre_v5 = catalogue[_study_mask(catalogue, "pre-v5")]
    for strategy in study.strategies:
        perfect_match = matching_runs(current, strategy, "realized")
        if perfect_match.empty:
            continue
        perfect_daily = load_daily_revenue(perfect_match.iloc[0]["solve_id"])
        matches = {
            model: match.iloc[0]["solve_id"]
            for model in study.optimizer_models
            if not (match := matching_runs(current, strategy, "forecast", model)).empty
        }
        model_daily = {model: load_daily_revenue(sid) for model, sid in matches.items()}
        if not model_daily:
            continue
        table, perfect_net, n_days = model_revenue_table(
            perfect_daily, model_daily,
            (load_long_format(perfect_match.iloc[0]["solve_id"]), {m: load_long_format(s) for m, s in matches.items()}),
        )
        out["revenue_by_strategy"][strategy] = {
            "perfect_net_eur": perfect_net, "n_days": n_days, "models": table.to_dict(orient="index"),
        }

        if pre_v5.empty:
            continue
        old_perfect_match = matching_runs(pre_v5, strategy, "realized")
        if old_perfect_match.empty:
            continue
        old_perfect_daily = load_daily_revenue(old_perfect_match.iloc[0]["solve_id"])
        old_model_daily = {
            model: load_daily_revenue(match.iloc[0]["solve_id"])
            for model in study.optimizer_models
            if not (match := matching_runs(pre_v5, strategy, "forecast", model)).empty
        }
        if not old_model_daily:
            continue
        old_table, _, _ = model_revenue_table(old_perfect_daily, old_model_daily)
        old_pct, new_pct = capture_comparison(old_table, table)
        if old_pct:
            out["capture_comparison"][strategy] = {"old": old_pct, "new": new_pct}
    return out


def _sanitize(obj):
    """Recursively replace NaN (not valid JSON) with None."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def build_snapshot() -> dict:
    study = load_study_config()
    battery = battery_from_config(load_battery_config())
    catalogue = load_run_catalogue()
    opt_start, opt_end = study.optimization_window

    return _sanitize({
        "study": study.name,
        "test_window": {"start": opt_start.isoformat(), "end": opt_end.isoformat(), "n_days": (opt_end - opt_start).days + 1},
        "battery_headline": {"energy_mwh": battery.energy_mwh, "power_mw": battery.power_mw, "power_nominal_mw": battery.power_nominal_mw},
        "markets": _markets_section(),
        "forecast_models": _forecast_models_section(study, catalogue),
        "optimizer": _optimizer_section(battery, study),
        "results": _results_section(study, catalogue),
        "lookahead": load_lookahead_summary(study.name),
    })


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    snapshot = build_snapshot()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"report_numbers_{snapshot['study']}.json"
    out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
