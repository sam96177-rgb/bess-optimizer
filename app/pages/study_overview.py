"""Study Overview: the study report's five sections
(https://claude.ai/artifact/UUTiDztG73g33b89BEiKWX), rebuilt inside the
dashboard (PLAN_V5_1_REVISION.md section 3). Every figure here is computed
live from results/, the sibling forecast repo and the configs; only the
explanatory prose is fixed text, so a number here can never disagree with
the page that actually computes it.
"""
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (_APP_DIR, _APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import streamlit as st

import ui
from analysis import BASELINE_MODEL, capture_comparison, forecast_error_metrics, model_revenue_table, perfect_foresight_summary
from data_loader import (
    load_daily_revenue,
    load_forecast_run_meta,
    load_long_format,
    load_lookahead_summary,
    load_median_forecasts,
    load_run_catalogue,
    load_timings,
    matching_runs,
)

from src.config import load_assumptions_config, load_battery_config, load_market_price_basis_config, load_study_config
from src.model.battery import battery_from_config

st.title("Study Overview")

study = load_study_config()
battery = battery_from_config(load_battery_config())
catalogue = load_run_catalogue()


def _study_labels(frame: pd.DataFrame) -> pd.Series:
    study_col = frame["study"] if "study" in frame.columns else pd.Series(index=frame.index, dtype=object)
    return study_col.fillna("pre-v5")


def _study_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    return _study_labels(frame) == name


def _previous_study_name(frame: pd.DataFrame, current_name: str) -> str | None:
    """The other study tag with the most recently created run, for the
    "previous study vs current" comparison -- not hardcoded to "pre-v5", so
    this still finds v5 once the current study is v6, v7, ... """
    if frame.empty:
        return None
    labels = _study_labels(frame)
    others = labels[labels != current_name]
    if others.empty:
        return None
    if "created_at" in frame.columns:
        return frame.loc[others.index].assign(_label=others).groupby("_label")["created_at"].max().idxmax()
    return sorted(others.unique())[-1]


current = catalogue[_study_mask(catalogue, study.name)] if not catalogue.empty else catalogue

try:
    opt_start, opt_end = study.optimization_window
    window_text = f"{opt_start} to {opt_end} ({(opt_end - opt_start).days + 1} days)"
except ValueError:
    window_text = "test windows don't overlap -- see config/study.yaml"

st.caption(
    f"Study **{study.name}** | battery {battery.energy_mwh:.0f} MWh, {battery.power_mw:.1f} MW usable "
    f"({battery.power_nominal_mw:.0f} MW nameplate) | test period {window_text}"
)

# ============================================================ Page 1 ====
st.header("1. The big picture")
st.markdown(
    "The study asks one question: **if a battery had to commit its bids the day before, using "
    "only forecasts, how much of the best possible revenue would it actually earn?**\n\n"
    "Answering that needs two separate programs, one per repository. They only talk through "
    "files on disk.\n\n"
    "- **The forecaster** (`bess-market-forecast`) predicts tomorrow's prices in four markets. "
    "It knows nothing about batteries.\n"
    "- **The optimizer** (`bess-optimizer`) reads those predictions and decides, for each day, "
    "when to charge, when to discharge and how much reserve capacity to offer. It then pays "
    "that plan at the prices that really happened."
)

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


timings = load_timings()
markets_table = pd.DataFrame(
    {
        "what it sells": [_MARKET_SELLS[m] for m in _MARKET_LABELS],
        "product": [_product_text(timings[m].product_length_hours) for m in _MARKET_LABELS],
        "bids close": [_gate_text(timings[m]) for m in _MARKET_LABELS],
    },
    index=[_MARKET_LABELS[m] for m in _MARKET_LABELS],
)
st.subheader("The four markets the battery can earn from")
ui.styled_table(markets_table, align_right=[])
st.caption("FCR and aFRR pay for *being available*, whether or not the grid actually calls on the battery.")

st.subheader("How a study run flows, in order")
_PIPELINE = [
    ("pipeline/01", "Ingest", "Download prices and weather from ENTSO-E, regelleistung.net and Open-Meteo"),
    ("pipeline/02", "Forecast", "Train each model and predict every validation + test day, as if in real time"),
    ("pipeline/03", "Select blend", "Pick each market's best model on the validation window only"),
    ("pipeline/04", "Optimize", "Solve each day's battery plan on the forecasts; pay it at real prices"),
    ("pipeline/05", "Lookahead", "Test whether seeing past midnight would help"),
    ("pipeline/06", "Export", "Write the run table; open the dashboard"),
]
cols = st.columns(len(_PIPELINE))
for col, (step, name, desc) in zip(cols, _PIPELINE):
    with col:
        st.markdown(f"`{step}`  \n**{name}**  \n<span style='font-size:.85rem;color:var(--bessr-muted)'>{desc}</span>", unsafe_allow_html=True)
st.caption(
    "One settings file, `config/study.yaml` in the forecast repo, defines the whole study: the "
    "test windows, how much history to train on, and which models and strategies to run. Both "
    "programs read it."
)

with st.expander("Glossary"):
    st.markdown(
        "- **Ceiling (real prices)**: the optimizer given the real prices in advance. Not "
        "achievable; reference only -- everything is measured against it.\n"
        "- **Baseline**: `seasonal_naive`, which uses no forecasting model at all -- tomorrow is "
        "the same day last week.\n"
        "- **Timing**: share of the model's charge (or discharge) quarter-hours where the ceiling "
        "plan does the same within 30 minutes.\n"
        "- **Amount**: 100% minus the daily MWh gap to the ceiling. Ceiling charges 14 MWh, model "
        "charges 17: gap 3, amount = 1 - 3/14 = 79%.\n"
        "- **Capture**: what a plan actually earned (paid at real prices), as a share of the "
        "ceiling's net on the same days.\n"
        "- **MAE**: mean absolute error -- how far a forecast is from the real price on average, "
        "in EUR/MWh.\n"
        "- **SOC**: state of charge -- how much energy is in the battery."
    )

# ============================================================ Page 2 ====
st.header("2. The forecast models")
st.markdown(
    f"{study.name} runs {len(study.models)} model(s), each producing a price for every "
    "quarter-hour (day-ahead) or 4-hour block (reserves) of the next day, made before that "
    "market's bidding deadline."
)
ui.model_picker(study.optimizer_models, key="overview_model_picker")

forecasts = load_median_forecasts()
if forecasts.empty:
    st.info("No forecast result files found yet in the sibling repo's `results/`.")
else:
    local_day = forecasts["target_time"].dt.date
    in_study_models = forecasts["model"].isin(study.models)
    keep = pd.Series(False, index=forecasts.index)
    for market, window in study.test_window.items():
        in_market = forecasts["market"] == market
        keep |= in_market & (local_day >= window.start) & (local_day <= window.end)
    windowed_forecasts = forecasts[keep & in_study_models]
    errors = forecast_error_metrics(windowed_forecasts, baseline=BASELINE_MODEL)
    if errors.empty:
        st.info("No forecast results for this study's models inside its test windows yet.")
    else:
        skill = errors["skill_vs_baseline"].unstack("market")
        skill_cols = [m for m in _MARKET_LABELS if m in skill.columns]
        accuracy = skill[skill_cols].rename(columns={m: f"{_MARKET_LABELS[m]} skill" for m in skill_cols})
        if "day_ahead" in errors.index.get_level_values("market"):
            accuracy["day-ahead MAE"] = errors.xs("day_ahead", level="market")["mae"]
        highlight = accuracy["Day-ahead skill"].idxmax() if "Day-ahead skill" in accuracy.columns and accuracy["Day-ahead skill"].notna().any() else None
        st.subheader("How accurate are they now?")
        st.caption(f"Skill = 1 - RMSE(model) / RMSE({BASELINE_MODEL}); 0 = baseline, 1 = perfect. Day-ahead MAE in EUR/MWh.")
        formats = {c: "{:.3f}" for c in accuracy.columns if c != "day-ahead MAE"}
        if "day-ahead MAE" in accuracy.columns:
            formats["day-ahead MAE"] = "{:.2f}"
        ui.styled_table(accuracy, formats=formats, highlight_row=highlight)

    run_id_cols = [c for c in current.columns if c.startswith("forecast_run_ids.")] if not current.empty else []
    forecast_rows = current[current["decision_source"] == "forecast"] if not current.empty else current
    train_days, retrain_every = [], None
    generation_forecast_published_local = None
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
            generation_forecast_published_local = meta.get("generation_forecast_published_local", generation_forecast_published_local)
    if train_days and study.history_start is not None:
        span = f"{min(train_days)} to {max(train_days)}" if min(train_days) != max(train_days) else f"{train_days[0]}"
        retrain_text = f", retrained every {retrain_every} days" if retrain_every else ""
        st.info(
            f"Training history for **{study.name}** starts {study.history_start.isoformat()} "
            f"({span} days before each market's first test day{retrain_text})."
        )

# ============================================================ Page 3 ====
st.header("3. The optimizer")
st.markdown(
    "For each day, the optimizer solves a **mixed-integer linear program** (MILP) using the "
    "open-source HiGHS solver: it decides charge/discharge power for every quarter-hour and how "
    "much standby power to offer FCR, aFRR+ and aFRR- in each 4-hour block."
)
price_basis = load_market_price_basis_config()
granularity = price_basis.bid_granularity_mw.get("fcr", 1.0)
rules = pd.DataFrame(
    {
        "in plain words": [
            f"Charge in minus discharge out, with {battery.round_trip_efficiency * 100:.0f}% round-trip "
            f"efficiency and a {battery.aux_load_mw * 1000:.0f} kW standby load",
            f"Stay between {battery.soc_min_mwh:.0f} and {battery.soc_max_mwh:.0f} MWh, keeping headroom "
            "free for any reserve it has sold",
            f"Trading power plus reserve power can't exceed {battery.power_mw:.1f} MW in either direction",
            "FCR needs 1.25 MW of battery power per MW sold (TSO prequalification rule); aFRR+ and aFRR- each keep their own power band",
            f"Reserve bids only in {granularity:g} MW steps (regelleistung.net rule)" if granularity else "No bid-size granularity enforced for this market",
            "No charging and discharging in the same quarter-hour",
            "Each day ends with the same charge it started with",
        ],
    },
    index=["Energy balance", "Charge limits", "Shared power", "Reserve power", "Whole megawatts", "One direction at a time", "Daily loop"],
)
st.subheader("The rules it must obey")
ui.styled_table(rules, align_right=[])
st.caption(
    f"Goal: maximise trading profit, plus reserve payments, plus expected activation income, minus "
    f"wear (EUR {battery.degradation_cost_eur_mwh:.0f} per MWh moved)."
)

st.subheader("Decide on forecasts, get paid on reality")
st.markdown(
    "The optimizer only ever sees the *forecast* prices when it plans. Afterwards, the same plan "
    "is paid at the *real* prices. **Capture** is that plan's net, divided by the ceiling's net on "
    "the same days -- what the battery would really earn, as a share of the unreachable best case "
    "(PLAN_V6_REVISION.md V6-1 dropped the separate forecast-price headline figure)."
)

_STRATEGY_INFO = {
    "day_ahead_only": ("Day-ahead only", "Isolates pure arbitrage, where forecast shape matters most"),
    "full_cooptimized": ("All four, decided together", "The realistic revenue-stacking case"),
}
strategies_table = pd.DataFrame(
    {"why run it": [_STRATEGY_INFO.get(s, ("", s))[1] for s in study.strategies]},
    index=[_STRATEGY_INFO.get(s, (s, ""))[0] or s for s in study.strategies],
)
st.subheader("Strategies in this study")
ui.styled_table(strategies_table, align_right=[])

# ============================================================ Page 4 ====
st.header("4. Results")
revenue_tables: dict[str, pd.DataFrame] = {}
perfect_nets: dict[str, float] = {}
n_days_by_strategy: dict[str, int] = {}

if current.empty:
    st.info(f"No runs tagged `study: {study.name}` in `results/` yet. Run `python -m src.batch`.")
else:
    perfect_rows = current[current["decision_source"] == "realized"]
    if perfect_rows.empty:
        st.info("No perfect-foresight runs for this study yet.")
    else:
        daily_by_id = {sid: load_daily_revenue(sid) for sid in perfect_rows["solve_id"]}
        summary = perfect_foresight_summary(current, daily_by_id)
        st.subheader("The ceiling (real prices)")
        formats = {c: "{:,.0f}" for c in summary.columns if c != "cycles_per_day"}
        formats["cycles_per_day"] = "{:.2f}"
        ui.styled_table(summary, formats=formats)

    st.subheader(f"Forecast-driven, by strategy ({study.name})")
    for strategy in study.strategies:
        perfect_match = matching_runs(current, strategy, "realized")
        if perfect_match.empty:
            st.caption(f"`{strategy}`: no perfect-foresight run yet.")
            continue
        perfect_daily = load_daily_revenue(perfect_match.iloc[0]["solve_id"])
        model_daily, model_long = {}, {}
        for model in study.optimizer_models:
            match = matching_runs(current, strategy, "forecast", model)
            if not match.empty:
                model_daily[model] = load_daily_revenue(match.iloc[0]["solve_id"])
                model_long[model] = load_long_format(match.iloc[0]["solve_id"])
        if not model_daily:
            st.caption(f"`{strategy}`: no forecast-driven runs yet.")
            continue
        table, perfect_net, n_days = model_revenue_table(
            perfect_daily, model_daily, (load_long_format(perfect_match.iloc[0]["solve_id"]), model_long)
        )
        revenue_tables[strategy], perfect_nets[strategy], n_days_by_strategy[strategy] = table, perfect_net, n_days
        highlight = table["charge_timing_pct"].idxmax() if table["charge_timing_pct"].notna().any() else None
        st.markdown(f"**`{strategy}`** -- ceiling net EUR {perfect_net:,.0f} over {n_days} common day(s)")
        ui.styled_table(table, formats=ui.AGREEMENT_FORMATS, highlight_row=highlight)
        st.caption(ui.AGREEMENT_CAPTION)

    previous_name = _previous_study_name(catalogue, study.name)
    previous = catalogue[_study_mask(catalogue, previous_name)] if previous_name else pd.DataFrame()
    if not previous.empty and revenue_tables:
        st.subheader(f"Capture: {previous_name} vs {study.name}")
        any_chart = False
        for strategy, new_table in revenue_tables.items():
            old_perfect_match = matching_runs(previous, strategy, "realized")
            if old_perfect_match.empty:
                continue
            old_perfect_daily = load_daily_revenue(old_perfect_match.iloc[0]["solve_id"])
            old_model_daily = {}
            for model in study.optimizer_models:
                match = matching_runs(previous, strategy, "forecast", model)
                if not match.empty:
                    old_model_daily[model] = load_daily_revenue(match.iloc[0]["solve_id"])
            if not old_model_daily:
                continue
            old_table, _, _ = model_revenue_table(old_perfect_daily, old_model_daily)
            old_pct, new_pct = capture_comparison(old_table, new_table)
            if not old_pct:
                continue
            any_chart = True
            ui.capture_bars(old_pct, new_pct, title=f"Capture, {strategy}")
        if not any_chart:
            st.caption(f"No model has a run in both {previous_name} and {study.name} for the same strategy.")

# ============================================================ Page 5 ====
st.header("5. What it means")

if revenue_tables:
    st.subheader("Takeaways")
    bullets = []
    for strategy, table in revenue_tables.items():
        if table["capture_pct"].isna().all():
            continue
        best_model = table["capture_pct"].idxmax()
        best_pct = table.loc[best_model, "capture_pct"]
        bullets.append(
            f"- In **`{strategy}`**, `{best_model}` captures the most of the ceiling: "
            f"**{best_pct:.1f}%** (leaves about {100 - best_pct:.0f}% on the table)."
        )
        value = table.drop(index=BASELINE_MODEL, errors="ignore")["value_vs_baseline_eur"]
        if value.notna().any() and value.max() > 0:
            best_vs_baseline = value.idxmax()
            bullets.append(
                f"  Over `{BASELINE_MODEL}` (no forecasting model), `{best_vs_baseline}` is worth "
                f"**EUR {value[best_vs_baseline]:+,.0f}** on these days."
            )
    st.markdown("\n".join(bullets) if bullets else "_No model has a capture figure yet._")

st.subheader("Would looking past midnight help?")
lookahead = load_lookahead_summary(study.name)
if lookahead is None:
    st.info(f"Lookahead comparison not run yet for study `{study.name}`. Run `python pipeline/05_lookahead.py`.")
else:
    baseline = lookahead.get("baseline_net_eur")
    chained = lookahead.get("chained_net_eur")
    windowed = lookahead.get("windowed_net_eur")
    rows = {"Each day on its own (today's method)": baseline}
    if chained is not None:
        rows["Energy carried overnight, no lookahead"] = chained
    rows[f"Energy carried overnight + {lookahead.get('lookahead_hours', '?')}h lookahead"] = windowed
    la_table = pd.DataFrame({"net_eur": rows})
    if baseline:
        la_table["vs_baseline"] = [None if pd.isna(v) else 100 * (v - baseline) / abs(baseline) for v in la_table["net_eur"]]
    ui.styled_table(
        la_table,
        formats={"net_eur": "{:,.0f}", "vs_baseline": "{:+.1f}%"},
        highlight_row="Each day on its own (today's method)",
    )
    st.caption(f"Window: {lookahead.get('start_date')} to {lookahead.get('end_date')}, day-ahead only, perfect prices.")

st.subheader("Read the reserve figures with care")
assumptions = load_assumptions_config()
generation_note = (
    f"<li><strong>Generation forecast timing:</strong> wind/solar forecast features are assumed "
    f"available by <code>{generation_forecast_published_local}</code> D-1 -- ENTSO-E's regulation only "
    "guarantees 18:00 D-1, which is after every market's cutoff.</li>"
    if generation_forecast_published_local else ""
)
st.html(
    '<div class="bessr-note"><b>Inputs still assumptions rather than measurements.</b><ul>'
    f"<li><strong>aFRR activation:</strong> the battery is assumed to be called "
    f"{assumptions.alpha_pos * 100:.0f}% of the time and paid EUR {assumptions.activation_energy_price_eur_mwh:.0f}/MWh. "
    "ENTSO-E publishes activation prices but not volumes.</li>"
    f"<li><strong>Power rating:</strong> the 5 MW rating is derated to {battery.power_mw:.1f} MW; "
    "no rule for that derating was found.</li>"
    f"<li><strong>Reserve energy:</strong> the model keeps {assumptions.fcr_reserve_hours:g}-{assumptions.afrr_reserve_hours:g} "
    "hour(s) of energy per MW sold, against 4-hour blocks.</li>"
    "<li><strong>Old day-ahead history:</strong> prices before October 2025 were hourly, spread evenly "
    "across their quarter-hours for training.</li>"
    f"{generation_note}"
    "</ul></div>"
)

st.subheader("Decisions still open")
st.markdown(
    "- Power derating (4.5 of 5 MW): no rule found; a newer prequalification document than July 2024 may exist.\n"
    "- Activation: replace the assumed rate/price placeholders with measured data.\n"
    "- Nothing in `results/` is committed to git yet (see `PLAN_V3_REVISION.md` through "
    "`PLAN_V5_1_REVISION.md`); the history split is still a call for later."
)
