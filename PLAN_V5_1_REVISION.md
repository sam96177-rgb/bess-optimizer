# Plan v5.1: the study report inside the dashboard (presentation only)

> **Status (2026-10-01):** approved, not started. Presentation only: no change to forecasting,
> the MILP, settlement or batch runs, so no re-solve. Commit only when the user asks.

## Context

After V5 the user found the dashboard hard to follow: too many models, including V4 leftovers,
and bare interactive grids. The static five-page report
(https://claude.ai/artifact/UUTiDztG73g33b89BEiKWX) was much easier to read. v5.1 brings that
report into the dashboard website:
- a landing page with the report's five sections,
- clickable forecasters that show a short introduction,
- tables styled like the report,
- the same look carried into a slide deck for the final presentation.

This is **presentation only**. There are no changes to forecasting, the MILP, settlement or
batch runs, so no re-solve is needed.

(Named v5.1 so it doesn't clash with section "V5-1" inside `PLAN_V5_REVISION.md`.)

User decisions (2026-09-30):
- new landing page,
- a model intro everywhere a model is picked,
- numbers computed live from `results/`,
- dashboard and a slide deck share one style.

**Step 0:** copy this plan to `bess-optimizer/PLAN_V5_1_REVISION.md` with a status block.

## 1. One source for model introductions

New file `app/model_info.yaml`, one entry per name the dashboard can show. Each entry has:
- `kind` (e.g. "Lookup · learns nothing"),
- a one-sentence `summary`,
- `strength`, `weakness` and `learns`.

Entries: the 4 V5 models, `per_market_best` (blend), perfect foresight, plus short "dropped in
V5" entries for `analog_day`, `naive_4w` and `flat_trailing_mean`, so old runs still get a card.
The text comes from the report's page 2.

`data_loader.load_model_info()` is cached. A test checks that every model in `study.yaml` and
every blend has an entry.

## 2. Shared look: `app/ui.py` + `.streamlit/config.toml`

**`.streamlit/config.toml` theme:**
- primary colour `#0d6b86`, matching the report's backgrounds and text colour, light and dark;
- fonts IBM Plex Sans / Plex Sans Condensed / Plex Mono through Streamlit's theme font
  settings. Verify that 1.64 accepts Google-font URLs; if it doesn't, inject one `<link>` +
  CSS from `ui.inject_theme()`.

**`app/ui.py` (new, small, Streamlit-only; `analysis.py` stays pure pandas):**
- `inject_theme()`: called once in `streamlit_app.py`. It adds only the report's table and card
  CSS tokens, the same values as the report, with dark variants.
- `styled_table(df, formats, highlight_row=None, align_right=None)`: renders a DataFrame as the
  report's table via `st.html`:
  - hairline rows,
  - right-aligned mono numbers with `tabular-nums`,
  - the best/selected row tinted,
  - horizontal scroll on narrow screens.
- `model_card(name)`: the intro card from `model_info.yaml`.
- `model_picker(models, key)`: `st.pills` of the models. Clicking one shows its `model_card`.
  Returns the pick.
- `study_filter(catalogue)`:
  - a sidebar "Study" selector that defaults to the current `study.yaml` name (`v5`);
  - untagged runs are grouped as `pre-v5`;
  - returns the filtered catalogue.
  - Every page uses it, so the V4 leftovers disappear unless `pre-v5` is chosen.
- `capture_bars(v_old, v_new)`: the report's paired V4/V5 bar chart, as HTML.

## 3. New landing page `app/pages/study_overview.py` (first in nav, `default=True`)

Five sections mirroring the report. Every figure is computed from `results/`, the forecast
repo and the configs; only explanations are fixed text.

1. **Big picture:**
   - fixed prose,
   - the markets table (from the forecast repo's `markets.yaml` via `load_timings()`),
   - the numbered pipeline, glossary.
2. **Forecast models:**
   - `model_picker` over the study's models + blends;
   - a skill/MAE table from `analysis.forecast_error_metrics(load_median_forecasts())`,
     filtered to the study's models and windows;
   - the V5 change note with `first_train_days` / `history_start` read from `study.yaml` and
     the forecast run JSONs.
3. **Optimizer:**
   - the rules table (fixed text; battery numbers read from `config/battery.yaml` via
     `battery_from_config`);
   - the realized vs headline explanation, and the strategies table.
4. **Results:**
   - perfect foresight gross/wear/net/cycles from the catalogue's `run_meta` columns;
   - the market split via `analysis.summed` on `daily_revenue`;
   - forecast-driven net and realized capture per strategy via `analysis.model_revenue_table`
     (best row highlighted);
   - the V4-vs-V5 capture bars, comparing `pre-v5` against the current-study runs for models
     in both.
5. **What it means:**
   - takeaways. The data-driven sentences (e.g. "best model in full co-optimization") are
     generated from the computed tables, so they can't contradict the numbers;
   - the lookahead table;
   - assumptions read from `config/assumptions.yaml`;
   - open decisions.

**Lookahead numbers.** They are only printed to stdout today. `scripts/lookahead_value.py`
(via `pipeline/04_lookahead.py`) will also write `results/lookahead_<study>.json`. This is
output-only and doesn't change the calculation. The overview reads it, or shows "not run yet".

**Shared calculations.** New pure functions in `app/analysis.py`, e.g.
`perfect_foresight_summary(catalogue, daily_by_id)` and `capture_comparison(...)`, unit-tested.
The slide deck reuses them.

## 4. Existing pages

- **All pages:** `study_filter` in the sidebar.
- **Summary tables** switch from `st.dataframe` to `styled_table`: capture tables
  (`oracle_capture.py`), the error/revenue tables (`forecast_accuracy.py`), the cost matrix
  (`forecast_error_cost.py`).
  - Large per-day tables (digital twin per-day, dropped-day counts) keep `st.dataframe`, where
    scrolling and sorting matter.
- **Model intro cards:**
  - `dispatch_explorer.py`: the card under the sidebar "Model" selector;
  - `oracle_capture.py`: under the detail model picker;
  - `forecast_accuracy.py`: a `model_picker` above the tables.

## 5. Slide deck (after the dashboard is done)

- A "Slides" artifact, ~10 slides following the report's five sections:
  - title,
  - the question and the two programs,
  - the markets,
  - the pipeline,
  - the 4 models (one slide, cards),
  - model accuracy,
  - the optimizer rules,
  - results (ceiling + capture table),
  - V4 vs V5 chart,
  - takeaways and open decisions.
- Same palette, fonts and table style as the dashboard.
- **Numbers:** taken from a snapshot written by a small `scripts/export_report_numbers.py`
  that calls the same `analysis.py` functions (JSON in `results/`). Deck and dashboard can't
  disagree.
- Also republish the report artifact from the same snapshot, so all three match.

## Files

- **New:** `app/model_info.yaml`, `app/ui.py`, `app/pages/study_overview.py`,
  `.streamlit/config.toml`, `scripts/export_report_numbers.py`, `PLAN_V5_1_REVISION.md`.
- **Changed:**
  - `app/streamlit_app.py` (nav + theme),
  - `app/data_loader.py` (`load_model_info`),
  - `app/analysis.py` (summary functions),
  - the 5 pages,
  - `scripts/lookahead_value.py` (JSON output only),
  - `tests/test_app_pages.py`, `tests/test_app_analysis.py`,
  - README "Dashboard" section.

## Verification

- `.venv\Scripts\python.exe -m pytest tests/ -q` green (270 before), with new tests:
  - a model-info coverage test,
  - a `styled_table` HTML test: numbers right-aligned, highlighted row present;
  - summary functions against hand-computed values;
  - `study_filter` default;
  - `study_overview.py` added to the headless page test.
- Run the dashboard (detached, as before) and check in the browser pane:
  - the overview opens first, and all five sections render with numbers matching the report
    (e.g. `full_cooptimized` perfect-foresight net 586,881; `linear_lags` 94.1%);
  - clicking each model chip shows its card;
  - the default study `v5` hides the V4-only models; switching to `pre-v5` shows them;
  - light and dark themes are both readable.
- Deck: open the Slides artifact and confirm its numbers equal the snapshot JSON.
- No commit unless asked.
