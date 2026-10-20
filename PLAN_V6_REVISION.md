# Plan v6: a clear three-level comparison, wind/solar inputs, a fair best-per-market blend, a tighter dashboard

> **Status (2026-10-01): DONE.** All of V6-1 through V6-7 built and the real pipeline run
> through `pipeline/06_export.py`. Both suites green: bess-optimizer 274 passed, forecast
> repo 81 passed. `pipeline/05_lookahead.py` (not listed in the Execution order's step 4)
> was not re-run -- `results/lookahead_v6.json` does not exist yet. Nothing committed; still
> the user's call.
> Background (DONE): `PLAN_V5_REVISION.md`, `PLAN_V5_1_REVISION.md`.
>
> **Two real bugs found and fixed along the way (not in the original plan):**
> - `src/ingest/entsoe.py`'s `get_wind_and_solar_actual`: ENTSO-E's `query_generation`
>   returns a `(psr_name, flow)` MultiIndex whenever a type has more than one flow (Wind
>   Onshore and Solar both carry "Actual Consumption" alongside "Actual Aggregated"; Wind
>   Offshore has only one). A bare `df[["Wind Onshore", ...]]` matched every flow under each
>   name, not just generation, and broke the first real feature build with a
>   `"Data must be 1-dimensional, got ndarray of shape (96, 2)"` error. Fixed by selecting
>   `"Actual Aggregated"` via `.xs(..., level=-1)` before the name filter; regression test in
>   `tests/test_entsoe_ingest.py`. The stale cache under
>   `bess-market-forecast/data/raw/entsoe/wind_solar_actual_mw/` had to be deleted and
>   re-fetched once after the fix.
> - `src/batch.py`: a resumed **ceiling** (`decision_source=="realized"`) run doesn't depend
>   on the forecast model, so V6's batch correctly reused the existing solve instead of
>   re-solving it -- but its `run_meta.json`'s `study` field still said `"v5"`, which made it
>   invisible to every page/script that filters the catalogue by the CURRENT study (both
>   `scripts/export_report_numbers.py` and the dashboard came back completely empty for v6
>   until this was found). Fixed by `_restamp_study_if_resumed`, called whenever a `"realized"`
>   combo is skipped; `_already_solved` was split into a pure `_find_already_solved` (returns
>   the matched `solve_id`) plus a thin bool wrapper, so existing tests needed no signature
>   change. Tests in `tests/test_batch.py`. Applied retroactively to `results/` by re-running
>   `python -m src.batch` once after the fix (0 solved, 10 skipped, as expected).
>
> **A pre-existing data-hygiene issue found, not caused by V6:** `results/` (bess-optimizer)
> and the forecast repo's old `results/*.parquet` carried more than one run for some
> (strategy, model, window) combinations from earlier iterative development (e.g. three
> `full_cooptimized` ceiling runs, two `full_cooptimized`/`seasonal_naive` forecast runs with
> different `net_revenue_eur` under the same nominal inputs). The dashboard and
> `matching_runs` always take the most recently created match, which is the right rule going
> forward, but it means a "v5 baseline" figure quoted here is whichever run last carried the
> `v5` tag -- not necessarily byte-identical to an earlier interim figure quoted mid-build.
> Worth a `results/` cleanup pass; not attempted here (destructive, and out of V6's scope).
> The old V5 forecast-repo result files (16 `.parquet` + `.json`, superseded by V6's 3-model
> re-forecast) were deleted with the user's explicit confirmation before the re-forecast, per
> this plan's own instruction.

## Context

After v5.1 the user wants the study's comparison to be easy to state:
- what a battery earns with **no forecasting model**,
- what it earns **with forecasting**,
- how far both are from the perfect-knowledge limit.

Today the dashboard mixes two capture definitions (headline and realized), spreads one story
over three pages, and carries a digital twin the user no longer uses. The `per_market_best`
blend still reflects V3 accuracy rankings. LightGBM has no wind/solar generation input.

User decisions (2026-10-01):
- **Three levels:**

  | Level | What it is |
  |---|---|
  | Baseline (no forecasting model) | `seasonal_naive`: tomorrow = same day last week |
  | With forecasting | `linear_lags`, `lgbm_point`, `per_market_best` |
  | Ceiling | the optimizer planning on the real prices. Not achievable; reference only |

- **One capture figure:** the plan paid at real prices ÷ the ceiling paid at real prices.
  Headline capture is dropped.
- **Models:** keep `seasonal_naive`, `linear_lags`, `lgbm_point`. Drop `naive_level_adjusted`.
- **LightGBM** gets the ENTSO-E day-ahead wind/solar forecast for the delivery day, plus past
  actual generation.
- **Best per market:** chosen by accuracy on a validation period before the test window.
- **Remove the digital twin.**
- **Dashboard:** 3 pages.
- **Strategies:** both stay.

## V6-1. Three-level comparison and one capture definition (optimizer repo)

**`app/analysis.py`:**
- `oracle_capture` and `model_revenue_table` return a single `capture_pct` (today's
  `realized_capture_pct`). The headline columns are removed.
- Add two EUR columns per model:
  - `cost_of_error_eur` = ceiling net − model net,
  - `value_vs_baseline_eur` = model net − `seasonal_naive` net.
- All on the common days, as today (`common_days`).

**`daily_revenue.parquet`:** it keeps its `expected_*` columns; they are just no longer
displayed. Leaving them avoids touching `rolling.py` and its tests.

**Labels:** "perfect foresight" becomes "ceiling (real prices)" and `seasonal_naive` gets the
tag "baseline" in `app/model_info.yaml` and the pages.

## V6-2. Model line-up (forecast repo)

- **Remove** `naive_level_adjusted`: `src/models/naive_level_adjusted.py`, `_daily.py` (no
  other user), the registry entry, its tests.
- **`config/study.yaml`:** `name: v6`, `models: [seasonal_naive, linear_lags, lgbm_point]`.
- **Old results:** delete its result files from the forecast repo's `results/`. Ask the user
  first.

## V6-3. Wind and solar generation inputs for LightGBM (forecast repo)

**Future input.** ENTSO-E's day-ahead wind onshore / wind offshore / solar generation forecast
for the delivery period (`get_wind_and_solar_forecast`, already in `src/ingest/entsoe.py`).
- Joined to each target with the existing `join_block_mean`: 15-min for day-ahead, block mean
  for FCR/aFRR.

**Past input.** Actual wind and solar generation for the same slot two days before delivery.
- New `get_wind_and_solar_actual` beside the forecast function, using the same `fetch_series`
  monthly cache.
- Availability: `after_delivery_rule`.

**Availability of the forecast.** ENTSO-E only guarantees publication by 18:00 D-1, which is
after every cutoff. The user chose to use it anyway.
- Register the features with `daily_publication_rule(<assumed time>)`.
- The assumed time comes from a new `config/markets.yaml` key
  `entsoe_generation_forecast_published_local`, flagged as an ASSUMPTION in the config.
- It is recorded in every run's metadata and shown in the dashboard's assumptions box.

**Which models use it.** Both trained models (`linear_lags` and `lgbm_point`), so they are
compared on the same inputs. No per-model feature switch is needed. `seasonal_naive` ignores
features.

**Assumed publication time: 07:00 D-1**, before all four cutoffs, so FCR can use it too.

**Backfill.** Only 2024-01 is cached. Ingest the forecast and actuals from `history_start`
to the end of the test window.

**Report.** MAE and skill per market for both trained models vs their V5 figures (day-ahead MAE:
`lgbm_point` 31.43, `linear_lags` 31.17).

## V6-4. Best forecaster per market, chosen fairly

**`config/study.yaml`** gains `validation_window` per market: the 60 days before each test
start.
- With `history_start: 2025-01-01` the first fit still has at least 365 days.
- `run_backtest.py` forecasts validation + test.

**New pipeline step** `pipeline/03_select_blend.py` (optimizer repo):
- It computes MAE per market per model on the validation window only, using
  `analysis.forecast_error_metrics`.
- It writes `config/forecast_blends.yaml`: `per_market_best: {day_ahead, fcr, afrr_pos,
  afrr_neg}`.
- The file gets a generated-by header: window, metric, the MAE table.
- aFRR+ and aFRR− are picked separately.

**Renumbering:** later steps become 04 solve, 05 lookahead, 06 export.

**Scope of the validation window:** the optimizer window and all scoring stay on the test
window only.

## V6-5. Remove the digital twin (optimizer repo)

Delete:
- `src/twin/`,
- `app/pages/digital_twin.py`,
- `tests/test_twin.py`,
- the twin loaders in `app/data_loader.py`,
- `twin_headlines` in `app/analysis.py`,
- twin references in `dispatch_explorer.py`, `streamlit_app.py`, tests, README.

`twin_*.parquet` files inside old run folders are left alone.

## V6-6. Dashboard: 3 pages

1. **Study Overview:** updated to the three levels and the single capture figure.
2. **Forecast comparison** (new). It replaces `forecast_error_cost.py`, `oracle_capture.py`
   and `forecast_accuracy.py`.
   - One table per strategy: model, level, day-ahead MAE, net EUR, capture %, cost of error
     EUR, value vs baseline EUR.
   - An expander with capture per market.
   - The error-vs-revenue scatter.
   - The shape/timing table.
   - A model picker with intro cards.
3. **Dispatch Explorer:** unchanged apart from the twin removal.

## V6-7. Batch resume must notice new forecasts

The V5 gap: after re-forecasting, the batch skipped runs as "already solved". Fix: add the
forecast run ids (`run_meta.forecast_run_ids.*`, compared with `load_forecast_run_ids`) to
the resume check in `src/batch.py::_already_solved`.

## Execution order

0. Save this plan as `bess-optimizer/PLAN_V6_REVISION.md` with a status block.
1. Remove the twin (V6-5) and `naive_level_adjusted` (V6-2); both suites green.
2. V6-1 analysis changes + V6-6 pages; tests.
3. V6-3 ingest + features + availability; V6-4 validation window; V6-7 resume fix; tests.
4. Run the pipeline: ingest backfill, backtest (3 models × 4 markets), select blend, batch
   (2 strategies × (ceiling + 3 models + blend) = 10 runs), export.
5. README / SPEC / `PLAN_V6_REVISION.md`; refresh the report artifact.

## Verification

- Both suites green.
- New tests:
  - a single capture definition;
  - cost-of-error and value-vs-baseline, hand-computed;
  - the generation features are kept at every market's cutoff under the assumed 07:00 time,
    and dropped under the verified 18:00 time;
  - future actual generation is rejected by the availability filter;
  - the blend selection uses only validation dates;
  - resume re-solves when the forecast run ids change.
- Forecast metadata: `kept_features` contains the wind/solar columns for `lgbm_point`; the
  assumption is recorded.
- Dashboard in the browser pane:
  - 3 pages,
  - the three levels are shown,
  - capture ≤ 100% everywhere,
  - no twin page.
- Report: accuracy and capture for V6 vs V5, including if wind/solar did not help.
- No commit unless asked.

## What the real run found

- **`day_ahead_only` (pure arbitrage):** both trained models now finish BELOW the baseline
  (`seasonal_naive` 81.4%). `linear_lags` 76.2% (EUR −10,158 vs. baseline), `lgbm_point` /
  `per_market_best` 67.3% (EUR −27,378 vs. baseline). `linear_lags` was close to the baseline
  under V5 (~79.8%); it fell further behind here, so the wind/solar inputs did not help pure
  arbitrage for either trained model, and arguably hurt `linear_lags`'s.
- **`full_cooptimized` (the realistic strategy):** both trained models now finish ABOVE the
  baseline (93.3%). `lgbm_point` / `per_market_best` 93.4% (+EUR 872), `linear_lags` 94.0%
  (+EUR 4,466) -- `linear_lags` stays the best model overall, same as V5.
- **Blend selection:** `pipeline/03_select_blend.py` picked `lgbm_point` for all four markets
  on validation-window MAE -- `per_market_best` is a plain alias for `lgbm_point` this run, not
  a true blend. It is also not the best-earning model in either strategy (`linear_lags` earns
  more in `full_cooptimized`; `seasonal_naive` earns more in `day_ahead_only`): the validation
  window's fairer, leak-free selection still does not guarantee the revenue-best model, the
  same "accuracy isn't everything" finding V5 reported for `lgbm_point` specifically.
- **`seasonal_naive`** is byte-identical to its pre-V6 run (ignores every feature) -- a sanity
  check that the pipeline change didn't disturb anything it shouldn't have.

## Open decisions

- **`results/` cleanup.** Several `(strategy, model, window)` combinations have more than one
  historical run with different `net_revenue_eur` under the same nominal inputs (see the
  status block). Harmless for now (the dashboard always takes the latest), but worth a pass
  to delete the superseded ones.
- **`pipeline/05_lookahead.py`** has not been run for v6 (it wasn't in the plan's step 4
  list); `results/lookahead_v6.json` doesn't exist, so the Study Overview page's "Would
  looking past midnight help?" section shows "not run yet". Run it if that comparison matters
  for v6 too -- it doesn't depend on the forecast model, so it should reproduce v5's numbers.
- **Report artifact / slide deck refresh** (this plan's step 5) not done as part of this
  status update -- the HTML report and deck from v5.1 still show v5's numbers.
- Nothing in `results/` is committed to git yet (v3 through v6); the history split is still
  the user's call.
