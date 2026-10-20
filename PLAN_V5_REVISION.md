# Plan v5: a year+ of training data, a leaner model set, a correct lag rule, faster solves, one study config and a numbered pipeline

> **Status (2026-09-23):** DONE. Every section (V5-1 through V5-7) implemented, both suites
> green (bess-market-forecast 78 passed, bess-optimizer 270 passed), the full pipeline
> (01-05) run end to end against live data, batch re-solved (12/12, 0 failed), README/SPEC
> updated with the new numbers.
> The forecast repo now has its own `.venv\` (created this run, Python 3.13.9); the
> optimizer repo's is `.venv\` too. `config/paths.yaml`'s `forecast_venv_python` points at it.
> Both repos are still uncommitted (v3, v4, v5); commit only when the user asks.
> Background (DONE, do not redo): `PLAN_V2_DISPATCH.md`, `PLAN_V3_REVISION.md`, `PLAN_V4_REVISION.md`.
>
> **One-off fixes found and made along the way (not in the original plan text):**
> - A DST bug in `bess-market-forecast/src/ingest/weather.py`: Open-Meteo's hourly series
>   never repeats the autumn fall-back hour, so `ambiguous="infer"` raised and silently
>   dropped every October's weather for all 5 regions. Fixed to `ambiguous=False`
>   (an ASSUMPTION, flagged in the code) plus regression tests.
> - The optimizer's batch resume fingerprint doesn't know when the underlying forecast
>   data changes (it only fingerprints battery/solver/assumptions/price_basis config), so
>   after the 02_backtest_forecasts step it initially skipped every combo as "already
>   solved" against the STALE V4 forecasts. Fixed by deleting the 32 stale pre-V5 result
>   files for the kept models (seasonal_naive/naive_level_adjusted/lgbm_point/linear_lags,
>   16 json + 16 parquet) and re-running `python -m src.batch --force`. This is a real gap
>   in the fingerprint's design (it doesn't cover "the forecast repo's results changed
>   underneath you") that a future plan should probably address properly rather than relying
>   on manual cleanup each time.

## Context

V4 showed the trained forecast models (`linear_lags`, `lgbm_point`, `analog_day`) earning far
less than naive lookups. The user's diagnosis (2026-09-23) is that they were starved of data,
and the run metadata supports it. Every market's price history starts **2026-02-01**, so:
- the first fit saw only ~48 days (day-ahead) or ~31 days (FCR/aFRR),
- FCR's first LightGBM fit had ~186 rows.

The retrain is expanding, every 91 days (`RETRAIN_EVERY_DAYS`,
`bess-market-forecast/src/backtest/engine.py`). The raw cache (`data/raw/`) holds Dec 2023,
Jan 2024, then Jan/Feb 2026 onward; everything from 2024-02 to 2025-12 is missing.

A second finding makes the starvation worse. The availability filter drops `lag_24h` for every
market because it treats auction prices as known only after delivery, although they are
published the day before. So the only price history any model sees is 1 week and 4 weeks back.

Separately:
- The 48h lookahead test was not worth its ~3h runtime; the user wants today + 12h instead.
- Timing showed that solves are slow because of model construction, not the solver.

From the BessBidder reference (Miskiw et al., ACM e-Energy 2025) the user adopted two
structural ideas (V5-6, V5-7):
- one study-scope config set before any data is downloaded,
- numbered pipeline scripts whose names show the run order.

This addresses the difficulty of following what feeds what across the two repos.

The remaining V3/V4 open decisions (5.0 vs 4.5 MW reserve cap, reserve backing hours, aFRR
activation placeholders) are deferred until V5's results are in.

## V5-1. Lookahead test: today + next 12h

In `scripts/lookahead_value.py`, rename `_two_day_window_net_eur` to `_lookahead_window_net_eur`.

- **Window:** append only tomorrow's steps before
  `tomorrow.grid.steps[0] + pd.Timedelta(hours=LOOKAHEAD_HOURS)`, plus the matching
  `tomorrow.day_ahead` slice.
  - Use absolute-time arithmetic, the `src/data/grid.py` convention, so DST days slice correctly.
  - `LOOKAHEAD_HOURS = 12`, with a one-line reason: at the noon D-1 gate you know day D; ~12h
    past midnight is the realistic horizon.
- **Unchanged:**
  - the synthetic `DayPrices`/`DayGrid` window,
  - `rolling._solve_day_via_subprocess(ancillary=None)`,
  - committing only today's steps,
  - carrying today's final SOC,
  - `compute_revenue` on today.
- **`--skip-chained` flag** skips config 2, whose value doesn't depend on the window length,
  and prints "chained: skipped".
- **Labels:** "2-day window" becomes "today + 12h lookahead" in the docstring and output.
- **README:** the "Overnight carry / 48h lookahead" section becomes the 12h result. Keep V4's
  48h figure as a one-line historical note.

## V5-2. At least 1 year of training history before the test window

- **Test windows are unchanged** (they now live in `study.yaml`, V5-6):
  - day-ahead 2026-03-21..2026-09-17,
  - FCR/aFRR 2026-03-04..2026-08-31.
- **Backfill the raw cache** from `history_start: 2025-01-01` through 2026-01-31, filling the
  gap. That gives more than 365 days before both test starts. Use the existing clients in
  `bess-market-forecast/src/ingest/` through pipeline step 01:
  - `entsoe.py` for day-ahead prices,
  - `regelleistung.py` for FCR/aFRR,
  - `weather.py` for Open-Meteo.
  - Downloads are network-bound; run them in the background.
  - Any month that fails is logged and reported, never filled in.
- **Load the history.** `HISTORY_BUFFER_DAYS = 30` in `scripts/run_backtest.py` is replaced by
  `history_start` from `study.yaml`, so `load_target_series` loads the older data.
- **Guard.** `run_backtest` warns loudly and records `first_train_days` in the run metadata
  whenever the first training window is shorter than `min_training_history_days: 365`.
- **Day-ahead before 2025-10-01:** that product was hourly. Each hourly price is repeated
  into its four 15-min slots at load time in `src/backtest/data.py` `load_target_series`; the
  raw cache stays immutable. The run metadata records `hourly_upsampled_before: 2025-10-01`.
  Test-window data is native 15-min.

## V5-3. Model line-up

| Model | Kind | V5 |
|---|---|---|
| `seasonal_naive` | lookup (last week) | keep (best realized revenue in V4) |
| `naive_level_adjusted` | last week's shape + Ridge level | keep (2nd by revenue, best DA MAE) |
| `lgbm_point` | LightGBM | keep, retrained on 1+ year |
| `linear_lags` | scikit-learn Ridge | keep, retrained on 1+ year |
| `per_market_best` | blend (optimizer) | unchanged; it uses only the two kept naive models |

- **Drop** `analog_day`, `flat_trailing_mean` and `naive_4w`: their `src/models/*.py` files,
  their registry entries and their tests (`tests/test_spike_models.py` analog_day parts).
  - `_daily.py` stays: `naive_level_adjusted` uses it.
  - Also remove their old result parquet/json files, plus the stale `lgbm_quantile*` ones,
    from the forecast repo's `results/` so the optimizer can't pick them up. Ask the user
    before deleting.
- **Where the list lives:** in `study.yaml` (V5-6), not in `DEFAULT_MODELS`. Update the
  hardcoded combination counts in `tests/test_batch.py`.
- **README:** the V4 information-ladder section stays as a V4 historical record.

## V5-4. Speed

Measured on an idle machine (2026-09-23), one day solve costs ~6s:

| Step | Time |
|---|---|
| Fresh process + imports | ~1.3s |
| `build_model` | ~3-4s |
| `solve_model` | ~0.7-0.9s |

HiGHS itself takes only 0.03-0.06s. The ~3h lookahead run was ~330 sequential solves at
17-33s each under load.

- **Optimizer:** replace the per-step C1 loop (`src/model/build.py:220`, one
  `add_constraints` call per quarter-hour) with a single vectorised constraint.
  - `prev_e` is `e` shifted one step, with the first step taking `e[-1]` if cyclic, else
    `initial_soc_mwh`.
  - Activation becomes one per-step expression from `step_block_index`.
  - The meaning is identical.
- **Forecast repo:** build each market's feature frame once per backtest invocation, one row per
  (cutoff, target), with the availability filter applied per cutoff as today.
  - Slice that frame for each retrain's training set and each test window, instead of
    `_build_training_set` rebuilding every cutoff per retrain per model.
  - Vectorise `join_block_mean`, which is currently a Python loop per row.

## V5-5. Price-lag availability rule

`lag_rule` / `calendar_day_lag_rule` (`bess-market-forecast/src/features/availability.py:34-51`)
say a lagged price is known at its delivery time. They are auction prices, published the day
before delivery.

- **New rule:** a lagged price is available at its delivery day − 1, at the market's
  `results_published_local`.
- **Config:** add `results_published_local` per market to `config/markets.yaml`, sourced from
  the EPEX / regelleistung.net result-publication times, with the source on the same line.
  - If a source can't be confirmed, use gate closure + 2h, flagged as an assumption. lag_24h's
    ~22h margin makes the exact value irrelevant.
- **Effect:** `lag_24h` becomes available for every market. The previous-period lag stays
  dropped, because it falls on day D and is published after the cutoff.
- **New tests:** `lag_24h` is kept at each market's cutoff; a lag landing on day D is still
  dropped. `tests/test_backtest_no_hardcoded_timestamps.py` keeps holding.

## V5-6. One shared study config (adopted from BessBidder)

A single file defines what a study is, before any data is downloaded:
`bess-market-forecast/config/study.yaml`.

It lives in the forecast repo because that repo is upstream. The optimizer already reads that
repo's `config/markets.yaml` the same way (via `config/paths.yaml` `forecast_repo_path`), so
there is no new dependency direction.

```yaml
name: v5
history_start: 2025-01-01          # ingest and train from here
min_training_history_days: 365     # guard in run_backtest (V5-2)
test_window:                       # delivery dates, inclusive
  day_ahead: {start: 2026-03-21, end: 2026-09-17}
  fcr:       {start: 2026-03-04, end: 2026-08-31}
  afrr_pos:  {start: 2026-03-04, end: 2026-08-31}
  afrr_neg:  {start: 2026-03-04, end: 2026-08-31}
models: [seasonal_naive, naive_level_adjusted, lgbm_point, linear_lags]
optimizer:                         # read only by bess-optimizer; ignored by the forecast repo
  strategies: [day_ahead_only, full_cooptimized]
  blends: [per_market_best]        # names from bess-optimizer/config/forecast_blends.yaml
```

**Forecast repo readers:**
- `load_study_config()` in `src/config.py`.
- `scripts/run_ingest.py`: default range = `history_start` .. latest test-window end.
- `scripts/run_backtest.py`: default markets/models/windows from the study; `history_start`
  replaces `HISTORY_BUFFER_DAYS`.

**Optimizer readers:**
- `load_study_config()` in `src/config.py`, via `forecast_repo_path`.
- `src/batch.py` replaces `DEFAULT_START_DATE`, `DEFAULT_END_DATE`, `DEFAULT_MODELS` and
  `DEFAULT_STRATEGIES` with the study.
  - The optimization window is **derived**, not restated: the intersection of the four
    market test windows. That gives 2026-03-21..2026-08-31, today's window.
  - The model list = `models` + `optimizer.blends`.

**Rules:**
- CLI arguments still override the study, so ad-hoc runs keep working.
- Every run's metadata records `study: <name>` (the forecast run JSON and the optimizer
  `run_meta.json`), so the dashboard and `export_runs.py` can group by study.
- **Not** in the batch fingerprint. The study name labels a run; it doesn't change its
  physics or config, so it must not force a re-solve.

## V5-7. Numbered pipeline (adopted from BessBidder)

A new folder `bess-optimizer/pipeline/`. It lives in the downstream repo, which already knows
where the forecast repo is. File names give the run order:

| Script | Runs | Reads | Writes |
|---|---|---|---|
| `01_ingest.py` | forecast repo `scripts/run_ingest.py` over `history_start`..max test end | study.yaml | forecast repo `data/raw/` |
| `02_backtest_forecasts.py` | forecast repo `scripts/run_backtest.py`, study markets × models | study.yaml, `data/raw/` | forecast repo `results/*.parquet` |
| `03_solve_batch.py` | `python -m src.batch` with study strategies × (models + blends) | study.yaml, forecast `results/` | `results/<solve_id>/` |
| `04_lookahead.py` | `scripts/lookahead_value.py` over the derived window, `--skip-chained` | `results/` baseline | stdout + `logs/` |
| `05_export.py` | `scripts/export_runs.py` | `results/` | `results/run_catalogue.csv`, `skip_log_all.csv` |

**Rules:**
- **Thin wrappers only.** Each script reads `study.yaml`, prints what it will run and where the
  output lands, then calls the existing entry point. No logic is duplicated.
- **Forecast steps use the forecast repo's own interpreter.** Steps 01-02 run in a subprocess
  with the forecast repo's venv Python and `cwd` set to that repo, because its dependencies
  (entsoe-py, lightgbm, ...) differ from the optimizer's. The venv path goes in
  `config/paths.yaml`; it is still to be confirmed (see status).
- **Each step is safe to re-run:**
  - ingest reuses its cache,
  - the batch resumes via its fingerprint,
  - the others are read-only or overwrite their own outputs.
- **README gets a "Pipeline" section** with this table. It becomes the one place that shows
  what feeds what across both repos.
- The dashboard stays as-is: `streamlit run app/streamlit_app.py` after step 03.

## Execution order

1. **V5-4 optimizer vectorisation.** Verify identical objectives, then time it.
2. **V5-6 study config.** Add the file and loaders in both repos; switch `batch.py`,
   `run_backtest.py` and `run_ingest.py` defaults to it. Both suites green.
3. **V5-1 lookahead script.** Smoke-run it on 5 days.
4. **Forecast repo changes:** V5-5 lag rule, V5-4 feature caching, V5-3 model removal. Run the
   forecast suite green.
5. **V5-7 pipeline scripts.** Dry-run each: steps 01-03 print their plan without running.
6. **Run the pipeline:**
   - `01_ingest` (V5-2 backfill, background),
   - `02_backtest_forecasts` (4 models × 4 markets),
   - `03_solve_batch` (background),
   - `04_lookahead` while the batch runs,
   - `05_export`.
7. **Docs:** update README (results, Pipeline section, study config), SPEC and this file's
   status with the new numbers.

## Verification

- **Vectorised C1:**
  - the optimizer suite passes unchanged (263),
  - re-solving 3 days of a V4 run (one of them the DST day 2026-03-29) reproduces each day's
    objective within the MIP gap,
  - build time before/after is reported.
- **Study config:**
  - with `study.yaml` in place and no CLI arguments, `python -m src.batch --dry-run` lists
    exactly the study's strategies × (models + blends) over 2026-03-21..2026-08-31,
  - `run_backtest.py` defaults to the study windows,
  - a CLI argument overrides it.
- **Feature caching:** for one market/model over a 2-week window, the new results parquet
  equals the old one (same `y_pred`) with the old lag rule. Then the lag rule change is
  measured separately.
- **Lag rule:** the backtest metadata `kept_features` now contains `lag_24h` for all 4 markets.
- **History:**
  - every run's metadata has `first_train_days >= 365` and `study: v5`,
  - `target_series_start` is at or before 2025-03-04.
- **Pipeline:** each numbered script runs end to end in order. Re-running 01 and 03 does
  (almost) no work, because of the cache and batch resume.
- **Suites:**
  - `pytest tests/ -q` green in `bess-market-forecast`,
  - `.venv\Scripts\python.exe -m pytest tests/ -q` green in `bess-optimizer`.
- **Re-solve:** the batch completes with 0 failures. The per-block reserve check still shows 0
  breaches of 5.0 MW and integer MW commitments.
- **Report:** accuracy (MAE) and realized capture for the 4 models vs V4. Do the trained models
  improve with more data and `lag_24h`? Report it either way.
- **Commits:** none unless the user asks. V3, V4 and V5 are still uncommitted, and the user
  decides how to split them.

## Costs and risks

- Backfill downloads depend on three live APIs, so there may be failed months to re-fetch.
- The forecast backtest will be much larger (1+ year of training rows); the V5-4 caching is
  what keeps it tractable.
- The optimizer re-solve is required: new models and new forecasts.
- The pre-2025-10 hourly day-ahead history has no within-hour shape. That could slightly teach
  the models flatter intra-hour profiles.
- `study.yaml` puts one optimizer-only block (`optimizer:`) inside the forecast repo. That is
  the cost of having a single file both repos read; the forecast repo ignores it.
