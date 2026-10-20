# Plan v3: constraints, forecast models, capture definition, dashboard

> **Handoff for a fresh Claude Code session (PowerShell).** Self-contained.
> Run everything with this repo's venv: `.venv\Scripts\python.exe`.
>
> **Status (2026-09-20):** approved, nothing executed yet. No source code has been
> changed for v3.
>
> Background, all DONE (read for context, do not redo): `IMPLEMENTATION_PLAN.md`
> (forecast-driven optimizer, batch, dashboard) and `PLAN_V2_DISPATCH.md` (battery
> physics v2, sequential clearing, digital twin, dashboard v2).
>
> Two repos are involved, siblings under `self_project/`:
> - `bess-optimizer/` (this repo): the MILP, batch runner, twin, dashboard.
> - `bess-market-forecast/`: trains the forecast models. Only section 2 touches it.

## Context

plan-v2 is finished. Reviewing its results the user asked for seven changes and then
asked for the whole design to be re-verified across both repos before writing this.

## Re-verification result: the design is what the user described

The user's model: `bess-market-forecast` trains forecast models, its prices feed
`bess-optimizer`, the optimizer decides charge/discharge and FCR/aFRR bids under the
battery's physics, and a dashboard shows how forecast differences change battery
performance and revenue. **Audited and confirmed this is what is built:**

- **All four markets are forecast-driven on the decision side.** One price-source swap
  in `src/data/forecast_loader.py:296-325`; the capacity bids `c_fcr`, `c_pos`, `c_neg`
  are decided on forecast capacity prices (`src/model/build.py:208-217`). The forecast
  repo forecasts exactly the four markets the optimizer requires (`_REQUIRED_MARKETS`,
  `src/config.py:21`), enforced at three boundaries.
- **Settlement is always realized prices, and there is no leak in either direction.**
  The settlement object never enters the solver subprocess payload
  (`src/model/rolling.py:118-127`). Every `compute_revenue` call receives settlement
  prices; the only decision-side input is the terminal credit, which is excluded from
  gross/net and is zero under cyclic SOC.
- **Perfect foresight = real price x real-price decision.** Exactly the user's definition.

### Three real findings that were not suspected

1. **Capacity bids are driven by the weakest forecasts.** Capacity plus activation is
   ~88% of gross revenue, yet the forecast repo's own `PROJECT_STATUS.md:80-82` says
   *every* model loses to `seasonal_naive` on FCR and aFRR-, and lists "fix this before
   wiring the optimizer?" as undecided (`PROJECT_STATUS.md:153-156`). It was wired anyway.
2. **The handoff is a convention, not a contract.** The optimizer reads the forecast
   repo's gitignored `results/*.parquet` through one relative path
   (`config/paths.yaml`: `forecast_repo_path: ../bess-market-forecast`). The forecast repo
   writes nothing for it, and its docs (`PROJECT_STATUS.md:120-161`) still describe the
   optimizer as unbuilt future work. All robustness lives on the optimizer side.
3. **Dead code in the parent folder.** `self_project/{src, data, notebook, venv,
   requirements.txt}` are untracked, outside both repos, referenced by neither, and
   superseded by the forecast repo's `src/ingest/`. **Leave untouched** (the notebooks are
   the user's coursework). The forecast repo's `results_15min/` is also unconsumed by the
   optimizer; leave it.

## Decisions taken by the user

| # | Decision |
|---|---|
| 1 | **Models:** keep `seasonal_naive`, `linear_lags`, `lgbm_point`; drop `lgbm_quantile` and `lgbm_quantile_tuned`; add two spike-preserving models. |
| 2 | **Capture headline = the user's definition:** `forecast price x forecast-based decision` against `real price x real-price decision`. |
| 3 | **Run Catalogue page dropped;** replaced by a CSV export script (no new dependency; `openpyxl` is not installed). |
| 4 | **Delete the 9 stale runs** solved with the old 1 MW / 2 MWh battery (window 2026-06-15..06-21). No battery filter. |
| 5 | **Strategies: delete `naive`; hide `full_sequential`.** Keep its code and tests, remove it from batch defaults and the dashboard. |

## 1. Physics and constraint fixes (highest priority, changes every number)

An audit of the four paths that produce a schedule (MILP, sequential, naive heuristic,
twin) found the SOC band `[soc_min, soc_max]` = 1.0-9.0 MWh **is enforced everywhere and
does bind** (`build.py:106-108`; `day_ahead_only` sits at exactly 1.0 MWh ~20% of steps).
It cannot go below zero. The real gaps are elsewhere. All fixes are in
`src/model/build.py` unless stated.

### 1a. Total reserve-power limit (the user's requirement)

Today there is **no** `c_fcr + c_pos + c_neg <= P` constraint. Only per direction:
`p_ch + c_fcr + c_neg <= P_op` and `p_dis + c_fcr + c_pos <= P_op` (`build.py:136-143`);
`c_pos` and `c_neg` never appear in one inequality.

Add, per block `b`: `c_fcr[b] + c_pos[b] + c_neg[b] <= P_limit`.

- Config key in `config/battery.yaml`: `reserve_total_limit: nominal | operational`,
  default `nominal` (5.0 MW, as the user specified). The derated 4.5 MW is arguably the
  true converter limit; record that and let the user flip the key.
- Place the constraint **outside** the `if len(block_steps) == 0: continue` guard
  (`build.py:131-132`). That guard also skips C2/C3 for a block with no steps in the
  day's grid (DST), leaving its `c_pos`/`c_neg` **unbounded while still earning revenue**
  in the objective. The new constraint closes that hole.
- **Measured impact today** (164-day perfect-foresight run): per-block total reaches
  **8.0 MW** (max) on a 5.0 MW nominal / 4.5 MW operational battery, mean 7.1-7.6 MW;
  **85% of blocks exceed P_op and 83% exceed P_nom**. It stops at 8.0 only because the C2
  energy headroom happens to cap aFRR+ plus aFRR- at the 8 MWh usable band, an accident,
  not a power rule.
- **Expect headline revenue to fall materially. That is the point.** Say so in the report.

### 1b. Hard power bounds on `p_ch` / `p_dis`

Both are declared with `lower=0` and no upper bound (`build.py:103-104`). Their caps come
only from C5 (needs `enforce_exclusivity`, `build.py:122`) or C3 (needs `ancillary`,
`build.py:127`). With exclusivity off and day-ahead-only, power is bounded only by the SOC
band: roughly 30 MW on a 4.5 MW battery. Set `upper=battery.power_mw` on both variables.
The existing test masks this (`tests/test_model_full_cooptimization.py:178-191` runs
exclusivity-off only *with* ancillary).

### 1c. SOC regression tests

No test asserts `e >= 0`; the existing ones assert `e >= soc_min - tol`
(`test_model_day_ahead.py:50`, `test_integration_real_data.py:56`, `test_sequential.py:142`,
`test_twin.py:88`). Add tests asserting `e >= 0`, `e >= soc_min`, `e <= soc_max` on every
step for day-ahead-only, co-optimized, DST spring/fall days, **and a config with
`soc_min_fraction: 0.0`** (`src/config.py:73` accepts it; nothing exercises it).

### 1d. Twin

`src/twin/replay.py:166-169` checks only per-direction power, and only counts it. Extend it
to flag the new total-reserve limit.

### 1e. OPEN DECISION for the user (default: unchanged in v3)

C2 backs committed reserve with only `afrr_hours = 1.0` h/MW and `fcr_hours = 0.5` h/MW of
energy (`config/assumptions.yaml:35-47`, flagged UNVERIFIED), against **4-hour** blocks.
Delivering X MW for a full block needs about `4/eta = 4.2X` MWh. So the model can sell
reserve it could not deliver for the whole block; only the twin measures this after the
fact (`activation_shortfall_*`). Record it, surface it, do not change it in v3 without the
user's decision.

### Also state plainly (do not imply otherwise)

Not modelled anywhere: ramp-rate limits, minimum up/down time, C-rate distinct from power,
temperature derating, SOC-dependent power, cycle-life and calendar ageing, inverter
efficiency curve, self-discharge distinct from aux load.

### Tests (new `tests/test_reserve_power_limit.py`, reuse `tests/dispatch_helpers.py`)

- Per-block `c_fcr + c_pos + c_neg <= P_limit` on random days and DST spring/fall days.
- Known-answer: prices where high FCR + aFRR would previously commit more than P; now
  capped.
- The previously unbounded empty-block DST case.
- `p_ch`/`p_dis` bounded with `enforce_exclusivity=False` and `ancillary=None`.

## 2. Forecast models (sibling repo `bess-market-forecast`)

The registry is 27 lines: `src/models/registry.py:14-20`. Adding a model is one new file in
`src/models/` plus one dict entry. Interface (`src/models/base.py:24-42`):
`fit(X, y)` and `predict(X, quantiles) -> DataFrame` with **string** quantile columns
(`"0.5"`). `X` carries a `target_time` column that must be excluded. `tests/test_model_registry.py`
auto-tests every registered model against the interface.

**Remove** `lgbm_quantile`, `lgbm_quantile_tuned`: the two model files, their registry
entries, the `--n-trials` special case (`scripts/run_backtest.py:38,56` and
`tests/test_model_registry.py:37`), the hardcoded display order
(`app/pages/overview.py:42-46`, which silently drops unlisted models), and Optuna if unused
elsewhere.

Record honestly in the report:
- This deletes the best day-ahead model **by error** (MAE 36.99, skill +0.124 vs naive
  MAE 39.57), deliberately, because decision value in euros ranks `seasonal_naive` first in
  every market. Recoverable from git.
- The user's premise "seasonal_naive has the best errors" is half right: on **error**,
  `seasonal_naive` wins FCR and aFRR- but LightGBM wins day-ahead and aFRR+. On **euros**,
  `seasonal_naive` wins everywhere (95.0% vs 93.6% capture in `full_cooptimized`).
  MAE-trained models smooth the price spikes that arbitrage pays for.

**Add two spike-preserving models**, motivated by that finding:

- `naive_level_adjusted`: take the `lag_168h` shape, remove its daily mean, and re-add a
  forecast of the target day's mean level (Ridge on calendar + weather, following the
  `linear_lags` pattern). Keeps naive's sharp profile, fixes its level bias.
- `analog_day`: nearest-neighbour over historical days; distance on calendar + weather
  aggregates + recent level; prediction is the **single** most similar day's profile,
  aligned by time of day. `k=1` by default (averaging over k>1 smooths spikes and defeats
  the purpose); make `k` a kwarg. Rebuild historical daily profiles from `fit`'s
  `(X.target_time, y)`.

Both must tolerate FCR, where the availability filter drops all 8 weather columns
(`src/features/availability.py:147-189`), by falling back to calendar + lags. They inherit
the availability guarantees, so no new leakage surface.

Backtest: `python scripts/run_backtest.py` (4 markets x 5 models). Update the forecast
repo's stale docs (`README.md:146-166` describes an old Jan-2024 window;
`PROJECT_STATUS.md:120-161` calls the optimizer future work).

## 3. Capture definition and whole-window comparison

The user's definition is the headline: `forecast price x forecast decision` against
`real price x real decision`.

- In `run_rolling` (`src/model/rolling.py:253`) add a second `compute_revenue` on
  `pair.decision` and persist `expected_*` columns in `daily_revenue.parquet`. Perfect
  foresight is unchanged (both sides are the same prices there).
- `app/analysis.py::oracle_capture` headline becomes expected / perfect-foresight.
- **Do not delete the realized-settlement columns.** `scripts/da_leg_diagnosis.py`, Forecast
  Error Cost and the twin depend on them.
- One-line risk note in the report, not a re-litigation: this metric scores a model on its
  own beliefs, so a model that over-predicts prices scores higher while earning less real
  money. The realized figures remain in the parquet as the check.
- The user wants comparison on the **whole window**. Re-solve over 2026-03-21..2026-08-31
  (164 solved days) **after** sections 1 and 2 land.

## 4. Dashboard (stays in bess-optimizer)

Today the two dashboards do not overlap and nothing joins them:
- `bess-market-forecast/app` (4 pages): forecast accuracy only, no battery, no euros.
- `bess-optimizer/app` (6 pages): euros only. **No page displays any error metric (MAE,
  RMSE, skill, coverage) anywhere.** Oracle Capture indexes revenue by model *name* only.
- Both default to port 8501 and cannot run together without an override.

Changes (leave the forecast repo's monitor as is):

- **New page "Forecast accuracy -> revenue":** per model, error metrics (MAE, RMSE, skill vs
  `seasonal_naive`) beside net revenue and capture over the whole window. Compute the
  metrics inside bess-optimizer from the forecast repo's parquet, read-only, with **no
  cross-repo import**. Add the function to `app/analysis.py` (pure pandas) with
  hand-computed unit tests, following the existing helpers there.
- **Revenue bars** (`app/pages/dispatch_explorer.py:175-195`): daily bars only; forecast and
  perfect foresight adjacent for each day (a small offset, not the current 12 h at
  `width=0.35` days); **delete both cumulative lines and the `twinx`** (`:177`, `:182`,
  `:188`); intersect on `analysis.common_days` (`dr_window`/`pf_window` are not intersected
  today, `:104-110`); drop the unused `bar_labels` (`:181`); adjust the 4-panel layout
  (`:120`).
- **Remove the quantile axis.** Only `q=0.5` is ever solved (`DEFAULT_QUANTILES = [0.5]`,
  `src/batch.py:54`); all 41 forecast runs on disk are 0.5; every "Quantile" dropdown is a
  one-option box. Remove the four selectors (`dispatch_explorer.py:48-49`,
  `forecast_error_cost.py:42-44`, `oracle_capture.py:45-46`, `sequential_clearing.py:44-45`),
  the `matching_runs` quantile plumbing (`app/data_loader.py:143-166`), `DEFAULT_QUANTILES`,
  `--forecast-quantile` (`src/cli.py:145`), `DA_PROXY_QUANTILE`, and the dead
  `FAN_QUANTILES` (`app/data_loader.py:30`, never referenced). Keep `forecast_quantile` in
  `run_meta.json` for provenance. Only `lgbm_quantile*` ever produced non-0.5 quantiles, so
  section 2 makes the axis fully dead; the day-ahead fan chart falls back to the median line.
  Note it is recoverable from git if quantile-based risk-averse bidding is ever built.
- **Remove pages:** Run Catalogue (`app/pages/run_catalogue.py`) and Sequential vs
  Simultaneous (`app/pages/sequential_clearing.py`). Add `scripts/export_runs.py` writing
  `results/run_catalogue.csv` (one row per run) and `results/skip_log_all.csv`. Keep
  `src/results_catalogue.py`: `src/batch.py:212` resume logic and
  `scripts/da_leg_diagnosis.py` use it.
- Resulting pages: Dispatch Explorer, Forecast Error Cost, Oracle Capture, Digital Twin,
  and the new accuracy -> revenue page.
- Update `tests/test_app_pages.py::test_all_expected_pages_are_registered`, which asserts the
  page set.
- README: document `--server.port 8502` so both dashboards can run side by side.

## 5. Cleanup

- **Delete `naive`:** `run_naive_benchmark`, `simulate_naive_day`,
  `_cheapest_and_priciest_blocks` and its `run_all_benchmarks` entry
  (`src/valuation/benchmarks.py`), the batch entry (`src/batch.py:55,83`), the tests
  (`tests/test_benchmarks.py`, `tests/test_cli.py:88-90`, matrix counts in
  `tests/test_batch.py`), and README/SPEC mentions. This also removes its two physics
  cheats: the aux load silently vanishing at the SOC floor (`benchmarks.py:197`) and a
  dispatch rule that ignores the EUR 8/MWh degradation cost while reporting revenue net of it.
- **Hide `full_sequential`:** keep `src/model/sequential.py`, `FixedReserves`,
  `reserve_stage_decision`, `da_proxy_model` and `tests/test_sequential.py`. Remove it from
  `ALL_STRATEGIES` defaults and the default batch matrix (`src/batch.py:55`) and from the
  dashboard. It remains runnable by explicit `--strategies full_sequential`.
- **Delete the 9 stale runs.** Identify them by `battery.power_mw == 1.0` (they also lack
  `battery.power_nominal_mw`); **list them and confirm before deleting**. Permanent:
  `results/` is gitignored. Also retire the 2026-06-15..06-18 smoke runs once the
  whole-window re-solve exists, so every dashboard picker shows complete data only.
- **Do not touch** `self_project/{src, data, notebook, venv}`; report only.

## 6. Order of work

1. Sections 1 and 5 (constraints, cleanup): optimizer, independent of the rest.
2. Section 2 (forecast models plus backtest): sibling repo, can run in parallel with step 1.
3. Section 3 code and section 4 dashboard.
4. Re-solve `day_ahead_only` and `full_cooptimized`, perfect foresight plus every model,
   over 2026-03-21..2026-08-31 (~2-4 h; the earlier 12-run batch took about an hour). This
   **must come after steps 1 and 2** or the batch runs twice.
5. Docs: README (results tables, dashboard page list, assumptions table), `SPEC.md` (still
   carries v1 battery text in places), and a note in `PLAN_V2_DISPATCH.md` that its phases
   are done.

## 7. Verification

- `.venv\Scripts\python.exe -m pytest tests/ -q` green (currently ~180 tests); the forecast
  repo's suite (`python -m pytest tests/ -q` there) also green.
- **Re-run the per-block check from this plan's research**: for every run, every block's
  `c_fcr + c_pos + c_neg <= P_limit`. Expected **0% breaches, against 85% today**.
- **Report the expected result explicitly:** `full_cooptimized` headline revenue falls once
  reserve commitments are capped (perfect-foresight net was EUR 652,775 over 164 days).
- The two new models appear in the forecast repo's `results/`, and its dashboard still loads.
- Both dashboards launch (8501 and 8502); all optimizer pages render headlessly
  (`tests/test_app_pages.py`) **and** open `streamlit run app/streamlit_app.py` in a real
  browser to check the regrouped bars. Up to now the pages have only been tested headlessly.
- `scripts/export_runs.py` writes both CSVs.

## Open questions carried forward

- **1e:** back reserve with the full 4-hour block energy, or keep the 1 h / 0.5 h
  assumption? Changes revenue a lot; needs the user's decision and ideally the real
  prequalification rule (`assumptions.yaml` flags those hours UNVERIFIED).
- **`reserve_total_limit`:** nominal 5.0 MW (user's wording) or operational 4.5 MW (the
  derated converter)? Default is nominal; one config key.
- **aFRR activation** is still the flat placeholder (`alpha` 0.05, EUR 100/MWh). The user is
  gathering real call statistics separately. ENTSO-E offers aFRR activated-energy *prices*
  but not volumes; regelleistung.net offers only bid stacks
  (see the memory note `project_afrr_activation_data.md`).
- **Capacity bids ride the weakest forecasts** (finding 1 above). Improving FCR/aFRR
  forecasting is the highest-value follow-up after v3, larger than any dashboard change.
