# Plan v7: score plans by timing and amount; give day-ahead models real 15-minute inputs

> **Status (2026-10-02):** built and run; nothing committed. V6 files archived (moved, not
> deleted) to `bess-market-forecast/results_archive/v6/` and `bess-optimizer/results_archive/`.
> Backtest took 8 minutes, the 8 solves about 10 minutes (estimate was up to 3.5 h).
> - Day-ahead forecast step at :00/:15: `lgbm_point` 24.1 -> 6.5, `linear_lags` 21.0 -> 5.0
>   (real prices 9.5). Smoother than real, below the "near 9-10" target. MAE 27.5 -> 21.7 and
>   28.1 -> 23.3. `seasonal_naive` forecasts identical to V6 in all 4 markets.
> - Ceiling runs were reused (config matched), so their totals 193,341 / 580,444 match by
>   construction, not by a fresh re-solve.
> - day_ahead_only capture (162 days): lgbm_point 67.3 -> 88.4%, linear_lags 76.2 -> 90.4%,
>   seasonal_naive 81.4%. full_cooptimized: 96.1% / 95.7% vs seasonal_naive 93.3%.
>   The trained models now beat the baseline in both strategies.
> - Both trained models are under-trading day-ahead (charge 12.8 / 10.4 MWh per day vs ceiling
>   14.3), but timing is up (charge 93-94%, discharge 85-91%). `per_market_best` equals
>   `lgbm_point` (picks it in all four markets).
> - Tests: forecast repo 87 pass, optimizer 279 pass at the last full run.
> - Not checked: the dashboard by eye (reserve panel, new columns), `lookahead` (step 05, not
>   in V7), the optimizer suite after the final re-run.

## Context

Three things the user found in the V6 dashboard (2026-10-01/02):

1. **lgbm_point has the best MAE but seasonal_naive has the best capture.** The user wants a
   model judged by two things: did the battery act at the right time, and did it move the
   right amount. Measured on V6 (day-ahead only, vs the ceiling plan):

   | | seasonal_naive | linear_lags | lgbm_point |
   |---|---|---|---|
   | Charge timing, within 30 min | 88% | 87% | 87% |
   | Discharge timing, within 30 min | 75% | 68% | 61% |
   | Charge amount right | 77% | 73% | 66% |
   | Discharge amount right | 75% | 71% | 63% |
   | MWh charged per day (ceiling 14.3) | 14.5 | 14.7 | 17.8 |

   Same ranking as revenue capture: seasonal_naive's plans really are closer to the ideal.

2. **The trained forecasts oscillate.** Cause found: weather and wind/solar are handed to the
   feature builder hourly, and `join_block_mean` only finds a value for the :00 quarter. The
   :15/:30/:45 quarters get blanks (LightGBM) or zeros (`linear_lags` uses `fillna(0.0)`).
   Evidence: forecast jumps 25 EUR at :00/:15 against 6 EUR at :30/:45 (real: about 9
   everywhere); MAE is 24 on :00 quarters and 32 on the rest. V6's finding "wind/solar did
   not clearly help" is therefore not reliable.

3. **No FCR/aFRR prices on the Dispatch Explorer.** The price panel is hard-coded to
   day-ahead; the reserve forecasts are on disk but never drawn.

Decisions taken (2026-10-02):
- Timing and amount become the headline model scores; net EUR and revenue capture stay as
  smaller columns.
- Headline timing rule: within 30 minutes (2 quarter-hours either side).
- Wind/solar: use ENTSO-E's native 15-minute data (already in `data/raw/entsoe/`, no
  re-download). Weather: straight-line fill from hourly to 15 minutes.
- No new forecasting model in V7. LEAR (Lago et al. 2021) is the candidate for V8, decided
  after the fixed numbers are in.

## V7-1. Real 15-minute inputs for the day-ahead market (forecast repo; needs re-run)

- `src/backtest/data.py`: `load_wind_and_solar_forecast` / `load_wind_and_solar_actual` call
  the existing `get_wind_and_solar_*(..., resample=False)` so the native 15-minute series is
  returned. `join_block_mean` then finds one value per quarter for day-ahead and still
  averages 16 values per 4-hour block for FCR/aFRR (same mean as today).
- `src/features/build.py`: before joining weather, when `market.product_length_hours < 1`,
  put the hourly weather on a 15-minute index with linear time interpolation (one small
  helper next to `join_block_mean`; no fill beyond the last known hour). Fix the
  `join_block_mean` docstring, which still says day-ahead is hourly.
- Availability rules are unchanged: same features, same names, same publication times.
- Invalidates every V6 forecast for `lgbm_point` and `linear_lags` (all four markets are
  re-backtested so run ids stay consistent), `config/forecast_blends.yaml`, and every
  forecast-driven optimizer run. Ceiling runs and `seasonal_naive` forecasts are unaffected
  and should reproduce exactly.

## V7-2. Timing and amount scores (optimizer repo; no re-run)

- `app/analysis.py`: new pure function `plan_agreement(model_long, ceiling_long, days,
  tolerance_quarters=2)` returning, for charge and discharge: timing % (share of the model's
  active quarter-hours with the ceiling active within the tolerance, same day), amount %
  (`1 - sum|daily MWh model - ceiling| / sum daily MWh ceiling`), MWh per day, and wrong-way
  quarter-hours per day. Uses `common_days` and the `p_ch` / `p_dis` rows of
  `long_format.parquet` (via the existing `load_long_format`).
- `model_revenue_table` gains those columns, placed first; `net_eur`, `capture_pct`,
  `value_vs_baseline_eur` follow. `cost_of_error_eur` is dropped from the table (it is
  ceiling minus net, already implied by capture).
- `app/pages/forecast_comparison.py` and `app/pages/study_overview.py`: lead with the new
  columns, one caption line defining each with the worked example (ceiling charges 14 MWh,
  model charges 17, 3 off). Glossary entry updated; exact-quarter timing goes in the
  existing expander, not the main table.
- `scripts/export_report_numbers.py`: add the new figures to the snapshot.

## V7-3. Reserve prices on the Dispatch Explorer (optimizer repo; no re-run)

- `app/data_loader.py`: generalise `load_forecast_prices(model)` to take a `market`
  argument (default `day_ahead`), reusing `load_median_forecasts`.
- `app/pages/dispatch_explorer.py`: when the strategy has capacity rows, add one panel under
  the day-ahead price with forecast (solid) and real (dashed) capacity price for FCR, aFRR+
  and aFRR-, drawn as 4-hour steps in the colours the commitment panel already uses. For
  `per_market_best`, each market's line comes from the model named in
  `config/forecast_blends.yaml`.

## Execution order

1. V7-3 and V7-2 code and tests (no data touched). Check the V7-2 table reproduces the
   figures in Context on the V6 runs.
2. V7-1 code and tests.
3. `config/study.yaml`: `name: v7`.
4. Move the V6 forecast result files and the 8 forecast-driven V6 optimizer runs to an
   archive folder (move, not delete; needed because duplicate forecast files per model break
   `load_forecast_run_ids`). Ask before doing this step.
5. Time the backtest on a few days, state the estimate, then run `pipeline/02`, `03`, `04`,
   `06` as detached processes.
6. Update the status block with what the run found.

## Verification

- Tests, both repos: `.venv\Scripts\python.exe -m pytest tests/ -q` (currently 274 and 81).
- New tests: a day-ahead feature frame has no blank weather or generation value on
  :15/:30/:45; 4-hour block means are unchanged by the 15-minute switch; `plan_agreement` on
  a hand-made two-day case; the Dispatch Explorer renders with the reserve panel (AppTest).
- After the run: mean forecast step at :00/:15 should fall from about 25 EUR to near the
  real 9-10; `seasonal_naive` and both ceiling figures (193,341 and 580,444 on common days)
  must reproduce exactly.
- Dashboard: new columns lead on Forecast Comparison; reserve prices visible under
  `full_cooptimized`.

## Costs and risks

- One full forecast backtest plus 8 optimizer solves; run time to be measured in step 5.
- The fix may not lift the trained models above `seasonal_naive`: smoother forecasts remove
  the false within-hour trades, but an MAE-trained model can still flatten the daily spread.
  That outcome is itself the answer to whether V8 needs a different formula.
- The ceiling plan is not unique when neighbouring prices tie, so timing scores understate
  every model a little; the 30-minute rule limits this.
- Wind/solar forecast availability by 07:00 D-1 is still the V6 assumption.

## Open decisions

- V8: add LEAR, and whether it replaces `linear_lags`.
- Carried from V6: reserve cap (5.0 vs 4.5 MW), reserve energy backing rule, measured aFRR
  activation, lookahead not yet re-run.
