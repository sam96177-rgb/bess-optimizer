# Plan v8: reserve rules aligned with the market, real aFRR energy, separate price charts, LEAR

> **Status (2026-10-03):** BUILT AND RUN for V8-0, V8-1, V8-2 and V8-4 (LEAR); V8-3 (real aFRR
> energy) stopped at the probe, so the aFRR energy is still the flagged placeholder. Nothing
> committed. `study.yaml` is `v8` with `lear` in `models`. V7 files archived (moved, not
> deleted): forecast repo `results_archive/v7/`, optimizer `results_archive/`; the first
> unclipped LEAR files are in `results_archive/v8_lear_unclipped/`.
> RESULTS (162 common days; `results/report_numbers_v8.json`):
> - Reproduction: `day_ahead_only` ceiling 193,341 and seasonal_naive 157,438.59 reproduce
>   exactly (fresh re-solve; fingerprint changed). `seasonal_naive`, `lgbm_point` and
>   `linear_lags` forecasts are identical to V7 in all 4 markets, and so are their
>   `day_ahead_only` results.
> - `full_cooptimized` ceiling 580,444 -> 615,748 (+35,304, +6.1%) from V8-2 alone (no total
>   reserve cap, FCR 1.25 x). Whole window (164 days) 622,878: aFRR- 295k, aFRR+ 128k,
>   day-ahead 77k, FCR 3.5k and the placeholder activation income 128k (20%). V7's 96%
>   capture is not comparable with the new numbers.
> - Capture, `day_ahead_only`: seasonal_naive 81.4, lgbm_point 88.4, linear_lags 90.4, lear
>   76.7 (worst; discharge timing 67.7%), per_market_best 88.4.
>   `full_cooptimized`: 94.7 / 96.9 / 96.8 / 96.7 / 97.1 (per_market_best). The four trained
>   models differ by 0.2 points (about 1.4k EUR of 616k); no significance test was run.
> - LEAR forecast MAE (test window): best in aFRR+ and aFRR-, a near-tie in FCR, second in
>   day-ahead (22.7 vs lgbm_point 21.7). Mean non-zero weights per period model: about 12-23
>   of 106 inputs (day-ahead), 11-17 of 34 (reserves). `per_market_best` now picks `lear` for
>   aFRR+ (validation window) and `lgbm_point` for the other three.
> - Run time: backtest about 25 min for 4 models x 4 markets (LEAR re-run about 12 min);
>   12 solves about 25 min.
> - LEAR needed a fix after the first run: one day-ahead period model (20:45) exploded to
>   ~1e198 on 89 days (solar almost constant in winter training, then nonzero). Fix: clip
>   standardised inputs to +-5 and predictions to the period's training range; a test fails
>   without it. The availability filter was also sped up (shared rule per lag day).
> - V8-0 done: `results/report_numbers_v7.json` written from the V7 runs (before any re-run);
>   exact-quarter timing added to the Forecast Comparison expander; `linear_lags` and
>   `per_market_best` cards fixed, `lear` card added; `test_join_block_mean.py` docstring fixed.
> - V8-1 done: separate FCR and aFRR price panels on the Dispatch Explorer (6 panels when the
>   strategy sells capacity). Looked at by AppTest only, not by eye.
> - V8-2 done in code: C8 removed (and with it `reserve_total_limit` in `battery.yaml`,
>   `config.py`, `battery.py`, fingerprint, run meta, dashboard text, README, SPEC); C3/C4
>   count FCR as `FCR_POWER_FACTOR = 1.25`; `c_pos` / `c_neg` bounded by the 4.5 MW rating;
>   the fingerprint and run meta carry `model.fcr_power_factor`, so every old
>   `full_cooptimized` run is re-solved. Tests: `test_reserve_power_limit.py` rewritten (the
>   1.25 test fails with the factor at 1.0 and passes at 1.25).
> - FOUND while testing V8-2: with no total cap, the placeholder aFRR activation income (5% x
>   100 EUR/MWh, paid in both directions) makes the model sell 4 MW up AND 4 MW down in every
>   block at ANY capacity price, even 0. That is legal under the TSO rules (separate bands) but
>   it inflates the `full_cooptimized` ceiling until V8-3 replaces the placeholder. Report the
>   ceiling before and after V8-2 and V8-3 separately, as planned. Also: FCR no longer reaches
>   0.8 x 4.5 = 3.6 MW in the block where the battery charges for its standby load.
> - V8-3 probe, corrected 2026-10-03 (the first probe tried only the DE-LU code and wrongly
>   called ENTSO-E inconclusive; the 2026-09-19 memory note already had the answer):
>   ENTSO-E HAS aFRR activated-energy PRICES for Germany under the TSO codes (A84, process A16,
>   business A96; TenneT 10YDE-EON------1, TransnetBW 10YDE-ENBW-----N, 50Hertz
>   10YDE-VE-------2; Amprion and DE-LU return "no matching data"). 15-minute, per direction
>   (A01 up, A02 down), EUR/MWh, identical across the three TSOs. March 2026: a price in 85%
>   (up) / 83% (down) of quarter-hours; up mean 92.6, down mean 74.4, down min -1696. ENTSO-E
>   does NOT have activated VOLUMES: A83 is rejected for every combination tried ("not valid or
>   not allowed via this service"). The aFRR- sign convention is still unverified. So the plan's
>   share = activated MW / procured MW cannot be built from ENTSO-E; Netztransparenz data or a
>   flagged share rule is needed. Real prices alone would replace only the 100 EUR/MWh.
> - V8-4 done in code (forecast repo): `src/models/lear.py` + registry entry; the
>   `lear_p_*` price-history block in `src/features/build.py` (`add_price_history_features`,
>   `calendar_day_lag_rule` availability); `select_feature_columns` hides that block from the
>   other models; the engine stores `model_diagnostics` (non-zero weights) in the run metadata.
>   CORRECTED 2026-10-20: day-ahead prices before 2025-10-01 are stored hourly (on the hour
>   only, not spread), so the :15/:30/:45 period models train on about 110 (first retrain) to
>   290 (last) days and only the :00 models on about 430; the earlier "about 430 for every
>   period model" was wrong. The price-history inputs are unaffected (hourly means).
>   Tests: forecast suite 102 pass.
> - Optimizer suite: 4 tests had to change for V8-2 (FCR cap with the 1.25 factor, aFRR price
>   monotonicity on a competing day-ahead, resume fingerprint); final run: optimizer 276 pass (was 279: the three removed
>   tests were about the deleted cap setting), forecast 102 pass.
> - Tests after the run: forecast 103 pass, optimizer 276 pass. Five tests hard-coded the V7
>   study (name, model list, blend picks, matrix size of 10) and were changed to follow the
>   study instead of pinning it.
> - Dashboard: all pages and the `lear` / `full_cooptimized` Dispatch Explorer render without
>   error in AppTest on the V8 results.
> - Not checked: the dashboard by eye (new panels, LEAR rows); `lookahead` (step 05, not
>   re-run). Open: V8-3 data (credentials, CSVs, or keep the placeholder).
> - Git: a `git stash` + `git stash pop` was run by mistake mid-session. Contents restored
>   identically; only effect: `run_catalogue.py` and `sequential_clearing.py` deletions went
>   from staged to unstaged.

## Context

After V7 the user asked for a new forecaster and questioned the reserve revenue. Checks on
2026-10-02 against the German TSOs' prequalification conditions (v1.05, 5 July 2024) and
market description (January 2025) found:

- FCR is correct: MW x block price, `GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]` is per 4-hour
  block (test-window mean 90 EUR/MW per block = 22.5 per hour), capacity only, pay-as-cleared.
- aFRR energy is a placeholder: 5% x 100 EUR/MWh, paid as income in BOTH directions
  (`config/assumptions.yaml`, `src/valuation/revenue.py`, `src/model/build.py`). About 15% of
  the ceiling's net (87,720 of 586,881 EUR).
- The total cap `fcr + pos + neg <= reserve_limit_mw` (C8 in `src/model/build.py`) is not a
  market rule: reserves may be combined if each keeps a separate power band.
- FCR needs battery power of at least 1.25 x the marketed FCR (PQ conditions eq. 3.9); the
  model counts 1.0.
- Energy backing matches: aFRR 60 min per direction (ours 1.0 h); FCR 0.91 MWh per MW
  symmetric in the document's example (ours 0.5 h per side, slightly conservative).
- The V7 reserve price panel puts FCR (per block) and aFRR (per hour) on one axis.

Decisions taken (2026-10-02):
- Reserve limit: per direction only; remove the total cap.
- aFRR activation share: market pro-rata from real data (activated volume / procured
  capacity, per quarter-hour and direction). Plan on a historical average, settle on real.
- aFRR capacity price: keep the auction average, always accepted; flagged as an assumption.
- aFRR+ and aFRR- use the same energy formula, each with its own real price and share;
  aFRR- with the correct payment direction.
- Add LEAR as a 4th forecaster (details in V8-4).

## V8-0. V7 leftovers
Exact-quarter timing in the Forecast Comparison expander; generate `report_numbers_v7.json`
(`scripts/export_report_numbers.py`); re-run the 8 page tests; look at the dashboard; fix the
stale `linear_lags` card in `app/model_info.yaml` (also the `per_market_best` card's list of
three ingredients, once LEAR exists) and the stale docstring in
`bess-market-forecast/tests/test_join_block_mean.py`. (The user may do part of this first.)

## V8-1. Separate price charts (no re-run)
`app/pages/dispatch_explorer.py`: one panel each for day-ahead (EUR/MWh), FCR (EUR/MW per
4-hour block) and aFRR+/aFRR- (EUR/MW per hour), reusing `load_forecast_prices(model, market)`.

## V8-2. Reserve rules (optimizer; re-run of all full_cooptimized runs incl. the ceiling)
- `src/model/build.py`: drop C8; in C3/C4 count FCR as 1.25 x `c_fcr`.
- `config/assumptions.yaml`: record the sources for the 60 min aFRR and FCR energy rules; mark
  them verified; note 0.5 h FCR is conservative vs 0.455.
- Tests that fail without each change (`tests/test_reserve_power_limit.py`,
  `tests/test_reserve_backing.py`).

## V8-3. Real aFRR energy (both repos; needs new data + re-run)
- Ingest (forecast repo, `src/ingest/`): activated aFRR per quarter-hour and direction
  (Netztransparenz CSV/API) and aFRR energy prices per quarter-hour and direction. First
  step is a probe: availability for 2025-01..2026-09, units, and the sign convention of the
  aFRR- price. Missing quarter-hours are skipped and logged, never filled.
- Share = activated MW / procured MW (`GERMANY_ALLOCATED_VOLUME_[MW]`, already in the aFRR file).
- `src/valuation/revenue.py` and `src/model/build.py`: energy cash per quarter-hour =
  MW x 0.25 h x share x price, per direction; SOC dynamics use the same share.
- Decision inputs: a historical average of share and price per quarter-hour of day and
  direction, from data before the delivery day only. Settlement: the real values.

## V8-4. LEAR forecaster (forecast repo; needs backtest + solves)
Decided 2026-10-02: V8 includes a LEAR model (Lago et al. 2021): LASSO regression, one model
per delivery period, past days' prices as inputs, asinh price transform.
One file `src/models/lear.py` plus one `MODEL_REGISTRY` entry (`src/models/registry.py`), a
card in `app/model_info.yaml`, `lear` added to `models` in `config/study.yaml`.
Constraints found: day-ahead is quarter-hourly only since 2025-10-01 (about 110 days of
15-minute history at the first forecast day), the engine retrains every 91 days, and the
`fit(X, y)` interface gives a model one row per delivery period, so the past-day price
vectors need a new feature block in `src/features/build.py` with availability rules.
Decisions (2026-10-02):
- Inputs: hourly-average prices of 1, 2, 3 and 7 days ago (24 x 4 = 96 inputs for
  day-ahead; the 6 block prices x 4 days for FCR/aFRR), the delivery period's wind/solar
  forecast, weekday dummies. One LASSO per period of the day (96 day-ahead, 6 per reserve
  market), fitted inside `lear.py` by grouping rows on the period of day.
- Markets: all four. Model set: added as a 4th model; dropping `linear_lags` is decided
  after the V8 numbers. No second new forecaster in V8.
- Retrain stays at 91 days for every model, so the comparison is on equal footing.
- New feature block in `src/features/build.py` next to `add_lag_features`, each column with a
  `calendar_day_lag_rule` availability rule; the other models keep their current features
  (LEAR selects its columns by name prefix), so their V8 forecasts must equal V7's.
- LASSO strength chosen by cross-validation inside each training window only
  (scikit-learn `LassoCV`, already a dependency); never on the test window.

## Execution order
1. V8-0, V8-1 (no data touched).
2. V8-2 code and tests.
3. V8-3 data probe; report findings before building on them. If the activation data is
   unusable, V8-3 stops here (placeholder stays, flagged) and the user decides; V8-4 does not
   wait for it.
4. V8-3 code and tests.
5. V8-4: first a data-sufficiency count (training rows per quarter-hour model at the first
   and last retrain, and whether hourly-era rows can be used); report it before writing the
   model. Then model and tests.
6. `study.yaml` name `v8`; archive V7 forecast files and runs (ask first); time, then run
   pipeline 02, 03, 04, 06 from bash with plain file redirection (the PowerShell chain died).
7. Update the status block.

## Verification
- Both test suites; `day_ahead_only` figures must reproduce V7 exactly (reserve changes do not
  touch them): ceiling 193,341, seasonal_naive 157,438.59.
- Report the full_cooptimized ceiling before and after each of V8-2 and V8-3 separately, with
  the FCR/aFRR split and mean committed MW.
- One hand-computed quarter-hour of aFRR energy cash in each direction as a test.
- LEAR: a test that its feature columns pass the availability filter at the real cutoffs and
  that no column uses a price published after the cutoff; `seasonal_naive`, `linear_lags`
  and `lgbm_point` forecasts identical to V7; LEAR scored on the same days as the others
  (timing, amount, capture, MAE), and the count of non-zero weights reported per market.

## Costs and risks
- Every full_cooptimized number changes, so V7's 96% capture is not comparable afterwards.
- The energy data may be incomplete for the window, or the aFRR- sign may be unclear; then
  V8-3 stops at the probe and the user decides.
- Pro-rata share and always-accepted average capacity price remain assumptions.
- No intraday market: the rules' recharge-over-4-hours requirement is not modelled.

## Open decisions
- Drop `linear_lags` or not, after the V8 numbers.
- Retrain frequency after V8 (the paper refits LEAR daily), blend picked by MAE vs capture,
  lookahead re-run, committing V7 first, restoring V6 runs for the comparison chart.
- The 4.5 MW power derating (no rule found); a newer PQ document version than July 2024.
