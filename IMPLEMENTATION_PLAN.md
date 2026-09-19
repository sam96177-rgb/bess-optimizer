# Implementation plan: forecast-driven optimizer + decision dashboard

Handoff doc for a fresh Claude Code session (PowerShell). Self-contained —
assumes no prior conversation context.

**Run everything with this repo's own venv:** `.venv\Scripts\python.exe`
(the solver worker spawns subprocesses via `sys.executable`).

---

## Context

Two sibling projects under `self_project\`:

- **`bess-market-forecast`** — price forecasting, 4 German markets, 5 models.
  Real 6-month backtest saved in `results\*.parquet`, schema:
  `market, model, run_id, cutoff_time, target_time, horizon_step, quantile, y_pred, y_true`.
  Has a working Streamlit app in `app\`. 55 tests pass.
- **`bess-optimizer`** (this repo) — MILP battery dispatch (linopy + HiGHS,
  1 MW / 2 MWh) co-optimizing day_ahead + fcr + afrr_pos + afrr_neg.
  73 tests pass; `python -m src.cli --start-date 2026-06-15 --end-date 2026-06-17`
  solves real days.

**Both were verified working on 2026-09-19** (73/73 and 55/55 tests; CLI
solved 3 real days: day_ahead_only €725.79, naive €591.99,
full_cooptimized €3492.01, with a correct warning that 36.7% of the
co-optimized total comes from an assumed activation term).

**The gap.** The optimizer never reads a forecast — `src\data\forecast_loader.py`
originally read only the `y_true` column, i.e. pure perfect foresight.
`SPEC.md` §6 deliberately excluded `y_pred` and any dashboard from v1. So a
"change the forecasting model" control would change nothing. Also: the CLI
writes **no files** (`write_rolling_run_results` is never called) and
`results\` is empty, so a dashboard has nothing to read.

**Goal.** Pick a forecast model and a time window; see what the optimizer
decided, and what the forecast error cost.

### Core design rule: decide on forecast, settle on realized

The MILP is built from `y_pred` (chosen model + quantile); revenue is
always valued at `y_true`. Optimizing *and* valuing at `y_pred` would make
an overconfident forecast look more profitable, because the schedule would
never be checked against reality.

```
cost of forecast error = perfect-foresight revenue − forecast-driven revenue
                         (both settled at y_true, over the same day set)
```

---

## Verified facts (checked against the real data — don't re-derive)

- `(market, model, quantile, target_time)` is **unique** across all 350,710
  forecast rows; exactly 1 cutoff per target_time. No horizon disambiguation needed.
- Models: `seasonal_naive`, `linear_lags`, `lgbm_point` carry **only q=0.5**;
  `lgbm_quantile`, `lgbm_quantile_tuned` carry 7 quantiles (0.05…0.95).
- Coverage: day_ahead 2026-03-21→09-17; fcr/afrr 2026-03-04→08-31.
  **4-market overlap = 2026-03-21 → 2026-08-31 (~164 days).**
- `seasonal_naive` has 10 NaN `y_pred` rows in *each* capacity market, on
  exactly **2026-03-29 and 2026-04-05** (DST day + its one-week echo).
- `src\model\build.py:59-60` and `:76-77` **raise** on NaN prices — so NaNs
  must be caught at the loader, where the skip reason is legible.

---

## STATUS: Part A step 1 is DONE

`src\data\forecast_loader.py` has already been modified and **verified**
(`pytest tests/test_forecast_loader.py tests/test_rolling.py tests/test_benchmarks.py -q`
→ 21 passed). Read the file before changing it. It now contains:

- Updated module docstring explaining y_true vs y_pred.
- `DayPricePair(day, decision, settlement)` — frozen dataclass. `decision`
  is y_pred (or y_true when perfect-foresight); `settlement` is always
  y_true. Deliberately a *pair of* `DayPrices` rather than extra fields on
  `DayPrices`, because `DayPrices` gets pickled into the solver subprocess
  and settlement prices must not be visible to the solver.
- `load_raw_y_pred(paths_config, model, quantile) -> pd.Series` — filters
  to one model+quantile, indexed by `(market, target_time)`. Raises on
  unknown model, on a quantile absent for that model (e.g. `lgbm_point` at
  q=0.05), on missing markets, and on duplicate rows. **Keeps NaN rows
  deliberately** so the day gets skipped with a legible reason.
- `load_price_pair_range(start, end, da_timing, cap_timing, paths_config=None,
  decision_source="realized", forecast_model=None, forecast_quantile=None)`
  → `(list[DayPricePair], list[SkipRecord])`. Skips a day if **either**
  side is incomplete. Forecast runs get `decision: ` / `settlement: `
  reason prefixes; realized runs keep the original unprefixed reason text
  (this is why existing tests still pass — don't "tidy" it).
- `load_price_range(...)` is now a thin wrapper returning the settlement
  side, so all existing callers and tests are unaffected.

---

## Part A step 2 — thread it through the solve loop (DONE)

`src\model\rolling.py`, `src\valuation\benchmarks.py`, and `SPEC.md` §6a
have been updated and verified: `pytest tests/ -q` → 73/73 still pass
(no new tests added yet — that's the separate "Tests to add" section
below), and a smoke run confirms `decision_source="forecast"` earns less
than perfect foresight over the same days (€3479.07 vs €3492.01 over
2026-06-15→17 with `lgbm_quantile_tuned` q=0.5).

### `src\model\rolling.py`

`run_rolling` gains `decision_source`, `forecast_model`, `forecast_quantile`;
switch it from `load_price_range` to `load_price_pair_range`.

The existing nested `_ancillary_for(day_prices)` body stays as-is but must
now be called **twice** — this duplication is mandatory, because
`AncillaryInputs` carries the capacity prices that feed *both* the MILP
objective and `compute_revenue`:

- `_ancillary_for(pair.decision)` → into the solve (subprocess payload
  stays exactly as it is today — pass `pair.decision` only)
- `_ancillary_for(pair.settlement)` → into valuation

Line ~213 becomes:

```python
revenue = compute_revenue(result, pair.settlement.day_ahead,
                          pair.settlement.grid.delta_t_hours,
                          _ancillary_for(pair.settlement))
```

`day_to_long_rows` should keep using `pair.decision.grid` (same grid either
way — both sides are built by the same `build_day_grid`).

`RollingRunResult` gains `decision_source`, `forecast_model`,
`forecast_quantile` so the sidecar writer needs no extra plumbing.

### `src\valuation\benchmarks.py`

Add the same three parameters to `run_naive_benchmark` (it calls
`load_price_range` directly) and thread them through
`run_day_ahead_only_benchmark`, `run_full_cooptimized_benchmark`, and
`run_all_benchmarks`. In `simulate_naive_day`, pick cheapest/priciest block
from `pair.decision.day_ahead`; leave its internal `objective_value` on
decision prices (it is the decision-time expectation, not revenue), and
settle via `compute_revenue` on `pair.settlement`.

### `SPEC.md`

Add a `§6a — Scope extension (Milestone 8)` recording that the §6 bullets
"Feeding the forecaster's y_pred / quantile columns into the optimizer" and
"Any Streamlit or dashboard code" are now **in scope**, with the
decide-on-forecast / settle-on-realized rule and the cost-of-forecast-error
definition written down. Strike through the two superseded §6 bullets with
a pointer — do not delete them.

---

## Part B — persistence, catalogue, batch (DONE)

`write_rolling_run_results` now writes `run_meta.json` last (after both
parquets and skip_log.csv); `src\results_catalogue.py::scan_runs` reads
it back via `pd.json_normalize` (dotted columns like `battery.power_mw`);
`src\batch.py` (`python -m src.batch`) solves the 18-run default matrix
with `scan_runs`-based resume (skips a combo whose strategy/
decision_source/forecast_model/forecast_quantile/date-range AND battery+
solver+assumptions fingerprint all match, unless `--force`) and per-combo
failure isolation (logs a traceback, keeps going). `src\cli.py` gained
`--decision-source`, `--forecast-model`, `--forecast-quantile`,
`--write-results`.

Verified: `pytest tests/ -q` → 73/73 still pass. `python -m src.batch
--dry-run` prints exactly the 18 combos described below. A real 2-combo
smoke run confirmed solve -> write -> second-invocation skip (resume) and
`--force` re-solving; `run_meta.json` fields checked against the schema
below field-by-field (including `forecast_run_ids` sourced from the
sibling repo's actual filenames-as-run_ids). CLI `--write-results` with
`--decision-source forecast` smoke-tested end to end too, then its output
under `results\` was deleted (it was a real write to the shared results
dir, not scratch).

### Sidecar
Extend `write_rolling_run_results` (`src\model\rolling.py:235`) to also
write `run_meta.json` into `results\<solve_id>\`. Fields:

```
solve_id, strategy, decision_source, forecast_model, forecast_quantile,
start_date, end_date, n_days_solved, n_days_skipped,
battery{power_mw, energy_mwh, round_trip_efficiency, soc_min_fraction, soc_max_fraction},
solver{name, mip_gap, time_limit_seconds},
model_flags{cyclic_soc, enforce_exclusivity, include_activation_in_soc},
assumptions{alpha_pos, alpha_neg, activation_energy_price_eur_mwh,
            fcr_reserve_hours, afrr_reserve_hours},
price_basis, total_revenue_eur, forecast_repo_path,
forecast_run_ids{market -> run_id}, git_commit, created_at
```

**Write `run_meta.json` last**, after both parquets, so a crashed run has no
sidecar, stays invisible to the catalogue, and gets retried.

### New `src\results_catalogue.py`
`scan_runs(results_dir) -> pd.DataFrame` — one row per `run_meta.json`.
One discovery path shared by CLI, batch script and dashboard.

### New `src\batch.py` (`python -m src.batch`)
Flags: `--start-date --end-date --models --quantiles --strategies
--include-perfect-foresight --force --dry-run`.

Default matrix over **2026-03-21 → 2026-08-31**: 3 perfect-foresight runs +
5 models × q=0.5 × 3 strategies = **18 runs**. Roughly 1.5–3 h wall time
(~4.5 s/day/strategy; days already run in parallel when `cyclic_soc=True`).
Run it in the background.

Resumability: before each combination, `scan_runs` for a matching
`(strategy, decision_source, forecast_model, forecast_quantile, start_date,
end_date)` plus matching battery/solver/assumptions fingerprint; skip unless
`--force`.

### `src\cli.py`
Add `--decision-source {realized,forecast}`, `--forecast-model`,
`--forecast-quantile` (default 0.5), and `--write-results`.

---

## Part C — dashboard in new `app\` (DONE)

Built `app\streamlit_app.py` (st.navigation, 3 pages), `app\data_loader.py`
(`load_run_catalogue`, `load_long_format`, `load_daily_revenue`,
`load_skip_log`, `load_forecast_prices`, plus `pivot_long_format`,
`load_timings`, `matching_runs` helpers), and the three pages
(`dispatch_explorer.py`, `forecast_error_cost.py`, `run_catalogue.py`),
each with the sibling repo's `_APP_DIR` sys.path bootstrap header and
`if x.empty: st.warning(...); st.stop()` guard. matplotlib + `st.pyplot`
throughout, no plotly/altair. Added `streamlit`/`matplotlib` to
`requirements.txt` and documented `streamlit run app\streamlit_app.py`
in README.md (new "Forecast-driven runs" and "Dashboard" sections,
replacing the stale "Not built yet" entries `SPEC.md` §6a already struck
through).

Found and fixed a real bug while verifying: `write_rolling_run_results`
wrote a column-less `skip_log.csv` for a run with zero skipped days
(`pd.DataFrame([])` has no columns), which `pd.read_csv` can't parse
(`EmptyDataError`) — every clean run's own skip log was unreadable.
Fixed at the source (explicit `columns=["day","reason"]`) plus a
defensive fallback in `app/data_loader.py::load_skip_log` for any
already-written runs from before the fix.

Verified: `pytest tests/ -q` → 73/73 still pass. Populated `results/`
with a real 9-run batch (2026-06-15→18, `seasonal_naive` +
`lgbm_quantile_tuned`, all 3 strategies + perfect foresight) and drove
all three pages headlessly via `streamlit.testing.v1.AppTest` (including
switching the Dispatch Explorer's model/strategy widgets, which exercises
the capacity-commitments stairs panel under `full_cooptimized`) — no
exceptions. Also launched the real server (`streamlit run
app\streamlit_app.py`) and confirmed HTTP 200 / `/_stcore/health` 200.
Smoke-test run directories were deleted afterward (real writes to the
shared `results\` dir, not scratch). Also fixed `.gitignore`: it only
matched top-level `results/*.parquet`/`results/*.csv`, not the
`results/<solve_id>/` subdirectories `write_rolling_run_results` actually
creates — changed to `results/*` / `!results/.gitkeep`.

Mirror the sibling repo's proven pattern at
`..\bess-market-forecast\app\` — read those files first. It uses
`st.navigation` / `st.Page`, `@st.cache_data`, and **matplotlib** +
`st.pyplot(fig)` (no plotly/altair anywhere). Copy its `_APP_DIR` sys.path
bootstrap header verbatim into every page (`st.Page` *execs* page files, so
sibling imports break without it) and its
`if results.empty: st.warning(...); st.stop()` guard.

```
app\streamlit_app.py              st.navigation, 3 pages
app\data_loader.py                @st.cache_data layer
app\pages\dispatch_explorer.py    the main view
app\pages\forecast_error_cost.py  model x strategy comparison
app\pages\run_catalogue.py        runs + skip logs
```

`data_loader.py`: `load_run_catalogue()`, `load_long_format(solve_id)`,
`load_daily_revenue(solve_id)`, `load_skip_log(solve_id)`, and
`load_forecast_prices(model)` reading the sibling repo via
`load_paths_config()` for the quantile fan. Pivot long-format once into a
wide frame keyed by `target_time` per `variable`.

Long-format schema to render (`src\model\rolling.py:29`):
`solve_id, solve_date, target_time, block_id, market, variable, value, unit, is_assumption`
with `variable ∈ {p_ch, p_dis, u, e}` per 15-min step and
`{c_fcr, c_pos, c_neg}` per 4h block.

**Dispatch Explorer** — sidebar: model, quantile, strategy, date-window
slider. Four stacked axes via `plt.subplots(4, 1, sharex=True, figsize=(14,10))`
(sharing x is safe: both grids are tz-aware `Europe/Berlin` from the same
`build_day_grid`):

1. Day-ahead price — `fill_between` for the 0.05–0.95 and 0.25–0.75 fans,
   plus median and realized lines.
2. Battery — SOC (`e`) as a line on a twin axis over `p_ch`/`p_dis` steps.
3. Capacity commitments — `c_fcr`/`c_pos`/`c_neg` per 4h block. Use
   **`ax.stairs(values, edges)`** with `edges = block_starts + [last_block.end]`.
   Plain `ax.step(..., where="post")` renders the final 4h block as a
   zero-width stub unless you append a sentinel point.
4. Revenue — daily bars + cumulative line, forecast-driven vs perfect-foresight.

**Forecast Error Cost** — models × strategies matrix of
`perfect-foresight total − forecast-driven total`. **Must intersect day sets
before differencing** (see Risks) and show the dropped-day count.

Add `streamlit` and `matplotlib` to `requirements.txt`; document
`streamlit run app/streamlit_app.py`.

---

## Tests to add (DONE)

All six added, following the existing hand-verified/real-data style:

- `test_forecast_loader.py` — `load_raw_y_pred` filters correctly (real
  data), raises on unknown model, raises for a quantile absent from a
  point model (`lgbm_point` at q=0.05), preserves NaN rows (verified:
  seasonal_naive's 30 NaN rows = 10 per capacity market on 2026-03-29 +
  2026-04-05). Also covered `load_forecast_run_ids` (added in Part B,
  previously untested).
- `test_price_pair.py` (new) — synthetic: settlement NaN with complete
  decision -> one SkipRecord reading `settlement:`; the reverse ->
  `decision (model qN):`; both-complete gives genuinely different
  decision/settlement prices; `decision_source="realized"` uses y_true
  for both sides.
- `test_rolling.py` — the hand-verified wrong-step test: y_true and
  y_pred have the cheap/expensive 10-step windows SWAPPED on an otherwise
  flat-50 96-step day (mirrors `test_model_day_ahead.py`'s own
  known-answer test). Empirically verified first via a prototype script
  that the decision-side schedule is a full cyclic swing with zero
  dispatch outside the two windows, so settlement has a closed form:
  `-1000 * (delta_e_max / eta)`, strictly less than perfect foresight's
  `+1000 * delta_e_max * eta` on the same day. Both assertions pass.
- `test_run_meta.py` (new) — every field present and correct for a
  forecast-driven run (including `forecast_run_ids`); empty
  `forecast_run_ids` for a realized run; "written last" verified by
  monkeypatching `rolling.build_run_meta` to raise and confirming the two
  parquets + skip_log.csv exist but `run_meta.json` does not.
- `test_batch.py` (new) — `build_matrix` hand-verified counts (18 default,
  scales correctly); `_already_solved` unit-tested against a hand-built
  synthetic catalogue (exact match, each field mismatched one at a time,
  fingerprint mismatch, empty catalogue); end-to-end `run_batch` resume
  (second run: 2/2 `[skip]`, same solve_ids) and `--force` (third run:
  2/2 `[run]`, 2 new solve_ids added alongside the original 2).
- `test_integration_real_data.py` — real 2026-03-29: `seasonal_naive`
  skips with `decision (seasonal_naive q0.5): fcr: ...`; `lgbm_point`
  solves the same day, decision and settlement genuinely different series.

Verified: `pytest tests/ -q` → **101 passed** (73 original + 28 new).
`results/` and git status confirmed clean after the run (all new tests
use `tmp_path`, nothing written to the repo's own `results/`).

## Tests to add — original plan text (superseded by the DONE section above)

Follow the existing style in `tests\` — hand-verified expected values, plus
real-data integration tests. Do not settle for "runs without error".

- `tests\test_forecast_loader.py` — `load_raw_y_pred` filters correctly;
  raises on unknown model; raises for a quantile absent from a point model;
  preserves NaN rows.
- New `tests\test_price_pair.py` — synthetic frames where decision is
  complete but settlement has a NaN → one `SkipRecord` reading `settlement:`,
  and the reverse case.
- `tests\test_rolling.py` — **the most important new test**: a hand-verified
  2-step day where `y_pred` points at the wrong step; assert revenue settled
  at `y_true` equals the hand-computed wrong-step value *and* is strictly
  less than perfect-foresight on the same day.
- `tests\test_run_meta.py` — sidecar carries every field and is written last.
- `tests\test_batch.py` — resume skips a matching run; `--force` overrides.
- `tests\test_integration_real_data.py` — real 2026-03-29 with
  `seasonal_naive` skips with an fcr/afrr reason; the same day solves with
  `lgbm_point`.

---

## Verification

1. `.venv\Scripts\python.exe -m pytest tests/ -q` → existing 73 still pass, plus new ones.
2. In `..\bess-market-forecast`: `python -m pytest tests/ -q` → 55 still pass
   (that repo should need **no** changes).
3. Smoke a forecast-driven run and confirm it earns **less** than perfect
   foresight over the same days:
   ```
   .venv\Scripts\python.exe -m src.cli --start-date 2026-06-15 --end-date 2026-06-17 --decision-source forecast --forecast-model lgbm_quantile_tuned --write-results
   ```
4. `.venv\Scripts\python.exe -m src.batch --dry-run` to confirm the matrix, then run it in the background.
5. `streamlit run app/streamlit_app.py` — actually drive it: change model,
   move the window, confirm all four panels re-render aligned on one x-axis
   and the capacity panel's last 4h block renders full width.

---

## Risks / gotchas (found in the real code — don't rediscover the hard way)

- **`seasonal_naive` will skip 2026-03-29 and 2026-04-05** while every other
  model solves them. Model-vs-model revenue totals are therefore **not**
  directly comparable — the comparison page must intersect day sets, not sum
  raw totals.
- **Autumn DST**: `map_steps_to_blocks` currently *raises* on an uncovered
  fall-back step. 2026-10-25 is outside the 03-21→08-31 window, so this plan
  never hits it — but do not widen the batch window past October without
  fixing that first.
- **Carried SOC is partly an assumption** when `cyclic_soc=False`:
  `result.e` folds in the assumed activation term. Flag it in the sidecar.
- **Window end is 08-31, not 09-17**: even `day_ahead_only` loads all four
  markets through the loader, so days after capacity data ends would skip.
- **2026-09-13** (ENTSO-E published hourly that one day) sits outside the
  default window; leave its skip reason generic — never date-special-case it.
- The solve worker runs with `sys.executable` at `cwd=REPO_ROOT`, so batch
  runs must be launched from this repo's own `.venv`. The dashboard never
  solves, so it is unaffected.

---

## Project hard rules (from SPEC.md — still apply)

1. Never fabricate data. If a price is missing, skip the day with a reason —
   never zero-fill or interpolate.
2. Assumptions live in YAML, are printed in the CLI banner, and must be
   flagged in results (`is_assumption`).
3. The forecast repo is **read-only** from here.
4. Config-driven: no market timing constant may be hardcoded in code.
5. Fail loudly rather than silently guessing.
