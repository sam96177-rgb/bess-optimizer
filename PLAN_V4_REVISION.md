# Plan v4: cleanup, market bid rules, per-market models, forecast-value ladder

> **Handoff for a fresh Claude Code session (PowerShell).** Self-contained.
> Run everything with this repo's venv: `.venv\Scripts\python.exe`.
>
> **Status (2026-09-22): PLAN V4 COMPLETE.** All sections (1, 2, 3, 4, 5a, 5b, 5c, 6, 7),
> the re-solve, and the README/SPEC updates are done, confirmed by clean full-suite runs
> throughout (240 -> 253 -> 263 -> 263 passed, 0 failed at every step). Both repos
> uncommitted -- v3 and v4 should land as separate commits, per the user's own instruction
> below; nothing has been committed without being asked.
>
> **Re-solve done (2026-09-22).** The batch was interrupted mid-run by the user closing the
> process; resumed via a dry-run diff against the log (2 of 16 combos already solved matched
> exactly) and completed the remaining 14. Final: `day_ahead_only` 7 solved + 2 skipped
> (already solved) out of 9; `full_cooptimized` 7 solved out of 7; 0 failed either time.
> `results/` now holds 28 UUID run dirs: the pre-v4 fingerprint no longer matches (new
> `bid_granularity_mw` keys), so the original 12 v3 runs were re-solved under new UUIDs
> rather than reused, plus the 16 v4 combinations -- the old 12 are still on disk
> (gitignored) but orphaned from the catalogue's current fingerprint.
> Notable: `flat_trailing_mean` (cat 0) nets **-36,033 EUR** on `day_ahead_only` -- confirms
> the plan's prediction that a flat forecast collapses DA arbitrage once degradation is
> netted out. On `full_cooptimized` (realized ceiling 586,881 EUR), `per_market_best` is the
> best forecast-driven model at 541,326 EUR but only barely beats plain `seasonal_naive`
> (541,268, +0.01%) -- the blend does not clearly win, as the plan warned it might not.
>
> **5b done (2026-09-22).** `forecast_shape_quality` and `_realized_prices` moved from
> `scripts/da_leg_diagnosis.py` into `app/analysis.py` (renamed `realized_prices`, one
> implementation, two callers); `da_leg_diagnosis.py` now imports both and its `_realized_prices`
> wrapper just adapts the parquet read. Removed now-unused `numpy`/`datetime` imports there.
> `app/pages/forecast_accuracy.py` gained a new "shape, timing, and the price it traded at"
> table: realized buy/sell price, margin, shape correlation, level bias and peak-timing gap
> per model, with a perfect-foresight row for reference.
>
> **5c done (2026-09-22).** New `scripts/lookahead_value.py` compares the three configurations
> (baseline / chained / 2-day window) described below. The 2-day window reuses
> `rolling._solve_day_via_subprocess` (not an in-process loop) to avoid the same
> linopy/HiGHS slowdown over many sequential solves that motivated that subprocess split in
> the first place. Full 164-day window (2026-03-21..2026-08-31): baseline (independent
> cyclic days) **195,493 EUR**, chained (continuity, no lookahead) 191,401 EUR (-2.1%),
> 2-day window (continuity + lookahead) 193,261 EUR (-1.1%). **Counter-intuitive result:**
> continuity alone is a net LOSS vs. the cyclic baseline -- cyclic's "reset to any SOC each
> midnight" is more generous to the optimizer than physical continuity, since every day
> gets a free, independent choice of start/end SOC. Lookahead recovers +1,860 EUR on top of
> chained but doesn't close the gap. Conclusion written into README: do not build the 48h
> model; the measured gain is real but small and outweighed by continuity's own cost.
> Also flagged as a caveat on every other day-ahead headline number in this README (they
> all benefit from the same per-day cyclic freedom).
> Deviations: `DEFAULT_STRATEGIES` lives in `benchmarks.py` (batch imports it; the reverse
> would be circular); `REVENUE_COMPONENTS` stays public because panel 4 reuses it via
> `analysis.stacked_bars`; `HIDDEN_STRATEGIES` still lists the long-deleted `naive`.
>
> **5a done:** `flat_trailing_mean` and `naive_4w` added to
> `bess-market-forecast/src/models/` and registered. Backtests run and saved to that
> repo's `results/` over the exact existing windows (day_ahead 2026-03-21..2026-09-17,
> 17372 rows each; fcr/afrr_pos/afrr_neg 2026-03-04..2026-08-31, 1086 rows each --
> matching every other model's row count). `flat_trailing_mean` verified piecewise-constant
> across the two quarterly retrain windows (not a peek at future prices). Forecast repo
> suite: 75 passing, up from 69. Not yet done: scoping cat 0/1 to `day_ahead_only` in the
> optimizer batch (section 4/re-solve step) and wiring these two models into the ladder
> table (section 5b).
>
> **Section 4 done (2026-09-22):** single-day solve time 1.12x (well under the 2x stop
> threshold); revenue fell 3890.67 -> 3725.82 EUR on that day, as expected. Full suite
> confirmed green: 253 passed, 0 failed.
>
> Files touched for section 4: `config/market_prices.yaml`
> (restructured: `price_basis:` + new `bid_granularity_mw:` sections), `src/config.py`
> (`MarketPriceBasisConfig.bid_granularity_mw` + `.granularity_for()`), `src/model/build.py`
> (`_constrain_to_granularity`, skipped per-market when `fixed_reserves` already pins that
> variable -- sequential clearing's earlier-stage commitments would otherwise conflict with
> the new integer constraint), `src/model/rolling.py` (`build_run_meta` records
> `market_prices.bid_granularity_mw`), `src/batch.py` (`_fingerprint` takes `price_basis`,
> adds 4 per-market granularity keys), `tests/dispatch_helpers.py` (`solve_case` takes an
> optional `price_basis` override), `tests/test_batch.py`, `tests/test_run_meta.py`,
> `tests/test_reserve_backing.py` and `tests/test_reserve_power_limit.py` (known-answer
> values updated: FCR's 3.6 MW cap floors to 3.0; the operational 4.5 MW reserve cap floors
> to 4.0 since two integer MW can't sum to an odd half), new `tests/test_bid_granularity.py`.
>
> **Section 3 done (2026-09-22).** Full suite confirmed green: 263 passed, 0 failed. New
> `config/forecast_blends.yaml` (`per_market_best`), `src/config.py`
> (`ForecastBlendsConfig`/`load_forecast_blends_config`), `src/data/forecast_loader.py`
> (`load_raw_y_pred` resolves a blend name via new `_load_raw_y_pred_blend`, one call per
> DISTINCT underlying model then sliced per market by the existing MultiIndex -- all
> existing guards run unchanged, once per underlying model; new `resolve_per_market_models`
> for provenance; `load_forecast_run_ids` also made blend-aware, since it crashes on a
> blend name otherwise -- not explicitly in the plan text but required for a blend run to
> write results at all), `src/model/rolling.py` (`build_run_meta` gains `forecast_models:
> {market: model}`, the plan's stated shape), `src/batch.py` (`DEFAULT_MODELS` +=
> `per_market_best`, now 6 models). Nested nested-blend names would resolve recursively
> (untested, not in scope). Dashboard: no changes -- `per_market_best` isn't a real model in
> the sibling repo's own parquet files, so the day-ahead forecast-price panel and the
> forecast-error table both gracefully show "no data" for it rather than crashing; only the
> revenue-by-model tables (keyed on this repo's own run catalogue) show it. Tests: new
> tests in `test_config.py` (blend config + the granularity config tests section 4 had not
> added), `test_forecast_loader.py` (blend resolution matches each market's own model,
> against real data), `test_run_meta.py` (`forecast_models` field, including an end-to-end
> blend solve); `test_batch.py`'s hardcoded combination counts updated (12->14, 10->12) for
> the 6th default model.
>
> Background, all DONE (read for context, do not redo): `IMPLEMENTATION_PLAN.md`,
> `PLAN_V2_DISPATCH.md`, `PLAN_V3_REVISION.md`. v3 is finished and green
> (240 tests here, 69 in the forecast repo) but **not committed** in either repo.
>
> Two repos, siblings under `self_project/`:
> - `bess-optimizer/` (this repo): MILP, batch runner, twin, dashboard.
> - `bess-market-forecast/`: trains forecast models. Only section 5a touches it.
>
> **`results/` currently holds exactly the 12 v3 runs** (2026-03-21..2026-08-31,
> `reserve_total_limit: nominal`). It is gitignored. All older runs were deleted
> with the user's confirmation. A Streamlit server may still be running on port 8501.

## How to write the code (read before touching anything)

The user's explicit instruction for v4, and the reason section 7 exists:

- **Comments only where code cannot speak.** A unit, a market rule and its source, or
  why an obvious-looking alternative is wrong. Never restate the line below it.
  v3 over-commented; do not repeat that.
- **Modify before adding.** Before adding a feature, constraint or file, check whether
  an existing function, config key or code path can carry it. A new constraint must
  state its cost: added solve time, and what it invalidates in `results/`.
- **One-line docstrings** unless the function encodes a market rule or a decision that
  would otherwise read as a bug.
- Do not add defensive branches for states callers cannot produce.

## Context

Reviewing v3 the user asked five things. Three needed no code, one exposed a real
modelling gap, and one redirected the 48-hour question. Those answers are recorded
below so this session does not re-derive them.

### Verified facts (do not re-research)

**Minimum trade size — the one real gap.**
- **FCR / aFRR:** minimum bid **1 MW**, bid increment **1 MW**, 4-hour products, daily
  auctions ([regelleistung.net tendering page](https://www.regelleistung.net/en-us/Become-a-balancing-service-provider/Tendering-and-bidding-process)).
  On a 5 MW battery, reserve bids may only be 0,1,2,3,4,5 MW. **The model currently bids
  fractional MW freely, so v3 reserve revenue is overstated.** → section 4.
- **Day-ahead (EPEX SPOT):** minimum order 0.1 MW, price tick 0.1 EUR/MWh. On a 4.5 MW
  battery this is a rounding detail. EPEX's site refuses automated fetches, so this
  figure is **secondary-source, unverified** — record it as such and do **not** enforce
  it (192 extra integer variables per day to move the third decimal).

**"twin draw 0/1/2" — a labelling problem, not a model problem.** The twin replays one
stored plan `--n-draws` times (default 500) under random aFRR activation; each draw is
one independent scenario. `twin_paths.parquet` keeps step paths for only the first few
(`n_path_draws`), which is why three appear. → section 6.

**The battery has no lookahead past midnight, but its within-day optimisation is already
complete.** `build_model` receives all 96 quarter-hourly prices at once and solves them
as one problem — it does not walk forward deciding "trade or not". It already charges at
the day's local minima and discharges at its maxima under every constraint. Evidence from
the v3 164-day run: perfect foresight achieves a realized buy price of **47.1** EUR/MWh
and sell price **176.9** (margin 129.9). What is missing is only the *cross-midnight*
view: `cyclic_soc=True` forces SOC back to its starting level each midnight, so cheap
night energy cannot be carried into an expensive morning.

**A 48-hour horizon is blocked on data, not just code.** `markets.yaml` declares
`horizons_hours: [24, 48]` for day_ahead, but `bess-market-forecast/src/backtest/engine.py:65,100`
filters every backtest to `min(horizons_hours)` = 24, and FCR/aFRR declare 24 only. So no
D+1 forecasts exist. Adding them would also break `load_raw_y_pred`'s one-row-per-period
guard until the loader disambiguates on horizon.

**`lag_24h` is not available for day-ahead.** The availability filter drops it: at the
11:00 D-1 cutoff, the late hours of D-1 have not happened. Confirmed in the day-ahead run
metadata: `dropped: ['lag_0.25h', 'lag_24h']`, `kept: [... 'lag_168h', 'lag_672h' ...]`.
So "yesterday's prices" cannot be a baseline; the crudest honest lookup is 4 weeks back.

### Decisions taken by the user

| # | Decision |
|---|---|
| 1 | **1 MW reserve bid granularity: config flag, default ON.** It is the real rule; expect revenue to fall again. |
| 2 | **Per-market models: named blend.** `forecast_model` stays a flat scalar everywhere. |
| 3 | **48h horizon: measure the upside first**, then decide. Reframed by the user into the information ladder in 5a. |

---

## 1. Dead code and duplication

Confirmed by call-site search, not inspection.

**Delete outright — zero call sites:**

| File | Symbol |
|---|---|
| `app/data_loader.py:174` | `filter_proxy` |
| `app/data_loader.py:68` | `load_skip_log` (duplicates `src/results_catalogue.py:46 scan_skip_logs`) |
| `app/data_loader.py:134` | `available_forecast_models` |
| `app/analysis.py:84` | `sequential_comparison` |
| `app/analysis.py:105,15` | `proxy_label`, `NO_PROXY_LABEL` (only reachable via `filter_proxy`) |
| `src/valuation/benchmarks.py:97` | `STRICT_DA_PROXY_MODEL` |

Drop their tests too: `tests/test_app_analysis.py` `test_sequential_comparison_hand_computed`
and `test_proxy_label` (~lines 109-124).

**The big duplication — 8 near-identical parameter lists for one call.**
`src/valuation/benchmarks.py` has three `run_*_benchmark` wrappers that only forward 12-14
identical kwargs to `run_rolling`, each adding one or two literal flags; `run_all_benchmarks`
re-lists them a fourth time; and `src/batch.py:61-94 _run_strategy` is a **second, independent**
dispatch table over the same three strategies.

Collapse to one table and one entry point in `benchmarks.py`:

```python
STRATEGY_FLAGS = {
    "day_ahead_only": {"include_ancillary": False},
    "full_cooptimized": {"include_ancillary": True},
    "full_sequential": {"include_ancillary": True, "sequential": True},
}

def run_strategy(name, **kwargs) -> RollingRunResult:
    if name not in STRATEGY_FLAGS:
        raise ValueError(f"Unknown strategy '{name}'. Must be one of {list(STRATEGY_FLAGS)}")
    return run_rolling(**kwargs, **STRATEGY_FLAGS[name])
```

`run_all_benchmarks` becomes a dict comprehension over the requested names;
`batch._run_strategy` is deleted and `batch.run_batch` calls `run_strategy`;
`ALL_STRATEGIES` becomes `list(STRATEGY_FLAGS)` so the two lists cannot drift.
Roughly −120 lines. Keep `da_proxy_model` routing (sequential-only) intact.

**Real inconsistency to fix:** `src/cli.py:170` calls `run_all_benchmarks`, which
unconditionally runs `full_sequential` — the strategy v3 hid from the batch and the
dashboard. Give `run_all_benchmarks` a `strategies` argument defaulting to
`batch.DEFAULT_STRATEGIES` and have `cli.py` pass it.

**Smaller fixes:**
- `app/data_loader.py:122 load_forecast_prices` already calls `load_median_forecasts` then
  filters 6 lines — drop its `@st.cache_data` (it re-caches an already-cached frame).
- `app/pages/forecast_error_cost.py:68-73` hand-rolls a day intersection in a local named
  `common_days`, shadowing `analysis.common_days`. Use the shared helper.
- `src/model/rolling.py:48` docstring still names the deleted `naive` strategy.
- Privatise constants with no external importer: `REVENUE_COMPONENTS`, `ERROR_COLUMNS`,
  `REVENUE_TABLE_COLUMNS`, `CAPTURE_COLUMNS` in `app/analysis.py`. (`CAPTURE_ROWS` **is**
  imported by `app/pages/oracle_capture.py` — leave it public.)
- Tests: remove unused imports (`test_app_pages.py` `glob`/`os`; `test_model_day_ahead.py`
  `math`; `test_model_full_cooptimization.py` `pytest`; `test_reserve_backing.py`
  `PRICE_BASIS`; `test_reserve_power_limit.py` `np`; `test_block_invariance.py` `grid_for`).
  Settle on `from tests.dispatch_helpers import ...` everywhere (two styles in use).
  Delete the five redundant in-body re-imports in `tests/test_sequential.py`. Drop two
  tautological `naive` assertions (`test_cli.py:89`, `test_batch.py:27`).
- 10 test files re-load `BATTERY`/`SOLVER`/`ASSUMPTIONS`/`TIMINGS` that
  `tests/dispatch_helpers.py:24-28` already exposes. Import from there **only where it is a
  pure substitution** — do not restructure a test to make it fit.

## 2. Stacked revenue bars

`app/pages/dispatch_explorer.py` panel 4. Replace each plain bar with a stack of the five
components already in `daily_revenue.parquet` (`day_ahead`, `fcr`, `afrr_pos`, `afrr_neg`,
`activation`), keeping the forecast-driven and perfect-foresight stacks adjacent per day.

**The one real detail: `day_ahead` can be negative** on days the battery is a net buyer, so
a naive `bottom=` accumulation silently mis-stacks. Accumulate positive and negative parts
against separate baselines. Overlay `net` as a short black tick so degradation (a cost, not
a market) stays visible without being stacked. Under `day_ahead_only` the stack degenerates
to one component — that is fine, do not special-case it.

Reuse `analysis.common_days` for the day intersection (already applied in v3).

## 3. Per-market forecast models (named blend)

The v3 accuracy table supports it: `naive_level_adjusted` wins day-ahead (MAE 34.0) while
`seasonal_naive` wins FCR (27.6), aFRR+ (4.2) and aFRR− (6.6).

**Why a blend name and not a per-market dict:** `forecast_model` is a flat scalar that
batch resume, `scan_skip_logs`, `export_runs` and five dashboard pages all filter on. A dict
would flatten to `forecast_model.day_ahead` via `json_normalize` and leave **no flat column**
— and `matching_runs` / `_already_solved` both test `forecast_model.isna()`, so such runs
would be silently **mistaken for perfect-foresight runs**.

- New `config/forecast_blends.yaml`:
  `per_market_best: {day_ahead: naive_level_adjusted, fcr: seasonal_naive, afrr_pos: seasonal_naive, afrr_neg: seasonal_naive}`
- `src/data/forecast_loader.py:104 load_raw_y_pred` resolves a blend name into one
  `load_raw_y_pred` call per distinct underlying model, then concatenates each market's slice
  from its own model. Keep the `(market, target_time)` MultiIndex so `load_day_prices` and
  everything downstream is unchanged. **Keep the existing guards** (required markets present,
  one row per market/target_time, NaNs preserved).
- `build_run_meta` records `forecast_models: {market: model}` as a nested dict — the shape
  `forecast_run_ids` already uses, so it flattens harmlessly to dotted columns.
- Add `per_market_best` to `DEFAULT_MODELS` in `src/batch.py`.

Precedent: `da_proxy_model` already builds a second price view from a different model
(`forecast_loader.py:303`, `rolling.py:209`), so mixing models per run is established.

## 4. 1 MW reserve bid granularity

Config-driven, **default ON**.

- `config/market_prices.yaml` gains `bid_granularity_mw` per market: `1.0` for `fcr`,
  `afrr_pos`, `afrr_neg`; `0` (unconstrained) for `day_ahead`. Extend
  `MarketPriceBasisConfig` in `src/config.py`. That file already holds per-market trading
  conventions, so this is an extension, not a new concept.
- `src/model/build.py:114-119`: declare `c_fcr`/`c_pos`/`c_neg` as integer variables scaled
  by the granularity when it is non-zero. This satisfies the 1 MW **minimum** automatically —
  an integer grid admits 0 or ≥1 MW and nothing between. The existing C4 FCR cap and the v3
  C8 total-reserve cap must still hold.
- **Measure the solve-time change on one day before launching the batch.** 18 integer
  variables per day on top of 96 binaries. If it is worse than ~2x, stop and report rather
  than starting a 4-6 hour batch.
- Add `market_prices.bid_granularity_mw` to `src/batch.py::_fingerprint` so pre-v4 runs do
  not resume as valid — same pattern as `battery.reserve_total_limit` in v3.
- New `tests/test_bid_granularity.py`: committed reserve is integer MW on random and DST
  days; a known-answer case where the unconstrained optimum is fractional and the
  constrained one is not; granularity `0` reproduces today's fractional behaviour.

Expect reserve revenue to fall. Report it as the correction it is, not as a regression.

## 5. The value of forecast information

### 5a. Information ladder

Run the same MILP with identical physics and identical settlement on realized prices,
varying **only the information the decision sees**. Every rung is a forecast model, so this
needs **zero optimizer changes** — the existing capture, error-cost and accuracy pages pick
them up automatically. This is also a sounder version of the deleted `naive` benchmark,
whose flaw was bypassing the MILP's physics; here every rung faces the same constraints.

| Cat | Information | Implementation |
|---|---|---|
| 0 | **No shape at all** — one flat price per day | New `flat_trailing_mean`: predicts a constant, the mean of known history. Expect DA arbitrage to collapse to ~0 since degradation exceeds any gain. The floor that proves shape is what pays. |
| 1 | **Crudest legitimate lookup** — same period 4 weeks ago | New `naive_4w`: returns `lag_672h`, mirroring `seasonal_naive`'s 15 lines. |
| 2 | **Naive** — same period last week | `seasonal_naive` (exists) |
| 3 | **Trained / spike-preserving** | `linear_lags`, `lgbm_point`, `naive_level_adjusted`, `analog_day`, `per_market_best` |
| 4 | **Oracle** — real prices | perfect foresight (exists) |

Both new models: one file each in `bess-market-forecast/src/models/` plus one registry line
in `src/models/registry.py`, following `seasonal_naive.py`. Then
`python scripts/run_backtest.py` over the four markets on the **same windows the existing
results use** (day_ahead 2026-03-21..2026-09-17; fcr/afrr 2026-03-04..2026-08-31).

**Scope cat 0 and cat 1 to `day_ahead_only`** in the optimizer batch: in `full_cooptimized`,
reserve capacity is 76% of revenue and would mask the arbitrage effect the ladder exists to
show.

### 5b. Surface the shape diagnostics

The user's thesis — a wrong forecast puts the local minima and maxima in the wrong
quarter-hours, so the battery commits to the wrong hours — **is already measured** by
`scripts/da_leg_diagnosis.py`, which nothing surfaces. On the 164-day window it reports
`seasonal_naive` shape correlation 0.85 with a realized margin of 111 EUR/MWh against
`lgbm_point` 0.79 and 72.8. The better-shaped forecast bought lower and sold higher.

Move `forecast_shape_quality` and `_realized_prices` from that script into `app/analysis.py`
(one implementation, two callers — same dedupe principle as section 1) and add to the
"Forecast accuracy → revenue" page, per model: **realized buy price, realized sell price,
margin, shape correlation, peak timing gap**. The page then reads error → shape/timing error
→ euros lost, which is the direct answer to "why does accuracy matter".

### 5c. Overnight carry (the literal 48h question), trimmed

Cheap **only if scoped to `day_ahead_only`**: with `ancillary=None` there are no capacity
blocks, so the block indices that restart at 0 each day — the hardest obstacle to a real
multi-day model — do not arise. Perfect foresight only, so no new forecasts are needed.

New `scripts/lookahead_value.py` compares three configurations over the same day set:
1. **Baseline** — independent cyclic days (from the existing v3 run).
2. **Chained** — `cyclic_soc=False`, SOC carried across midnight: continuity, no lookahead.
3. **2-day window** — one model over two concatenated days, commit the first day, roll.
   Reuses `build_model` unchanged: without ancillary it needs only a `steps` index and a
   matching price Series.

If the gain is small, document the 48h horizon as not worth building. If it is large, that
justifies a v5: block-index namespacing across days, per-delivery-date result splitting, and
48h forecasts in the sibling repo.

## 6. Digital twin labelling

`app/pages/digital_twin.py:87,101` — rename "twin draw N" to "random activation scenario N"
and state in the caption that each is one independent random draw of aFRR calls against the
same plan, that only the first few are stored, and that the headline percentages come from
all `meta['n_draws']` of them. No model change.

## 7. Coding skill file

Create `.claude/skills/writing-simulation-code/SKILL.md` in this repo so these rules apply
automatically in future sessions instead of being restated:

```markdown
---
name: writing-simulation-code
description: Conventions for writing and changing code in the bess-optimizer and bess-market-forecast repos - concise code, modify before adding, never fabricate market data, verify against the existing suite. Use when editing any Python in either repo.
---

# Writing simulation code

## Before adding anything
1. Search for a function that already does it. `app/analysis.py`, `tests/dispatch_helpers.py`,
   `src/model/rolling.py` and `src/data/forecast_loader.py` hold most shared logic.
2. Prefer changing a signature, a config key, or an existing code path over adding a parallel one.
3. A new constraint, variable or strategy must state its cost: added solve time, and what it
   invalidates in `results/` (the batch fingerprint forces a re-solve).

## Code style
- No comment that restates the code. Comment only what code cannot carry: a unit, a market rule,
  a source, or why an obvious-looking alternative is wrong.
- One-line docstrings, unless the function encodes a market rule or a decision that would
  otherwise read as a bug.
- Name the quantity and its unit: `reserve_limit_mw`, not `limit`.
- No defensive branches for states the callers cannot produce.

## Data and assumptions
- Never fabricate, interpolate or zero-fill a missing price. Skip the period and log the reason.
- Every assumed number lives in `config/assumptions.yaml` and is flagged where it reaches a result.
- A market rule needs its source in the same line: "1 MW minimum bid (regelleistung.net)".
- Decision prices and settlement prices never mix: decide on forecasts, settle on realized.

## Verification
- `.venv\Scripts\python.exe -m pytest tests/ -q` before reporting anything done.
- A physics or constraint change needs a test that fails without it.
- Report the number that moved and by how much, including when it moved the wrong way.
```

Sources behind it, worth the user's own reading:
- [Anthropic skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices) — "concise is key", no voodoo constants.
- John Ousterhout, *A Philosophy of Software Design* — deep modules; comments record what code cannot express.
- Wilson et al., [Best Practices for Scientific Computing](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.1001745), PLOS Biology 2014.
- [Ten Recommendations for Engineering Research Software in Energy Research](https://arxiv.org/abs/2502.13510) — this exact domain; #1 "reinventing is a waste of time", #6 structure, #8 testing.
- [Google engineering practices](https://google.github.io/eng-practices/).

---

## Order of work

1. **Section 1** (cleanup) — no behaviour change. Suite must stay green at 240 minus the two
   deleted dead-code tests. Do this first; everything else builds on smaller files.
2. **Sections 2 and 6** (dashboard) — read-only against `results/`, verifiable immediately.
3. **Section 7** (skill file).
4. **Section 5a's two models** in the sibling repo + their backtests. Must precede the batch
   or it runs twice.
5. **Section 4** (bid granularity), then **section 3** (blend). Both change what gets solved.
   Time one day's solve after section 4 before going further.
6. **Re-solve once**, covering sections 3, 4 and 5a together.
7. **Section 5b** (dashboard diagnostics) and **5c** (overnight carry) — independent; 5c can
   run while the batch runs.
8. Update `README.md` results tables (they carry v3 numbers throughout) and the ladder table;
   note the bid-granularity constraint in `SPEC.md` beside C8.

## Verification

- `.venv\Scripts\python.exe -m pytest tests/ -q` here — 240 today; expect ~245 after the new
  granularity tests and the removed dead-code tests.
- `python -m pytest tests/ -q` in `bess-market-forecast` — 69 today; expect ~71 once the two
  ladder models register (`test_model_registry.py` auto-tests every registered model).
- Single-day solve timing with `bid_granularity_mw: 1.0` vs `0`, **before** the batch.
- Re-run the v3 per-block check across all co-optimized runs: every block's
  `c_fcr + c_pos + c_neg <= 5.0 MW` **and** every committed reserve an integer MW.
- `python scripts/lookahead_value.py` prints the three EUR figures.
- The ladder reads monotonically or explains why not: cat 0 should earn ~nothing on day-ahead
  arbitrage, each rung up should buy lower and sell higher. If a rung breaks the order, report
  it — v3 already showed the lowest-error model is not the top earner.
- Dashboard: `tests/test_app_pages.py` headless, then `streamlit run app/streamlit_app.py`
  and check the stacked bars **on a day where `day_ahead` is negative** (the sign case).
- `python scripts/export_runs.py` still writes both CSVs.

## Costs and risks

- **Re-solve required, bigger than v3's.** Section 4 changes the fingerprint so all 12 runs
  re-solve; section 3 adds `per_market_best` (14 runs); section 5a adds two ladder models
  scoped to `day_ahead_only` (16 runs). With MILP integrality on top, budget **4-6 hours**.
- **Headline revenue will fall again** once reserve bids are quantised.
- **The blend may lose.** v3 showed the lowest-error model did not earn the most, so
  `per_market_best` may underperform plain `seasonal_naive`. Report whichever happens.
- Nothing is committed in either repo. v3 and v4 should land as separate commits.

## Open questions carried forward from v3

- **`reserve_total_limit`:** nominal 5.0 MW (current default) or operational 4.5 MW? With
  nominal, 80% of blocks exceed the derated rating. One config key, but a re-solve.
- **Reserve backing energy (v3 item 1e):** C2 holds only `afrr_hours = 1.0` and
  `fcr_hours = 0.5` h/MW (both UNVERIFIED) against 4-hour blocks; delivering X MW for a full
  block needs ~`4/η ≈ 4.2X` MWh. The model can still sell reserve it could not deliver for a
  whole block. Unchanged in v4 — needs the real prequalification rule.
- **aFRR activation** is still the flat placeholder (`alpha` 0.05, EUR 100/MWh). ENTSO-E has
  activated-energy *prices* but not volumes; regelleistung.net has only bid stacks.
- **Capacity bids ride the weakest forecasts.** Capacity plus activation is ~76% of gross
  revenue, and every model loses to `seasonal_naive` on FCR and aFRR−. Improving FCR/aFRR
  forecasting remains the highest-value follow-up, larger than any dashboard change.
