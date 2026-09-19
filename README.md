# bess-optimizer

A deterministic MILP optimizer that decides how to operate a 10 MWh / 5 MW
(4.5 MW after derating) battery energy storage system (BESS) across four German markets —
day-ahead energy, FCR, aFRR+, aFRR- — **jointly**. It answers two
questions: given perfect foresight of what prices actually did, how much
revenue could this battery have earned and where does it come from
(v1) — and given only a forecast model's prediction at decide-time, how
much of that revenue is actually achievable once settled against what
prices really did (Milestone 8 / §6a)?

v1 deliberately used only realized prices (`y_true`), to isolate and
validate the optimizer's own logic — the co-optimization constraints, the
unit handling, the MILP formulation — before forecast error entered the
picture. That's done, so `SPEC.md` §6a lifted forecast-driven runs and a
Streamlit dashboard into scope: **decide on forecast (`y_pred`), settle on
realized (`y_true`)** — see [Forecast-driven runs](#forecast-driven-runs-spec-md-6a)
and [Dashboard](#dashboard) below.

Full build spec and milestone-by-milestone history: `SPEC.md`.

---

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
pytest                          # 101 tests, a few minutes (several solve real MILPs)
```

Run the benchmark comparison over a date range:

```bash
python -m src.cli --start-date 2026-03-21 --end-date 2026-08-31
```

This prints the assumptions banner, then total revenue / revenue split /
cycles-per-day / SOC utilisation for three strategies: day-ahead-only
arbitrage, a naive rule-based baseline, and the full co-optimized run.
`--start-date`/`--end-date` are required — there's no silent default
window, since which days get included materially changes the numbers you
see (see [Data availability](#data-availability-a-real-constraint-not-a-bug)).

This repo is **read-only with respect to `bess-market-forecast`** — it
only reads that repo's `config/markets.yaml` and `results/*.parquet`,
never writes to or modifies it.

---

## Repository structure

```
config/
  battery.yaml        # physical battery parameters (locked spec values, not assumptions)
  assumptions.yaml     # every genuinely-assumed value, commented with source + replacement
  solver.yaml          # solver choice/tolerances + model-behavior flags (cyclic_soc, etc.)
  market_prices.yaml   # per-market capacity-price unit basis (see "A real unit mismatch" below)
  paths.yaml            # location of the sibling bess-market-forecast repo
src/
  config.py            # all config loaders + dataclasses; fails loudly on missing/invalid config
  data/
    grid.py             # 15-min day-ahead grid + 4h capacity blocks, incl. DST handling
    forecast_loader.py  # reads bess-market-forecast's results/*.parquet: y_true (realized) and,
                         #   per §6a, y_pred (a chosen model+quantile) via DayPricePair
  model/
    battery.py          # Battery dataclass (power/energy/efficiency/SOC bounds)
    build.py            # linopy model construction (single function serves both the
                         #   day-ahead-only and full multi-market models)
    solve.py             # solves + extracts a solution, fails loudly on non-optimal
    rolling.py           # day-by-day orchestration across a date range; writes the
                          #   run_meta.json sidecar (write_rolling_run_results)
    _day_solve_worker.py # standalone worker: solves one day, invoked via subprocess
                          #   (see "Performance" below)
  valuation/
    revenue.py           # recomputes revenue independently of the solver's own objective
    benchmarks.py         # the three comparison strategies
  results_catalogue.py   # scan_runs(): one row per results/<solve_id>/run_meta.json,
                          #   the discovery path shared by the CLI, batch.py, and the dashboard
  batch.py                # python -m src.batch: solves the model x strategy results matrix,
                           #   resuming already-solved combinations
  cli.py                  # prints the summary + assumptions banner
app/                      # Streamlit dashboard (streamlit run app/streamlit_app.py) --
                           #   see "Dashboard" below
tests/                   # 101 tests; several run against the real sibling repo (read-only)
results/                 # run outputs land here: results/<solve_id>/{long_format,daily_revenue}
                          #   .parquet, skip_log.csv, run_meta.json (write_rolling_run_results)
```

---

## Model formulation

Index `t` over 15-minute steps (Δt = 0.25 h), `b` over 4-hour capacity
blocks (16 steps/block nominally — fewer on part of a DST day, see
`grid.py`).

**Variables**: `p_ch[t]`, `p_dis[t]` (MW, ≥0), `u[t]` (binary, 1=charging),
`e[t]` (MWh, SOC at end of step t), `c_fcr[b]`, `c_pos[b]`, `c_neg[b]`
(MW, ≥0, one value per block — capacity commitments).

**Constraints**:
- **C1 SOC dynamics**: `e[t] = e[t-1] + η·p_ch[t]·Δt − p_dis[t]·Δt/η − activation[t]`,
  where `η = √(round_trip_efficiency)` is applied on each side, and
  `activation[t] = (α_pos·c_pos[b] − α_neg·c_neg[b])·Δt` is the assumed
  aFRR activation energy (config flag `include_activation_in_soc`).
- **C2 SOC headroom**: committed capacity reserves energy headroom —
  `e[t] ≥ soc_min + c_fcr·T_fcr + c_pos·T_afrr`,
  `e[t] ≤ soc_max − (c_fcr·T_fcr + c_neg·T_afrr)`.
- **C3 shared power limits**: `p_ch + c_fcr + c_neg ≤ P_max`,
  `p_dis + c_fcr + c_pos ≤ P_max` (FCR appears on both sides — it's
  symmetric, so it reserves headroom in both directions at once).
- **C4**: `c_fcr[b] ≤ 0.8·P_max`.
- **C5 exclusivity** (`enforce_exclusivity`, default on): `p_ch ≤ P_max·u`,
  `p_dis ≤ P_max·(1−u)`.
- **C6 terminal SOC** (`cyclic_soc`, default on): see the note below.
- **C7 block alignment**: structural, not a separate constraint — `c_fcr`/
  `c_pos`/`c_neg` are one variable per block, so they're constant across
  all of that block's steps by construction.

**Objective** (maximize): day-ahead arbitrage revenue, plus each capacity
market's price × committed capacity (scaled per `market_prices.yaml`'s
basis — see below), plus the assumed activation energy revenue.

### A note on C6 and the exact meaning of "cyclic"

`SPEC.md` defines `e[t]` as "SOC at end of step t" and C6 as `e[last
step] == e[first step]`. Read fully literally, that ties `e[0]` — which
already reflects step 0's own dispatch — to `e[T-1]`, forcing step 0's
net energy change to exactly zero. That can't be the intent (it
contradicts C6's own stated purpose: preventing the optimizer from
draining the battery and never refilling it). The implementation instead
reads it as: **the SOC *before* step 0 happens equals `e[T-1]`** — i.e.
step 0's dynamics use `e[T-1]` as the "previous" state when
`cyclic_soc=True`, making the day a closed loop. `cyclic_soc=False`
instead takes an externally supplied `initial_soc_mwh` (used by
`rolling.py` to carry a real ending SOC across a day boundary). See the
docstring at the top of `src/model/build.py` for the full reasoning. If
this isn't what was intended, it's a one-line change in `build_model`.

---

## Configuration

- **`config/battery.yaml`** — the v2 test system: 10 MWh, 5 MW nominal x 0.9
  derating = 4.5 MW operational, 90% round-trip efficiency, 10–90% SOC band
  (1–9 MWh), 50 kW aux load drawn from the battery every step, and EUR
  8/MWh degradation cost on throughput (charge + discharge). Given values,
  not assumptions. `terminal_value: mean_decision_da_price` adds a
  decision-side credit for energy left at the end of a non-cyclic day
  (lambda = the day's mean decision DA price; skipped under cyclic SOC).
  The credit is never reported as revenue: `net = gross - degradation`,
  and `net + terminal_credit == solver objective` when settlement equals
  decision prices. `daily_revenue.parquet`'s `total` column is net revenue.
- **`config/solver.yaml`** — solver choice/tolerances, and the three
  model-behavior toggles: `cyclic_soc`, `enforce_exclusivity`,
  `include_activation_in_soc`.
- **`config/market_prices.yaml`** — **a verified data fact, not an
  assumption.** `bess-market-forecast/config/markets.yaml` labels FCR and
  aFRR capacity prices identically as `EUR/MW`, but tracing its own
  ingestion code (`src/ingest/regelleistung.py`) shows FCR's source
  column is a **total price for the whole 4h block**, while aFRR's is an
  **hourly rate**. Getting this wrong scales ancillary revenue by 4×.
  This file records `price_basis` per market (`per_block_total` /
  `per_mw_hour` / `per_mwh_energy`) so `build.py`'s objective scales each
  market correctly instead of assuming one convention for both.
- **`config/paths.yaml`** — the sibling repo's location (relative path),
  so nothing in `.py` hardcodes it.

---

## Assumptions — and how to replace each with real data

Every value below lives in `config/assumptions.yaml`, is printed in a
banner at the top of every CLI run, and is flagged (`is_assumption` /
`activation_is_assumption`) wherever it touches a results file. **None of
these are measurements.**

| Assumption | Current value | Why it's assumed | Real replacement |
|---|---|---|---|
| `alpha_pos`, `alpha_neg` (aFRR activation rate) | 0.05, 0.05 — **unconfirmed placeholder**, SPEC.md gave no default | Real activation is stochastic and unknown ahead of time; the deterministic MILP needs a point estimate | Activation volumes published on Netztransparenz.de, aggregated to an empirical rate per block/season |
| `activation_energy_price_eur_mwh` | 100.0 — **unconfirmed placeholder** | This repo has no activation-price feed; regelleistung.net is capacity-only | aFRR energy activation prices from the ENTSO-E Transparency Platform (not regelleistung.net) |
| `fcr_hours` (FCR reserve duration) | 0.5 | Prequalification energy-headroom rule, given as a SPEC.md default but **not verified against current German regulation** | BNetzA / regelleistung.net prequalification documentation |
| `afrr_hours` (aFRR reserve duration) | 1.0 | Same as above | Same as above |

**This matters a lot in practice.** In the full 164-day backtest
(2026-03-21 to 2026-08-31), the co-optimized strategy's total revenue was
€193,424 — but **30.7% of that (€59,391) is the assumed activation
term**, and the two aFRR capacity markets it enables (€99,392 combined)
dwarf actual day-ahead arbitrage (€34,071). Before using this repo's
output for anything beyond validating the optimizer's own logic, replace
`alpha_pos`/`alpha_neg`/`activation_energy_price_eur_mwh` with real
figures — the headline number is currently mostly an assumption, not a
measurement, and the CLI says so explicitly when it's large.

---

## Data availability — a real constraint, not a bug

`bess-market-forecast`'s `results/*.parquet` has `day_ahead` starting
2026-03-21, while FCR/aFRR start 2026-03-04 — a genuine gap in the
sibling repo's backtest window, unrelated to daylight saving time
(confirmed: the March DST transition, 2026-03-29, is a week *after* the
gap and is handled correctly — see `grid.py`'s DST tests). Per Hard Rule
1, this repo never fabricates or interpolates missing prices: a day
missing any market's price is skipped and logged (`forecast_loader.py`,
`SkipRecord`), never zero-filled. Over the full window, 20 of 184 days
(2026-03-01–20) are skipped this way; everything from 2026-03-21 onward
solved cleanly. If more March history is wanted, the sibling repo's
day-ahead backtest would need to be extended — that's out of scope here
since that repo is read-only from this one.

---

## Results: 164-day backtest (2026-03-21 → 2026-08-31)

```
--- day_ahead_only ---
  days solved: 164  |  days skipped: 0
  total revenue: EUR 56386.41
  cycles/day: 2.36  avg SOC utilisation: 100.0%

--- naive ---
  days solved: 164  |  days skipped: 0
  total revenue: EUR 39537.82
  cycles/day: 0.81  avg SOC utilisation: 100.0%

--- full_cooptimized ---
  days solved: 164  |  days skipped: 0
  total revenue: EUR 193424.47
    split -> day_ahead: 34070.71  fcr: 570.87  afrr_pos: 25671.23  afrr_neg: 73720.96  activation: 59390.70
  cycles/day: 0.65  avg SOC utilisation: 30.8%
  !! 30.7% of total revenue comes from the ASSUMED activation term.
```

Reproduce with `python -m src.cli --start-date 2026-03-21 --end-date 2026-08-31`.

Worth noting: the co-optimized run cycles the battery *less* (0.65/day
vs 2.36 for pure arbitrage) and uses less of its SOC range (31% vs 100%)
— it's trading cycling for holding capacity, which the numbers say pays
off, but that trade-off is exactly where the activation and reserve-
duration assumptions bite hardest.

---

## Performance note

Rolling multi-day runs solve each day in its **own subprocess**
(`src/model/_day_solve_worker.py`), not in-process. This was added after
observing a single long-running process solving 150+ days in sequence
degrade from a measured ~4.5s/day model-build cost to 50–120+s/day as the
run progressed — individual HiGHS solves stayed under 0.1s throughout, so
this was a memory/state accumulation issue in `linopy`/`highspy` across
many sequential `Model()` instantiations in one process, not a modeling
problem. Since `cyclic_soc=True` days are independent, `rolling.py`
dispatches them concurrently across CPU cores; the full 164-day, 3-
strategy backtest runs in about 4.5 minutes. `cyclic_soc=False` runs
still isolate each day in a subprocess but must run sequentially (day
N+1's initial SOC depends on day N's solved result).

---

## Forecast-driven runs (`SPEC.md` §6a)

Beyond perfect foresight, the optimizer can now **decide** on a chosen
forecast model's `y_pred` while always **settling** revenue on `y_true` —
optimizing and valuing both at `y_pred` would make an overconfident
forecast look more profitable than it is, since the schedule would never
be checked against reality:

```
cost of forecast error = perfect-foresight revenue − forecast-driven revenue
                         (both settled at y_true, over the same day set)
```

```bash
python -m src.cli --start-date 2026-06-15 --end-date 2026-06-17 \
    --decision-source forecast --forecast-model lgbm_quantile_tuned --write-results
```

`--forecast-quantile` defaults to 0.5 (the median); point models
(`seasonal_naive`, `linear_lags`, `lgbm_point`) only have that quantile.
`--write-results` persists each strategy's run to
`results/<solve_id>/` (see below).

To solve the full model x strategy matrix and populate the dashboard:

```bash
python -m src.batch --dry-run   # see the matrix without solving anything
python -m src.batch             # run it -- ~1.5-3h for the default 164-day window; run in the background
```

`batch.py` resumes automatically: re-running the same command skips any
`(strategy, decision_source, forecast_model, forecast_quantile,
start_date, end_date)` combination that already has a matching run on
disk (same battery/solver/assumptions config), unless `--force`.

---

## Sequential clearing (`full_sequential`)

The real D-1 order is FCR (~08:00) -> aFRR (~09:00) -> day-ahead (12:00).
`full_cooptimized` decides everything at once; `full_sequential`
(`src/model/sequential.py`) solves three times per day and commits one
market per stage (FCR, then aFRR with FCR fixed, then day-ahead dispatch
with all reserves fixed), each settled at realized prices as usual.

**What the FCR/aFRR stages see for day-ahead prices** is the only thing
that can make it differ from `full_cooptimized`:

- **Default: the run's own model at every stage.** All five day-ahead
  models use only calendar, weather *forecasts*, `lag_168h` and
  `lag_672h` (`lag_24h` is dropped as unavailable), so a forecast made
  at 08:00 would be nearly identical to the 11:00-cutoff one; the only
  possible difference is the weather-forecast vintage. Result: gate
  ordering costs essentially nothing (14 days, 2026-06-15..28: within
  0.02% of `full_cooptimized` for every model).
- **`--sequential-da-proxy seasonal_naive`: the strict, worst-case
  variant.** The early stages see only a 168h-lag forecast. Over the same
  14 days this cost 0.8-2.3% for the LightGBM models (and, being noise
  against a weak DA forecast, gained 1.6% for `linear_lags`).
- **Perfect foresight** (`full_sequential | realized`): every stage sees
  the real prices, which reproduces `full_cooptimized | realized` -- kept
  as its own run so the dashboard has a matching comparator.

Caveats: scored on the solver's own (decision-side) prices, a less
informed sequential plan can never beat the simultaneous one; after
settlement at *realized* prices either can come out ahead, since forecasts
are wrong. Reserve capacity is assumed fully awarded at the realized
clearing price (price-taker); aFRR is really pay-as-bid.

---

## Digital twin (`python -m src.twin`)

Replays a stored run (`results/<solve_id>/`) step by step under the real
battery physics and **random** aFRR activation, and logs where the plan
cannot be delivered. It never re-solves.

```bash
python -m src.twin --solve-id <id> --n-draws 500 --seed 0 --plot-day 2026-06-16
```

- The planner treats activation as its *expected* energy (`alpha * c * dt`
  every step). The twin draws each step: called upward with probability
  `alpha_pos / m`, downward with `alpha_neg / m` (never both), and a called
  step delivers a random fraction (Beta-distributed, mean `m`) of the
  committed capacity. `m` = `--call-degree-mean`, default 0.25: frequent
  small calls (`m = 1` = always the full capacity, the crudest shape). The
  mean equals the plan's expectation for any `m`, so any deviation is
  variance the plan could not see. `--mode expected` reproduces the plan's
  SOC exactly (tested) -- proof both share one physics.
- SOC is held inside the window. Planned dispatch and aux load come first;
  activation gets the SOC room left. Undeliverable energy is logged as a
  shortfall (`dispatch_shortfall_mwh`, `activation_shortfall_up/down_mwh`)
  and SOC is clipped. Days are replayed independently from the plan's own
  starting SOC.
- Outputs next to the run: `twin_summary.parquet` (one row per day x draw),
  `twin_paths.parquet` (step paths for the first draws), `twin_meta.json`,
  and optionally `twin_soc_<day>.png`.

**Read the numbers with this caveat.** `alpha`, the call-size mean and its
spread are all PLACEHOLDERS, and draws are independent per step (no
burstiness, no mean reversion by the TSO), so SOC is a random walk. On the
2026-06-15..18 `full_cooptimized` smoke run, 30% of (day, draw) replays hit
a limit with the default call size (47% when every call is full-capacity,
`--call-degree-mean 1`) -- the result moves a lot with the assumed shape,
so treat it as assumption-driven, not a finding. Calibrating the mean, call
size and burstiness against real activation statistics is the next step.
FCR activation energy is assumed to net to zero.

---

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

Reads only `results/<solve_id>/` sidecars (via `src/results_catalogue.py`)
and the sibling repo's `results/*.parquet` (read-only) — never solves
anything itself, so solve first (`--write-results` or `python -m
src.batch`). Three pages:

- **Dispatch Explorer** — pick a model, quantile, strategy, and date
  window: the day-ahead forecast fan vs. realized price, battery
  dispatch/SOC, ancillary capacity commitments per 4h block, and daily +
  cumulative revenue against the matching perfect-foresight run.
- **Forecast Error Cost** — a models x strategies matrix of the cost of
  forecast error defined above, with day sets intersected before
  differencing (see [Risks](#not-built-yet) below) and the dropped-day
  count shown alongside so a large gap isn't hidden.
- **Run Catalogue** — every solved run and, for a selected one, its skip
  log.

---

## Not built yet

Out of scope by design — these are real follow-ups, not oversights:

- **Stochastic optimization over quantiles.** Forecast-driven runs decide
  on one quantile at a time (`SPEC.md` §6a); optimizing over the full
  quantile distribution is separate future work.
- **Real activation-data ingestion.** `alpha_pos`/`alpha_neg`/activation
  price are placeholders (see above), not pulled from
  Netztransparenz.de/ENTSO-E.
- **Cycle/calendar-aging models.** Degradation is a flat EUR/MWh
  throughput cost, not a physical aging model.
- **Twin activation model calibrated on real data.** The twin exists, but
  its activation draws use the placeholder alpha and a crude call shape.
- **Intraday or real-time re-optimization.** This is a day-ahead batch
  optimizer, not a rolling/receding-horizon real-time controller.

**A risk worth restating:** `seasonal_naive` skips 2 extra days
(2026-03-29, 2026-04-05 — a DST day and its one-week echo) that every
other model solves cleanly, so raw revenue totals across models are
**not** directly comparable — any comparison must intersect day sets
first (the Forecast Error Cost page does this; `run_rolling`/benchmarks
totals do not, by design, since summing is the caller's job).

## Known limitation

DST **fall-back** (25-hour) days: the 6 nominal 4h capacity blocks end an
hour before such a day's local end, so `build_day_grid` assigns the
trailing hour's steps to the last block (`extend_last_block_over_long_day`).
This is an **unverified assumption**: no real fall-back-day data exists
yet (data ends 2026-09-17; the day is 2026-10-25), so how regelleistung
actually cuts that day is not confirmed. Check it against real data once
2026-10-25 is published.
