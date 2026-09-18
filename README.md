# bess-optimizer

A deterministic MILP optimizer that decides how to operate a 1 MW / 2 MWh
battery energy storage system (BESS) across four German markets —
day-ahead energy, FCR, aFRR+, aFRR- — **jointly**, on realized prices
(`y_true`) from the sibling `bess-market-forecast` repo. It answers one
question: given perfect foresight of what prices actually did, how much
revenue could this battery have earned, and where does that revenue come
from?

v1 deliberately uses realized prices, not forecasts. That isolates and
validates the optimizer's own logic — the co-optimization constraints,
the unit handling, the MILP formulation — before forecast error is
introduced as a separate variable. See [Not built yet](#not-built-yet).

Full build spec and milestone-by-milestone history: `SPEC.md`.

---

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
pytest                          # 73 tests, a few minutes (several solve real MILPs)
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
    forecast_loader.py  # reads bess-market-forecast's results/*.parquet, y_true only
  model/
    battery.py          # Battery dataclass (power/energy/efficiency/SOC bounds)
    build.py            # linopy model construction (single function serves both the
                         #   day-ahead-only and full multi-market models)
    solve.py             # solves + extracts a solution, fails loudly on non-optimal
    rolling.py           # day-by-day orchestration across a date range
    _day_solve_worker.py # standalone worker: solves one day, invoked via subprocess
                          #   (see "Performance" below)
  valuation/
    revenue.py           # recomputes revenue independently of the solver's own objective
    benchmarks.py         # the three comparison strategies
  cli.py                 # prints the summary + assumptions banner
tests/                   # 73 tests; several run against the real sibling repo (read-only)
results/                 # rolling-run outputs land here (write_rolling_run_results)
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

- **`config/battery.yaml`** — the locked v1 test system (1 MW / 2 MWh,
  88% round-trip efficiency, 5–95% SOC band). These are given values, not
  assumptions.
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
| `alpha_pos`, `alpha_neg` (aFRR activation rate) | 0.10, 0.10 — **unconfirmed placeholder**, SPEC.md gave no default | Real activation is stochastic and unknown ahead of time; the deterministic MILP needs a point estimate | Activation volumes published on Netztransparenz.de, aggregated to an empirical rate per block/season |
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

## Not built yet

Out of scope for v1 by design (`SPEC.md` §6) — these are real follow-ups,
not oversights:

- **Forecast-driven runs.** Only `y_true` (realized prices) is used.
  Feeding the forecaster's `y_pred`/quantile columns in — and the
  stochastic-optimization question that raises — is separate future work.
- **Dashboard.** No Streamlit or other UI; CLI + parquet output only.
- **Real activation-data ingestion.** `alpha_pos`/`alpha_neg`/activation
  price are placeholders (see above), not pulled from
  Netztransparenz.de/ENTSO-E.
- **Degradation modelling.** No cycle-aging or calendar-aging cost in the
  objective.
- **Intraday or real-time re-optimization.** This is a day-ahead,
  perfect-foresight batch optimizer, not a rolling/receding-horizon
  real-time controller.

## Known limitation

`grid.py`'s block-boundary logic doesn't handle a DST **fall-back**
(25-hour) day — it would raise a clear `ValueError` rather than silently
mis-assigning steps to blocks (`map_steps_to_blocks`). This never occurs
in the current data window (which ends before Germany's autumn 2026
transition), so it was left unhandled and loud rather than built and
untested.
