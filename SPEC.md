# BESS Multi-Market Optimizer — Build Spec

You are building a new, standalone Python repo: a deterministic MILP optimizer
that decides how to operate a battery energy storage system (BESS) across four
German markets simultaneously.

This spec is the source of truth. Work through it **milestone by milestone**.
Stop at the end of every milestone, report what you did, and wait for my
confirmation before starting the next one. Do not run ahead.

---

## 0. Hard rules (these override everything else)

1. **Never fabricate data.** If a file is missing, an API fails, a column isn't
   there, or a value is ambiguous — stop, say exactly what's missing, and ask.
   Do not fill gaps with plausible-looking numbers, do not silently
   interpolate, do not invent a fallback.
2. **Assumptions are not data.** Some inputs in v1 are deliberately assumed
   (listed in §3). Every assumed value lives in `config/assumptions.yaml`, is
   commented with *what it is, why it's assumed, and what real data source
   would replace it*, and is printed in a banner at the top of every CLI run.
   An assumption must never be silently mixed into a results file without
   being flagged there too.
3. **The forecast repo is read-only.** You may read from it. You may not write
   to it, modify it, or `git commit` inside it.
4. **Config-driven.** No market timing constant, battery parameter, or
   assumption hardcoded in `.py` files. They live in YAML.
5. **Fail closed and loudly.** Prefer raising with a clear message over
   degrading quietly.
6. If something in this spec turns out to be wrong or impossible when you meet
   the real data, **say so and stop** rather than working around it.

---

## 1. Context — what already exists

There is a sibling repo, `bess-market-forecast`, in the same parent directory
as this new repo. It is complete and validated (Milestones 0–7 done, 55 tests
passing). It forecasts four markets:

| Market key | What it is | Native product |
|---|---|---|
| `day_ahead` | Day-ahead energy price (€/MWh) | 15-minute products (EU SDAC reform, Oct 2025) |
| `fcr` | Frequency Containment Reserve capacity price | 4-hour blocks |
| `afrr_pos` | aFRR positive capacity price | 4-hour blocks |
| `afrr_neg` | aFRR negative capacity price | 4-hour blocks |

Files in that repo you will need:

- `results/*.parquet` — backtest output, long format, columns:
  `market, model, run_id, cutoff_time, target_time, horizon_step, quantile, y_pred, y_true`
- `config/markets.yaml` — per-market timing, gate closures, product durations,
  DST conventions. **Reuse these values; do not duplicate or re-derive them.**
- `src/evaluation/resample.py` — maps 4-hour FCR/aFRR products onto the
  day-ahead 15-minute grid. Use this as the reference for how the unified grid
  is constructed. Re-implement or import it, but do not invent a second,
  different resampling convention.
- `src/config.py` — `MarketConfig` dataclass, `load_markets_config()`.

Known characteristics of that data you must handle, not be surprised by:
- FCR and aFRR have ~10 unavoidable `NaN` predictions per market around the
  2026-03-29 DST transition and its one-week echo.
- One real day (2026-09-13) was published by ENTSO-E at hourly instead of
  15-minute resolution.
- `day_ahead` and `fcr`/`afrr_*` follow **different** DST conventions, modelled
  explicitly per market in `markets.yaml`.

---

## 2. What this repo does

Given realized prices for all four markets over a period, decide the
profit-maximising operation of one BESS across all four markets **jointly**,
respecting shared power and state-of-charge limits, and report the revenue
achieved against benchmarks.

German multi-market dispatch is **one joint problem, not four independent
ones**. FCR and aFRR capacity gates close earliest (~D-1 08:00–09:00); the
day-ahead gate closes later (D-1 12:00). So day-ahead energy trading must
respect capacity already committed to the ancillary markets. Capacity
committed to FCR/aFRR reserves power headroom *and* SOC headroom that
day-ahead arbitrage then cannot use.

v1 runs on **realized prices** (`y_true`), i.e. perfect foresight. This is
deliberate: it isolates and validates the optimizer's own logic before forecast
error is introduced. Connecting the forecaster's quantile outputs is a later,
separate piece of work — do not build it now.

---

## 3. Locked decisions

| Item | Decision |
|---|---|
| Method | Deterministic MILP |
| Solver stack | `linopy` + `highspy` (HiGHS). Both pip-installable on Windows, no separate solver binary |
| Time grid | 15-minute unified grid, Δt = 0.25 h, consistent with `resample.py` |
| Price input | `y_true` from the forecast repo's `results/*.parquet` |
| Markets | All four, co-optimized |
| Battery | v1: 1 MW / 2 MWh, 88% RTE, SOC 5–95%, no degradation cost. **v2 (PLAN_V2_DISPATCH.md Phase 1): 10 MWh / 5 MW x 0.9 derating, 90% RTE, SOC 10–90%, 50 kW aux load, EUR 8/MWh degradation** — see `config/battery.yaml` |
| aFRR energy activation | Modelled via an **assumed** activation rate (see below) |
| Scope of v1 | Parquet output + CLI summary. **No Streamlit dashboard** — that is a later session |
| Repo | New repo, `git init`, sibling to `bess-market-forecast` |

### Assumed values for v1 (all go in `config/assumptions.yaml`)

These are assumptions, not measurements. Each needs the comment format
described in Hard Rule 2.

- **aFRR activation rate** `alpha_pos`, `alpha_neg` — expected fraction of
  committed aFRR capacity actually activated. Real replacement: activation
  volumes published on Netztransparenz.de.
- **aFRR activation energy price** — €/MWh received for activated energy. Real
  replacement: aFRR energy activation prices from the ENTSO-E Transparency
  Platform (note: *not* regelleistung.net, which carries capacity prices only).
- **Reserve energy duration requirements** — the energy headroom that must be
  held in reserve per MW of committed capacity, to satisfy prequalification.
  Defaults to assume: FCR 0.5 h, aFRR 1.0 h. **Flag these prominently — they
  are prequalification rules I have not verified against current German
  regulation, and they materially change the result.** Ask me before treating
  them as settled.

---

## 4. Optimization model

Index `t` over 15-minute steps (Δt = 0.25 h), index `b` over 4-hour capacity
blocks. Each block contains 16 steps. Capacity decisions are constant within a
block; energy decisions vary per step.

### Variables

| Variable | Domain | Meaning |
|---|---|---|
| `p_ch[t]` | ≥ 0, MW | Day-ahead charging power |
| `p_dis[t]` | ≥ 0, MW | Day-ahead discharging power |
| `u[t]` | binary | 1 = charging mode, 0 = discharging mode |
| `e[t]` | MWh | State of charge at end of step t |
| `c_fcr[b]` | ≥ 0, MW | FCR capacity committed (symmetric) |
| `c_pos[b]` | ≥ 0, MW | aFRR positive capacity committed |
| `c_neg[b]` | ≥ 0, MW | aFRR negative capacity committed |

### Constraints

**C1 — SOC dynamics.** With `η = sqrt(RTE)` applied on each side:

```
e[t] = e[t-1] + η·p_ch[t]·Δt − p_dis[t]·Δt/η − expected_activation_energy[t]
```

where `expected_activation_energy[t] = (alpha_pos·c_pos[b(t)] − alpha_neg·c_neg[b(t)])·Δt`
(positive activation discharges, negative activation charges). Make inclusion of
this term a config flag `include_activation_in_soc`, defaulting to `true`.

**C2 — SOC bounds with reserved headroom.** Committed capacity must be
physically deliverable, so it eats into the usable SOC band:

```
e[t] ≥ SOC_min·E_max + (c_fcr[b]·T_fcr + c_pos[b]·T_afrr)
e[t] ≤ SOC_max·E_max − (c_fcr[b]·T_fcr + c_neg[b]·T_afrr)
```

`T_fcr`, `T_afrr` are the reserve duration assumptions from §3.

**C3 — Shared power limits.** This is the core co-optimization constraint. All
four markets draw on one power rating, per direction:

```
p_ch[t]  + c_fcr[b] + c_neg[b] ≤ P_max
p_dis[t] + c_fcr[b] + c_pos[b] ≤ P_max
```

FCR appears in **both** because it is symmetric — it must be able to deliver in
either direction, so it reserves headroom on both sides simultaneously.

**C4 — FCR cap.** `c_fcr[b] ≤ 0.8 · P_max`.

**C5 — Charge/discharge exclusivity.**

```
p_ch[t]  ≤ P_max · u[t]
p_dis[t] ≤ P_max · (1 − u[t])
```

Add a config flag `enforce_exclusivity` (default `true`). Also run one
comparison with it `false` and report whether the LP relaxation ever puts both
nonzero — with capacity payments in the objective it genuinely can, unlike pure
arbitrage, so this is worth knowing rather than assuming.

**C6 — Terminal SOC.** `e[last step of day] = e[first step of day]`, config flag
`cyclic_soc`, default `true`. Prevents the optimizer from manufacturing revenue
by draining the battery and never refilling it.

**C7 — Block alignment.** Capacity variables constant across all 16 steps of
their block. Derive block boundaries from `markets.yaml`, not from a hardcoded
4-hour assumption.

### Objective — maximize

```
  Σ_t  price_da[t] · (p_dis[t] − p_ch[t]) · Δt              (day-ahead energy)
+ Σ_b  price_fcr[b] · c_fcr[b] · block_hours                 (FCR capacity)
+ Σ_b  price_pos[b] · c_pos[b] · block_hours                 (aFRR+ capacity)
+ Σ_b  price_neg[b] · c_neg[b] · block_hours                 (aFRR− capacity)
+ Σ_b  (alpha_pos·c_pos[b] + alpha_neg·c_neg[b]) · block_hours · p_activation
                                                             (assumed activation energy)
```

**Units warning:** do not assume the capacity price unit. Check whether
`markets.yaml` and the parquet express capacity prices in €/MW/h or €/MW per
block, confirm it against the data, and report what you found in Milestone 0.
Getting this wrong scales ancillary revenue by 4× and will silently look
plausible.

---

## 5. Milestones

Stop and report at the end of each. Do not proceed without my go-ahead.

### Milestone 0 — Inspect, report, build nothing

No code beyond a throwaway inspection script. Report back:

1. The resolved absolute path to `bess-market-forecast`, confirmed to exist.
2. Whether an untracked `market-optimizer/` folder exists in the parent
   directory, and if so, exactly what's in it. **Do not delete or modify it.**
   Just tell me, and I'll decide.
3. From `results/*.parquet`: the real date range, row count, and the distinct
   values of `market`, `model`, `quantile`, `horizon_step`.
4. For each market: native resolution, and the count and location of `NaN`s in
   `y_true`.
5. The **unit** of each market's price in the data, and how you determined it.
6. What `markets.yaml` actually contains for gate closures, product durations,
   block boundaries and DST conventions.
7. Python version available and whether `linopy` + `highspy` install cleanly on
   this Windows machine.

Flag any mismatch between this spec and what you actually find.

### Milestone 1 — Scaffold

`git init`. Structure:

```
bess-optimizer/
  config/
    battery.yaml          # v2: 10 MWh / 4.5 MW operational, 90% RTE, SOC 10–90%, aux, degradation
    assumptions.yaml      # §3 assumptions, each commented
    solver.yaml           # solver choice, flags, tolerances
  src/
    config.py             # loaders + dataclasses
    data/
      forecast_loader.py  # reads sibling repo's parquet, y_true only for now
      grid.py             # 15-min unified grid, block mapping
    model/
      battery.py          # physical params, validation
      build.py            # linopy model construction
      solve.py            # solve, extract, diagnose infeasibility
    valuation/
      revenue.py          # recompute revenue independently of the solver
      benchmarks.py
    cli.py
  tests/
  results/
  README.md
```

`pyproject.toml` or `requirements.txt`: `linopy`, `highspy`, `pandas`,
`pyarrow`, `pyyaml`, `pytest`. Nothing else without asking.

Config loaders + their tests only. No optimization yet.

### Milestone 2 — Data layer

`forecast_loader.py` reads the sibling repo's `results/*.parquet` and returns a
tidy realized-price frame for a requested date range. `grid.py` builds the
15-minute unified grid and the step→block mapping, following `resample.py`'s
convention exactly.

Requirements:
- Handle the two known DST conventions per market correctly.
- Handle the known `NaN` regions explicitly: a day with missing prices in any
  market is **skipped with a logged reason**, never zero-filled.
- Handle the 2026-09-13 hourly-instead-of-15-minute day explicitly.
- Tests covering each of the above, including the DST day.

### Milestone 3 — Single-day day-ahead-only MILP

Simplest working optimizer: one day, `day_ahead` only, no ancillary markets, no
activation. Constraints C1, C5, C6 only.

This exists to prove the core is right before complexity lands. Tests:
- SOC never leaves bounds.
- Power never exceeds `P_max`.
- Never charges and discharges in the same step.
- Energy in/out reconciles with RTE to within tolerance.
- Revenue recomputed by `revenue.py` matches the solver objective.
- Known-answer test: on a hand-constructed two-price day, the optimal schedule
  is analytically obvious — assert the solver finds it.

### Milestone 4 — Full multi-market co-optimization

Add C2, C3, C4, C7 and the capacity + activation objective terms. Tests:
- FCR never exceeds 80% of `P_max`.
- Committed capacity + energy dispatch never breaches `P_max` in either
  direction.
- SOC headroom for committed capacity is genuinely reserved (construct a case
  where ignoring C2 would allow an infeasible commitment, assert it's rejected).
- Capacity is constant within each block.
- Raising a capacity price increases that market's committed capacity
  (monotonicity sanity check).
- With `enforce_exclusivity=false`, report whether simultaneous charge and
  discharge ever appears.

### Milestone 5 — Rolling multi-day run

Solve day by day across the full available date range, carrying SOC across day
boundaries if `cyclic_soc=false`, else resetting. Write long-format results:

```
solve_id, solve_date, target_time, block_id, market, variable, value, unit, is_assumption
```

Plus a per-day revenue breakdown by market. Skipped days must appear in a
skip log with reasons, not vanish silently.

### Milestone 6 — Benchmarks and CLI summary

Implement in `benchmarks.py`:
1. **Day-ahead-only perfect foresight** — arbitrage with no ancillary markets.
2. **Naive rule-based** — charge during the cheapest block, discharge during
   the most expensive, no ancillary.
3. **Full co-optimized perfect foresight** — the main run.

CLI prints: total revenue per strategy, revenue split by market, cycles per
day, average SOC utilisation, and the assumptions banner. Report how much of
total revenue comes from the assumed-activation term specifically — if that
fraction is large, the headline number is mostly an assumption, and it needs to
say so.

### Milestone 7 — README and handoff

README covering: what this is, how to run it, every assumption and how to
replace it with real data, the model formulation, and a clearly-scoped "not
built yet" section (forecast-driven runs, dashboard, real activation data,
degradation costs, intraday re-optimization).

---

## 6. Out of scope — do not build

- ~~Any Streamlit or dashboard code.~~ — now in scope, see §6a.
- ~~Feeding the forecaster's `y_pred` / quantile columns into the optimizer.~~ — now in scope, see §6a.
- Stochastic optimization over quantiles.
- Real activation-data ingestion.
- Degradation modelling.
- Intraday or real-time re-optimization.

These are planned follow-ups. Building them now makes v1 unreviewable.

---

## 6a. Scope extension (Milestone 8) — forecast-driven runs + dashboard

v1 (§2, §6) deliberately ran on `y_true` only — pure perfect foresight — to
validate the optimizer's own logic before forecast error entered the
picture. That validation is done (73/73 tests, real CLI solves). This
milestone lifts the two struck-through §6 bullets above into scope:
feeding `y_pred` into the optimizer, and building a dashboard.

**Core design rule: decide on forecast, settle on realized.** The MILP is
built from `y_pred` (a chosen model + quantile); revenue is always valued
at `y_true`. Optimizing *and* valuing at `y_pred` would make an
overconfident forecast look more profitable than it is, because the
schedule would never be checked against reality.

```
cost of forecast error = perfect-foresight revenue − forecast-driven revenue
                         (both settled at y_true, over the same day set)
```

This is implemented via `src/data/forecast_loader.py`'s `DayPricePair`
(`decision` — y_pred or y_true — vs. `settlement` — always y_true), threaded
through `src/model/rolling.py::run_rolling` and
`src/valuation/benchmarks.py` as `decision_source` / `forecast_model` /
`forecast_quantile` parameters. All other Hard Rules (§0) still apply
unchanged, including Hard Rule 3 (forecast repo read-only).

---

## 7. Start here

Begin with **Milestone 0**. Inspect, report, and stop.
