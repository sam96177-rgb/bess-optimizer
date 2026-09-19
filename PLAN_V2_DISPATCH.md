# Plan v2: battery physics, sequential clearing, digital twin

> **Handoff for a fresh Claude Code session (PowerShell).** Self-contained.
> Run everything with this repo's venv: `.venv\Scripts\python.exe`.
>
> **Status (2026-09-19):** approved, nothing executed yet. Phase 0 has NOT
> been run — tests not re-run, nothing committed.
>
> **Scope for this session: Phase 0 + Phase 1 only.** Stop after Phase 1 and
> report back for review. Phases 2–3 need their own design pass first;
> Phase 4 (intraday) is deferred until a real IDA price source exists.
>
> Previous phase (forecast-driven optimizer, batch, dashboard) is documented
> and DONE in `IMPLEMENTATION_PLAN.md` — read it for background, don't redo it.

## Context

The user pasted a full build spec for a new `bess-dispatch-mpc` repo: a two-stage
(Day-Ahead master + hourly Intraday MPC) dispatch engine for a 10 MWh / 5 MW BESS
across DAM, IDA, FCR, aFRR, with a digital twin, 3 forecasting models, a Plotly
dashboard, and a 30-day Oracle-vs-models benchmark. They asked for feedback and
the next step before building.

**Current state (scanned 2026-09-19):**
- `bess-market-forecast`: 55 tests, committed + pushed to GitHub.
- `bess-optimizer`: the previous plan (forecast-driven optimizer, run catalogue,
  batch, 3-page Streamlit dashboard) is **fully implemented and verified** by the
  PowerShell session — but **nothing is committed** (only the v1 commit `4de77ce`
  exists; 11 modified + 7 new files). `results/` holds a 9-run, 4-day smoke batch
  (2026-06-15→18), not the full 164-day batch.
- No `bess-dispatch-mpc` folder exists yet.

## Feedback on the spec

### Keep — genuinely valuable additions
- Richer battery physics: 4.5 MW derating, 50 kW aux load, €8/MWh degradation,
  10–90% window, terminal-energy value. All are config + a few constraint terms
  on top of the existing model.
- Intraday as a second decision layer, re-solving on updated SoC.
- A digital twin that replays the schedule with activations and logs violations —
  the existing optimizer only has the MILP's own view of SoC.
- Oracle-capture % as a headline metric (same idea as our "cost of forecast error").

### Must fix — correctness bugs in the spec's math
1. **The exclusivity constraint makes FCR impossible.** The spec writes
   `P_dis + R_fcr + R_up ≤ P_lim·u_dis` and `P_ch + R_fcr + R_down ≤ P_lim·u_ch`
   with `u_dis + u_ch ≤ 1`. One binary is always 0, so its constraint forces
   `R_fcr = 0` (and `R_down = 0` whenever discharging, `R_up = 0` whenever charging)
   in *every* interval. The model would never sell FCR. The existing optimizer
   already has the correct form (`src/model/build.py:106-127`): binaries gate only
   the energy flows (`p_ch ≤ P·u`, `p_dis ≤ P·(1−u)`), and reserve headroom is
   un-gated (`p_dis + c_fcr + c_pos ≤ P`). Same bug is in the spec's Stage 2 (`v`).
2. **aFRR has no energy backing.** Only FCR gets a backing constraint; an aFRR-up
   commitment could be sold with an empty battery. Existing `C2` (`build.py:130-138`)
   backs both FCR and aFRR over the whole block, not just one 15-min step.
3. **"Soft" terminal constraint is written as hard** (`E(T) ≥ 4.0`) *and* rewarded
   by a credit — pick one. Recommend: credit only (or a slack with penalty), since
   a hard floor can make DST/extreme days infeasible.
4. **Fixed `T=192`, `b(t)=⌊(t−1)/16⌋+1` breaks on DST days** (92/100 steps). This
   exact bug cost us a day of debugging already; the existing `grid.py` handles it.
5. **FCR price units**: the spec treats FCR as €/MW/h × Δt. We verified the real
   regelleistung FCR price is per-block total while aFRR is per MW-hour
   (`config/market_prices.yaml`) — a 4× revenue error if mixed.

### Must fix — market sequencing is not realistic
The spec co-optimizes DA and reserves in one Stage 1 solve at D-1 12:00. In
reality the order on D-1 is **FCR ~08:00 → aFRR ~09:00 → Day-Ahead 12:00 →
IDA1 15:00** (already encoded in `bess-market-forecast/config/markets.yaml`).
Reserves are committed *before* the DA price is known, using a DA forecast.
A single simultaneous solve uses information no operator has — it would
overstate revenue. A realistic hierarchy is: reserve stage → DA stage (reserves
fixed) → IDA stage (DA fixed, realized DA known).

Also: IDA is three discrete auctions (IDA1 D-1 15:00, IDA2 D-1 22:00, IDA3 D 10:00),
not an hourly market. "Re-optimize hourly" fits continuous intraday (XBID), which
is a different product with different prices.

### Must fix — forecasting spec leaks
Lags `t−1, t−2, t−4, t−8` and the 4h rolling mean are **not available** when
forecasting day D's DA prices at D-1 11:00 — the latest known DA price is from
day D-1. The spec's leakage test ("only data up to t−1") checks the wrong
boundary; the correct one is the market's gate-closure cutoff. This is exactly
the leakage `bess-market-forecast`'s availability layer
(`src/features/availability.py`) already enforces fail-closed. The spec's 3 models
(LightGBM, Ridge, 168h persistence) already exist there as `lgbm_point`,
`linear_lags`, `seasonal_naive` — plus 2 quantile models.

### Data blockers
- **No real IDA price source.** EPEX IDA prices are commercial data; SMARD and
  ENTSO-E transparency don't give us the IDA auction clearing price as far as I
  know. The spec's fallback — "DA + mean-reverting noise" — means Stage 2 would
  earn money trading a spread we invented. Violates the project's "never
  fabricate data" rule for any headline number.
- **Synthetic data generator**: fine for unit tests and offline dev, must never
  feed a reported benchmark. Keep it under `tests/fixtures` scope.
- **SMARD client is redundant**: we already ingest 15-min DA prices from ENTSO-E.
  (Also verify filter `4110` — I'm not certain it is the DE-LU DA price filter.)
- **aFRR activation ξ(t) for the twin**: needs real activation data or an explicitly
  flagged assumption (today α is already a flagged PLACEHOLDER).

### Things to know before trusting the benchmark table
- The table's numbers (€142,350 etc.) are illustrative — results must come from runs.
- Our smoke run: forecast-driven €3479 vs perfect-foresight €3492 — **0.4% cost of
  forecast error**, because ~60% of revenue is capacity payments and ~37% is the
  *assumed* activation term. Oracle-capture % will look high for every model
  unless we report it per market (DA arbitrage capture separately).
- Our real forecast results: LightGBM beats naive on DA, but *loses* to naive on
  FCR and aFRR-neg. The spec's expected ordering (LGBM > Ridge > Persistence)
  won't hold across all markets.

### Architecture
Recommend **evolving `bess-optimizer`** (and consuming `bess-market-forecast`
forecasts) rather than a new repo. A new repo would re-implement verified DST
handling, availability enforcement, real regelleistung parsing, price-unit
handling, the run catalogue, and the dashboard — and lose 128 tests. Pyomo vs
linopy: no benefit to switching; keep linopy + HiGHS. Plotly vs matplotlib:
Streamlit renders Plotly natively, so switching the chart layer for zoom/hover is
cheap and optional.

## Decisions (user, 2026-09-19)
1. **Evolve `bess-optimizer`** — no new repo. Keep linopy + HiGHS, Streamlit.
2. **Defer intraday** until a real IDA price source exists; build everything else on real data.
3. **Realistic sequential clearing** (reserves → DA → later IDA); the simultaneous
   co-optimization stays only as an upper-bound benchmark.

## Phases

### Phase 0 — secure current work (do first)
- `.venv\Scripts\python.exe -m pytest tests/ -q` in `bess-optimizer` (expect 73 + the new
  price-pair / run-meta / batch tests green).
- Commit Parts A–C in two commits (forecast-driven path + persistence/batch; dashboard),
  after showing the user the file list. Offer a private GitHub push like the forecast repo.
- `results/` is gitignored; the 9 smoke runs stay local.

### Phase 1 — battery physics v2 (the immediate build step; no new data needed)

**Config** — `config/battery.yaml` gains:
`power_nominal_mw: 5.0`, `power_derating_factor: 0.9`, `energy_mwh: 10.0`,
`round_trip_efficiency: 0.90`, `soc_min_fraction: 0.10`, `soc_max_fraction: 0.90`,
`aux_load_mw: 0.05`, `degradation_cost_eur_mwh: 8.0`, `terminal_value: mean_decision_da_price`.
`config/assumptions.yaml`: α → 0.05 per spec, still flagged PLACEHOLDER.
`src/config.py::BatteryConfig` + `load_battery_config` validate the new keys (fail loud
if missing, like today).

**`src/model/battery.py`** — `Battery.power_mw` becomes the operational limit
`power_nominal_mw × power_derating_factor` (4.5 MW), so every existing constraint that
reads `battery.power_mw` (exclusivity, C3 headroom, C4 FCR cap, naive heuristic)
picks up derating with no further edits. Add `aux_load_mw`, `degradation_cost_eur_mwh`.

**`src/model/build.py`**
- C1 SoC dynamics: subtract `aux_load_mw × Δt` each step.
- Objective: subtract `C_deg × (p_ch + p_dis) × Δt`.
- Terminal credit `λ × e[T]` with `λ` = mean *decision* DA price of the day —
  only when `cyclic_soc=False` (under cyclic SoC the day already ends where it
  starts, so a credit is meaningless). Credit only, no hard floor.
- Keep the correct exclusivity (`build.py:106-109`), un-gated headroom (`:121-128`)
  and C2 backing (`:130-139`) forms — do **not** port the spec's binary-gated version.
- Optional fidelity fix: activation energy in C1 currently has no η applied
  (`build.py:148-151`); apply `η_ch` to aFRR-down and `1/η_dis` to aFRR-up per the spec.

**`src/valuation/revenue.py`** — `RevenueBreakdown` gains `degradation` (a cost) and
reports `gross` and `net`. The terminal credit is **not** revenue: the existing
`revenue == objective` tests become `net_revenue + terminal_credit == objective`.

**`src/cli.py` / `run_meta.json`** — print and persist gross, degradation, net, and
equivalent full cycles (`Σ discharged MWh / usable MWh`, already computed in
`compute_operational_stats`). New battery fields flow into the sidecar fingerprint
automatically, so old 1 MW runs won't be mistaken for new ones by `batch.py` resume.

**Tests (hand-verified, per spec §8)**
- `test_energy_conservation.py` — 1-day solve: `e[t]−e[t−1]` equals
  `η·p_ch·Δt − p_dis·Δt/η − aux·Δt ± activation` within 1e-5 MWh at every step.
- `test_exclusivity.py` — every step: `p_ch·p_dis == 0` and binary consistency.
- `test_reserve_backing.py` — per block, SoC leaves room for the committed FCR/aFRR
  energy (C2) at every step; and a known-answer case where FCR *is* sold (guards
  against the spec's zero-FCR bug ever creeping in).
- `test_block_invariance.py` — reserve values identical across all steps of a block,
  including a DST day (92 steps).
- Known-answer: a flat-price day with degradation > spread → zero cycling.
- Existing 73+ tests stay green (update only the ones asserting the old
  `revenue == objective` identity and old battery defaults).

### Phase 2 — realistic sequential clearing (next, after Phase 1 review)
Split into gate-ordered solves per delivery day, each deciding on the forecasts
available at its own gate (cutoffs from `bess-market-forecast/config/markets.yaml`)
and settling at realized prices via the existing `DayPricePair`:
FCR (~08:00, forecast DA + FCR prices) → aFRR (~09:00, FCR fixed) → DA (12:00,
reserves fixed). Report sequential vs simultaneous vs perfect foresight.
Detailed design gets its own plan before building.

### Phase 3 — digital twin + activation model
Replay committed schedules step by step with aux, efficiency, activation draws;
log clipping/violations; plot twin SoC vs planned SoC. Needs a decision on
activation data (real regelleistung/netztransparenz activation series vs flagged
assumption) before building.

### Phase 4 — intraday (deferred)
Blocked on a real IDA1/2/3 price source. Machinery can later reuse Phase 2's staging.

### Phase 5 — dashboard
Extend the existing Streamlit app: gross/degradation/net split, per-market oracle
capture, stage-by-stage commitments (Phase 2), twin vs planned SoC (Phase 3).

## Verification
- Phase 0/1: `.venv\Scripts\python.exe -m pytest tests/ -q` in `bess-optimizer` green;
  `python -m pytest tests/ -q` in `bess-market-forecast` still 55 (untouched).
- Phase 1 smoke: `.venv\Scripts\python.exe -m src.cli --start-date 2026-06-15
  --end-date 2026-06-17` with the new 10 MWh battery — check the banner shows the new
  params, net = gross − degradation, cycles/day drop vs. the €0-degradation run, and SoC
  never leaves 1.0–9.0 MWh.
- Re-run a small batch (`python -m src.batch` over 06-15→18) and open the dashboard
  (`streamlit run app\streamlit_app.py`) in the browser to confirm new runs render.
- Phase 2 (later): sequential ≤ simultaneous revenue on the same days (less information
  can only cost money), and forecast-driven ≤ perfect foresight.
