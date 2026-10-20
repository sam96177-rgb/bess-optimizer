# Battery storage optimiser for German power and balancing markets

This project decides, one day ahead, how a 10 MWh battery energy storage system (BESS) should
use its power across four German markets at once: **day-ahead** energy trading and the three
balancing reserves **FCR**, **aFRR+** and **aFRR-**. It plans each day on price *forecasts*,
then pays the plan at the *real* prices, and compares that with a perfect-foresight plan that
knew the real prices in advance. The difference shows how much revenue each forecast model
actually delivers.

The forecasts come from the companion repository
[bess-market-forecast](https://github.com/sam96177-rgb/bess-market-forecast). Together:

```
bess-market-forecast                         bess-optimizer (this repo)
  price forecasts for 4 markets   ------->     plan the battery each day on the forecasts,
  (made the day before delivery)               settle at real prices, compare to perfect foresight
```

## Results

Battery: 10 MWh, 4.5 MW, 90% round-trip efficiency, state of charge kept between 10% and 90%,
EUR 8/MWh wear cost. 162 days from 21 March to 31 August 2026.

**Capture** = revenue earned when planning on a forecast, as a share of the perfect-foresight
revenue. Example: perfect foresight earns EUR 1,000 on a day, the forecast plan earns EUR 900,
capture is 90%.

| Forecast model | Day-ahead trading only | All four markets together |
|---|---|---|
| Perfect foresight (upper limit) | EUR 193,341 (100%) | EUR 615,748 (100%) |
| `seasonal_naive` (same time last week) | 81.4% | 94.7% |
| `linear_lags` (ridge regression) | **90.4%** | 96.8% |
| `lgbm_point` (gradient-boosted trees) | 88.4% | 96.9% |
| `lear` (LASSO per period of day) | 76.7% | 96.7% |
| `per_market_best` (best model per market, picked on a validation window) | 88.4% | **97.1%** |

- **Trading all four markets together earns about three times more** than day-ahead trading
  alone, because most of the money is in reserve capacity, which the battery can sell without
  cycling much.
- **The most accurate forecast is not the most profitable.** `lear` has the second-lowest
  day-ahead forecast error but the lowest day-ahead capture. The reason is timing: only 68% of
  its planned discharges fall within 30 minutes of a perfect-foresight discharge, against 91%
  for `linear_lags`. A battery is paid for selling at the right moment, which an average error
  score does not measure.
- **Across all four markets the trained models are within 0.4 points of each other**; that gap
  is too small to rank them without a significance test.

**Read the "all four markets" figures as provisional.** Energy actually delivered when aFRR is
called is still an assumption (5% of committed capacity, paid EUR 100/MWh), and it makes up
about 20% of the perfect-foresight revenue. See [Status](#status-and-next-steps).

## Quickstart

Both repositories must sit side by side in one folder; this one reads the other's forecasts
(path set in `config/paths.yaml`) and never writes to it.

```
your-folder/
  bess-market-forecast/     set up and run first: see its README
  bess-optimizer/
```

```bash
python -m venv .venv
.venv\Scripts\activate                     # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Run the whole study in order (steps 01-02 run inside the forecast repository with its own
environment):

```bash
python pipeline/01_ingest.py               # download market and weather data
python pipeline/02_backtest_forecasts.py   # forecast all markets with all models
python pipeline/03_select_blend.py         # pick the best model per market (validation window)
python pipeline/04_solve_batch.py          # plan and settle every day, every model (~25 min)
python pipeline/06_export.py               # one table of all runs
streamlit run app/streamlit_app.py         # dashboard
pytest tests/ -q                           # 276 tests (~10 min; several solve real models)
```

Each step can be re-run safely: downloads are cached and finished solves are skipped unless
the forecasts or settings changed. `--dry-run` shows what a step would do.

## How it works

```
 forecast repo results            one forecast price per market and delivery period,
          |                       plus the real price
          v
 src/data/forecast_loader.py      for each day: decision prices (forecast) and
          |                       settlement prices (real), never mixed
          v
 src/model/build.py               the day's optimisation model (MILP, linopy + HiGHS)
          |
          v
 src/model/rolling.py             solve day after day over the test window
          |
          v
 src/valuation/revenue.py         pay the plan at real prices; compare to perfect foresight
          |
          v
 results/<run>/                   dispatch per quarter-hour, revenue per day, run settings
          |
          +--> app/               dashboard: study overview, forecast comparison, dispatch explorer
```

**One day, step by step.** The day before delivery, the optimiser receives forecasts for 96
day-ahead quarter-hours and six 4-hour blocks of each reserve. It chooses, for every
quarter-hour, how much to charge or discharge, and for every block, how many MW of FCR, aFRR+
and aFRR- to offer, to maximise expected profit. The same plan is then paid at the real prices.
Running the same days with the real prices as the "forecast" gives the perfect-foresight upper
limit.

**What the model respects** (all in `src/model/build.py`):

- energy balance with charging and discharging losses, and a battery that ends each day at the
  charge it started with;
- state of charge between 10% and 90%, minus the energy held back for any reserve sold
  (1 hour per MW for aFRR, 0.5 hour per MW for FCR);
- power shared between trading and reserves in each direction, with each MW of FCR counting as
  1.25 MW of battery power (German TSO prequalification rule);
- reserves offered in whole megawatts (1 MW minimum and step, regelleistung.net);
- no charging and discharging in the same quarter-hour.

**Strategies.** `day_ahead_only` trades energy only. `full_cooptimized` decides all four
markets together in one model.

**Settings.** Battery data are in `config/battery.yaml`. Every value that is assumed rather than
measured is in `config/assumptions.yaml` with its source, and is shown on the dashboard.
`config/study.yaml` in the forecast repository defines the study period and models for both
repositories.

## Status and next steps

Working and tested. Open points:

- **Real aFRR activation energy.** Replace the assumed 5% / EUR 100 per MWh with real data.
  ENTSO-E publishes the activation prices per quarter-hour and direction; the activated volumes
  must come from another source (Netztransparenz).
- **Forecasts built for timing**: models that place daily highs and lows better, since that is
  what decides day-ahead revenue.
- **Longer test period**, and a significance test between the forecast models.

## Licence

View-only: you may read the code and run it unchanged to evaluate it, but not modify, reuse or
redistribute it. See [LICENSE](LICENSE).
