# New England Gas-Electric Production Cost Model

**A chronological unit-commitment and economic-dispatch model of ISO New England, co-optimized with the New England natural gas system.**
Built from first principles in Python with the open-source HiGHS MIP solver to reproduce the simulation logic of commercial power-market tools such as **PLEXOS, PROMOD and AURORA**, with an integrated gas layer in the spirit of PLEXOS gas-electric co-optimization.

*Author: Cathy Kam* · **[Live app](https://ne-gas-power-pcm.streamlit.app)**

---

## Why this model

New England's power prices are set by natural gas, and in winter the gas price is set by
**pipeline scarcity**: firm heating (LDC) customers fill the interstate pipes first, and
generators bid for what is left, switch to oil, or rely on LNG. Most power models treat gas as an
exogenous price. This model makes the citygate gas price **endogenous**, so the same optimization
that commits and dispatches the fleet also clears the gas market each day.

## What it does

* **365-day rolling-horizon simulation**: each day is a 36-hour MIP (24 h committed + 12 h look-ahead), with unit status, ramp state, storage state of charge and oil inventory carried forward.
* **Unit commitment**: ~40 clustered unit groups with integer commitment, start costs, min up/down, ramp limits and no-load + incremental heat-rate curves.
* **Dual-fuel operation**: gas or oil mode per hour, oil derate, on-site oil inventory with refill.
* **Integrated gas market**: daily gas balance (firm LDC demand + power burn) against a stepwise Algonquin-citygate supply stack: pipeline basis bands, LNG send-out, LDC peak-shaving. The shadow price is the daily citygate gas price.
* **8 ISO-NE load zones** with transmission limits and the North-South, Boston Import, SEMA/RI Export and CT Import interfaces.
* **Market prices** from an LP re-solve with commitment fixed (standard PCM practice), with **fast-start pricing** for quick-start peakers as ISO-NE does.
* 30-minute operating reserves with an RCPF-style shortage price; pumped hydro, batteries, hydro energy budgets; Quebec, New Brunswick and New York interchange.

See [docs/methodology.md](docs/methodology.md) for the full formulation and a **PLEXOS/PROMOD crosswalk**.

## Calibration: 2025 backcast

| Metric | Model (2025 backcast) | ISO-NE reported, 2025 |
|---|---|---|
| Average energy price | $68.1/MWh (load-weighted) | $71.8 day-ahead / $65.9 real-time |
| Average New England gas price | $6.80/MMBtu | $6.27/MMBtu |
| Net energy for load | 118 TWh | ~118 TWh (implied by IMM cost per MWh) |

Reported figures: ISO-NE External Market Monitor, 2025 annual report (summarized by ISO Newswire, June 2026).
The backcast uses an approximate 2025 Henry Hub strip and weak Quebec imports, consistent with 2025 conditions.

## Scenario results (2027 study year, same weather year in every case)

| Scenario | Load-wtd LMP ($/MWh) | Avg gas ($/MMBtu) | Winter gas | Energy cost ($B) | CO2 (Mt) | Oil gen (GWh) | Cold-week LMP | Cold-week oil share |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 2025 backcast (calibration) | 68.14 | 6.80 | 13.09 | 8.04 | 21.21 | 737 | 116.5 | 0.1% |
| **2027 Base** | **70.75** | **7.08** | **13.20** | **8.56** | **18.52** | **788** | **114.8** | **0.1%** |
| Arctic cold snap | 73.87 | 7.18 | 13.71 | 8.93 | 18.73 | 1,505 | 207.0 | 15.6% |
| Cold snap + LNG import loss | 77.60 | 7.53 | 14.89 | 9.39 | 18.92 | 2,687 | 209.6 | 17.2% |
| Offshore wind + storage build | 65.16 | 6.71 | 12.08 | 7.88 | 15.15 | 617 | 98.0 | 0.0% |
| Pipeline +0.5 Bcf/d | 64.28 | 6.24 | 11.03 | 7.77 | 18.71 | 399 | 96.4 | 0.0% |
| High carbon price ($50/t) | 82.03 | 7.06 | 13.19 | 9.92 | 15.93 | 775 | 122.4 | 0.1% |

"Cold week" = Jan 20-28, the window of the arctic event. Energy cost = sum of zonal LMP x load.

## Key findings

1. **The binding constraint is the gas pipe, not the wires.** Electric interfaces almost never bind; on cold days firm LDC demand fills the pipelines and power generators become the marginal gas buyer.
2. **A 9-day cold snap more than doubles energy costs for that week** (about $1.0B vs. $0.45B in the base) as oil fills 16% of generation and LMPs rise to about $207/MWh.
3. **Losing LNG send-out barely moves price but sharply raises oil dependence.** Once oil is the marginal fuel, LMPs are capped near oil cost; the LNG loss shows up as 79% more annual oil generation, a reliability and inventory risk rather than a price signal.
4. **Gas infrastructure and offshore wind both cut costs, through different channels.** +0.5 Bcf/d of pipeline lowers the load-weighted LMP by about $6.5/MWh, mainly by flattening the winter basis; 2.4 GW of offshore wind with 1 GW of storage lowers it by about $5.6/MWh and cuts CO2 by 18%.
5. **Energy margins alone do not support new gas plants.** Efficient combined cycles earn roughly $25-40/kW-yr of energy margin and peakers close to zero (see `results/*/unit_margins.csv`), which is the "missing money" that capacity markets are designed to fill.

![Cold snap with LNG loss](figures/cold_snap_lng_loss_dispatch.png)
![Gas balance](figures/cold_snap_gas_balance.png)
![Scenario comparison](figures/scenario_comparison.png)

## Live mode: the model on real data, refreshed daily

Every morning a GitHub Actions job (`.github/workflows/live-daily.yml`) pulls real data, re-solves the
model for the **past 7 days** and the **next 7 days**, and commits the results that the app's
**Live (real data)** tab shows.

| Source | What it supplies | Key |
|---|---|---|
| [Open-Meteo](https://open-meteo.com/) | Hourly temperature, solar radiation and 100 m wind speed for 8 New England cities and 2 wind sites, actual (31 days) and forecast (10 days) | none |
| [EIA-930](https://www.eia.gov/electricity/gridmonitor/) via the EIA API | ISO-NE hourly demand, EIA's demand forecast, generation by fuel | `EIA_API_KEY` |
| [ISO-NE ISO Express](https://www.iso-ne.com/isoexpress/) | Day-ahead hourly LMPs for the Internal Hub and the 8 load zones | none |
| NYMEX via Yahoo Finance | Henry Hub front month and the next 12 monthly contracts | none |
| EIA API | Daily Henry Hub spot and New York Harbor ULSD spot (oil-unit fuel cost) | `EIA_API_KEY` |

How the inputs are used (`src/negem/live.py`): temperature is weighted by zone load share and drives
load, LDC gas demand and the gas supply curve; past hours use actual ISO-NE demand, future hours use
the model's temperature-driven load scaled to the last 30 days of actual demand; wind and solar
capacity factors come from the forecast wind speeds and solar radiation; Henry Hub is the EIA spot
price for past days and the NYMEX front month for future days.

Each run is scored against ISO-NE's published day-ahead Internal Hub price, and its day-ahead
forecasts are saved to `results/live/track_record.csv` and scored as the actual prices are posted,
building an out-of-sample record. Every source is fetched independently; one outage keeps the previous
data and is flagged on the Live tab instead of stopping the run.

```bash
export EIA_API_KEY=...                  # optional locally; set as a repository secret for the daily job
PYTHONPATH=src python -m negem.live_data
PYTHONPATH=src python -m negem.live
```

## Interactive app

`streamlit run app.py` opens an explorer with eight tabs:

* **Overview**: headline results, key findings and the 2025 calibration table
* **Live (real data)**: the model re-solved each morning on real weather, load and prices; model vs. ISO-NE day-ahead prices, a 7-day forecast, and the day-ahead track record
* **Scenarios**: any metric across scenarios, generation mix, change vs. base, downloadable results
* **Cold snap**: hourly dispatch stack, LMP and gas price, and the daily gas balance for any window
* **Prices & network**: monthly prices, price duration curves, zonal LMPs, interface congestion and flows
* **Asset economics**: unit energy margins and how each scenario moves a given plant's margin
* **Run a scenario**: set Henry Hub, oil, CO2, pipeline and LNG capacity, offshore wind, storage, load growth and Quebec imports, then re-solve the model live for a 1-10 day window and compare it with the base
* **Methodology**: the full formulation and PLEXOS/PROMOD crosswalk

## Repository layout

```
data/            fleet, renewables, storage, zones, transmission, interfaces, imports (CSV)
scenarios/       one YAML per scenario: levers only, everything else inherits the base
src/negem/
  inputs.py      weather, load, renewables, hydro, gas supply stack
  lp.py          sparse MIP/LP builder on SciPy-HiGHS; MIP then LP pricing with fast-start relaxation
  model.py       36-hour SCUC/ED + gas-balance formulation
  simulate.py    rolling-horizon driver and state carry-over
  report.py      KPIs, event-window metrics, unit energy margins, figures
  live_data.py   connectors: Open-Meteo, EIA-930, ISO-NE day-ahead LMPs, NYMEX, EIA spot prices
  live.py        live run: past 7 + next 7 days on real inputs, scoring and track record
data/live/       daily-refreshed real data (written by the live job)
results/live/    latest live run and the day-ahead track record
results/         hourly, daily and unit-level outputs per scenario; summary.csv
figures/         publication charts
app.py           Streamlit explorer
tests/           energy-balance, price and flow-limit checks
```

## Run it

```bash
pip install -r requirements.txt
python run_scenarios.py               # all scenarios (~7-10 min each)
python run_scenarios.py base_2027     # one scenario
PYTHONPATH=src python -m negem.report # summary table + figures
pytest -q
streamlit run app.py
```

Add a scenario by dropping a YAML file in `scenarios/` that overrides any field of `Scenario`
in `src/negem/inputs.py` (gas strip, carbon price, pipeline/LNG capacity, offshore wind, retirements, load growth, hydrology).

## Limitations

The fleet is an aggregated, approximate representation built from public plant data. The scenario
studies use a synthetic but seeded and calibrated weather year; the live mode uses real weather, load
and prices. New England citygate (Algonquin) gas prices are not freely published, so the model's gas
price is not scored in live mode. The network is zonal, not nodal,
and outages are expected derates rather than Monte Carlo draws. The structure is designed so real
ISO-NE hourly load, NOAA weather and EIA-923 unit data can be dropped in without changing the
optimization code.
