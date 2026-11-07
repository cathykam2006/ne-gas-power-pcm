# Methodology

## 1. What the model is

A chronological **production cost model (PCM)** for ISO New England, co-optimized with the
**New England natural gas system**, written from first principles in Python and solved with the
open-source HiGHS MIP solver. It reproduces the core simulation logic of commercial tools such as
PLEXOS, PROMOD and AURORA, and adds an explicit gas-electric interdependency layer comparable to
PLEXOS integrated gas-electric or Encompass-style fuel modeling.

It simulates 8,760 hours as 365 rolling daily steps. Each step optimizes **36 hours** (24 committed
hours + 12-hour look-ahead), carries unit status, ramp state, storage state of charge and oil
inventory into the next day, and reports only the committed 24 hours. That mirrors the standard
"step size / look-ahead" setting in PLEXOS ST Schedule and PROMOD's daily commitment.

## 2. PLEXOS / PROMOD crosswalk

| Concept | PLEXOS / PROMOD | This model |
|---|---|---|
| Chronological simulation | ST Schedule, step = 1 day, look-ahead | 365 daily MIP steps, 24 h + 12 h look-ahead |
| Unit commitment | MIP with integer units (Units, Min Up/Down, Start Cost) | Clustered integer commitment per unit group (0..N online) |
| Heat-rate curves | Heat Rate Base + Heat Rate Incr | No-load fuel + incremental heat rate |
| Dual fuel | Fuel transition / multiple Fuels per Generator | Separate gas and oil modes with own commitment, oil derate |
| Fuel constraints | Fuel Offtake Max, Fuel Storage | Daily oil inventory with refill rate |
| Gas network | Gas Node, Gas Pipeline, Gas Demand, Gas Field | Daily gas node: firm LDC demand + power burn vs. stepwise supply stack |
| Gas price | Input or Gas Node shadow price (integrated) | Gas-balance shadow price (endogenous) |
| Transmission | Zonal pipeline / Interface constraints | Zonal transport with ISO-NE interface limits |
| Prices | LP re-solve with integers fixed | Same: MIP, then LP with commitment fixed for duals |
| Reserves | Reserve class with Min Provision, VoRS | 30-min operating reserve with $1,000 RCPF-style shortage |
| Storage | Storage / Battery objects | Pumped hydro and batteries with SoC and round-trip losses |
| Hydro | Max Energy Month / Day | Monthly CF-based energy budget with minimum flow |
| Interchange | Purchasers / Physical contracts | Quebec, New Brunswick fixed/gas-linked offers; NYISO bidirectional |

## 3. Formulation (one horizon)

**Sets.** Zones z, hours t, unit groups g with N_g identical units, fuel modes m in {gas, oil},
gas days k, supply tranches j, transmission links l, interfaces i.

**Objective.** Minimize

  Sum_j price_jk * y_jk (gas purchases at the citygate)
+ Sum oil fuel * (oil price + CO2 price * 0.0805)
+ Sum gas fuel * CO2 price * 0.0585
+ VOM * generation + start cost * starts
+ import offers - export bids
+ 5,000 * unserved energy + 1,000 * reserve shortfall

**Unit constraints** (per g, t)

* Sum_m u_gmt <= N_g; u integer
* Sum_m u_gmt - Sum_m u_gm,t-1 = s_gt - sd_gt (start-ups / shut-downs)
* Pmin_g * u_gmt <= p_gmt <= Pmax_gm * u_gmt
* Sum_m p_gmt + r_gt <= Sum_m Pmax_gm u_gmt (energy + reserve headroom)
* Ramp: |Delta P| <= ramp_g * u + Pmax * (start-ups or shut-downs)
* Min up: Sum over last MU hours of s <= u; min down: Sum over last MD hours of sd <= N - u
* Fuel_gmt = no-load_g * u_gmt + IHR_g * p_gmt
* Oil inventory: Sum_t Fuel_g,oil,t <= inventory_g (day 1), plus refill for look-ahead

**Gas node** (per gas day k)

  Sum_j y_jk  >=  LDC_k + Sum_{g,t in k} Fuel_g,gas,t,  with 0 <= y_jk <= Cap_j

The supply stack prices pipeline capacity in utilization bands (Henry Hub + basis of
$0.45 -> $21/MMBtu as the pipes approach 100%), then LNG send-out (global-LNG linked), then LDC
on-system peak-shaving. LDC demand is inelastic (firm customers have priority), so on cold days
power generators are the marginal gas buyers - the defining feature of New England winter markets.
The dual of this constraint is the **daily citygate gas price**.

**Network, balance, reserves**

* Zonal balance: generation + storage discharge - charge + imports - exports + inflows - outflows
  + unserved - dump = load (net of behind-the-meter PV). Dual = **zonal LMP**.
* Link limits and interface limits (North-South, Boston Import, SEMA/RI Export, CT Import).
* Sum of online headroom (30-min ramp-limited), offline quick-start CTs and storage >= 2,300 MW.

## 4. Inputs and calibration

* **Fleet:** ~40 unit groups representing ISO-NE's nuclear, combined-cycle, peaker, oil steam,
  biomass/refuse, hydro, wind, solar and storage fleet, built from public plant capacities and
  typical heat rates (approximate, aggregated).
* **Weather:** one seeded synthetic weather year with Boston climatology, shared by all scenarios
  so differences are attributable to the scenario lever alone.
* **Load:** temperature-driven hourly load with daily/weekly shape, calibrated to ~118 TWh, split
  into 8 load zones, net of behind-the-meter solar.
* **Gas:** LDC demand = 0.62 + 0.0405 x HDD (Bcf/d); 3.45 Bcf/d firm pipeline, 0.70 Bcf/d LNG,
  1.1 Bcf/d LDC peak-shaving.
* **Calibration target:** ISO-NE External Market Monitor 2025 results - day-ahead energy prices
  averaged about $72/MWh and New England gas about $6.27/MMBtu, with carbon pricing adding roughly
  $9/MWh. The `backcast_2025` scenario is benchmarked against these figures (see README).

## 5. Limitations and next steps

* Zonal transport, not a nodal DC-OPF; no losses or contingency-constrained flows.
* Synthetic weather, renewables and load shapes; plug in ISO-NE hourly load, NOAA temperatures and
  EIA-923 unit data for a production-grade study.
* Deterministic: forced outages are an expected derate. A Monte Carlo outage draw and
  stochastic weather years are natural extensions.
* Capacity market (FCA/prompt auction) and ancillary revenues are out of scope; energy margins are
  reported for each unit as an input to asset valuation.
