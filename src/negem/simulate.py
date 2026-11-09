"""Rolling-horizon chronological simulation (365 daily steps, 24 h + 12 h look-ahead)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import inputs as I
from .model import solve_horizon, CO2_GAS, CO2_OIL

LOOKAHEAD = 12


def prepare(sc: I.Scenario) -> dict:
    st = I.load_static(sc)
    w = I.build_weather(sc)
    load = I.build_load(sc, w, st["zones"])
    pv = I.solar_profile(w)
    wind = I.wind_profiles(w)
    month = w["month"].values
    tmean_daily = w["tmean"].values[::24]
    hdd_daily = np.clip(65 - tmean_daily, 0, None)
    hh_daily = np.array(sc.hh_monthly)[w["month"].values[::24] - 1]
    # Cold snaps also tighten Henry Hub (freeze-offs) - modest national pass-through
    hh_daily = hh_daily + 0.04 * np.clip(hdd_daily - 45, 0, None)
    ldc = I.ldc_demand_bcfd(tmean_daily) * I.MMBTU_PER_BCF
    nyb = sc.ny_basis + 0.12 * np.clip(hdd_daily - 25, 0, None)
    hh_h = np.repeat(hh_daily, 24)
    nyb_h = np.repeat(nyb, 24)
    imp_offer = {"HQ Phase II": np.full(I.HOURS, 22.0), "Highgate": np.full(I.HOURS, 24.0),
                 "New Brunswick": 8.0 * (hh_h + 2.5) + 3,
                 "NYISO": 7.4 * (hh_h + nyb_h) + 6}
    exp_bid = {"NYISO": 7.0 * (hh_h + nyb_h) - 6}
    hq = I.hq_availability(month, sc.hq_mult)
    imp_avail = {"HQ Phase II": hq, "Highgate": hq, "New Brunswick": np.full(I.HOURS, 0.8),
                 "NYISO": np.ones(I.HOURS)}
    # Nuclear refueling outages (spring/fall shoulders, ~30 days)
    nuc = {n: np.ones(I.HOURS) for n in ["Seabrook", "Millstone 2", "Millstone 3"]}
    nuc["Seabrook"][24 * 95:24 * 125] = 0
    nuc["Millstone 2"][24 * 280:24 * 308] = 0
    nuc["Millstone 3"][24 * 120:24 * 148] = 0
    return dict(sc=sc, st=st, w=w, load=load, pv=pv, wind=wind, hdd=hdd_daily, hh=hh_daily,
                ldc=ldc, imp_offer=imp_offer, exp_bid=exp_bid, imp_avail=imp_avail, nuc=nuc,
                hydro_cf=I.hydro_budget(month))


def _slice(arr, a, b):
    out = arr[a:b]
    if len(out) < b - a:
        out = np.concatenate([out, arr[-(b - a - len(out)):]])
    return out


def run(sc: I.Scenario, days: int = 365, start_day: int = 0, verbose=True, D: dict | None = None) -> dict:
    """Simulate `days` days. D is the prepared input set; by default the synthetic study year
    from prepare(sc). The live model passes real weather, load and gas prices instead."""
    D = prepare(sc) if D is None else D
    ndays = len(D["hdd"])
    st = D["st"]
    fleet = st["fleet"]
    G = len(fleet)
    oil_max = (fleet["oil_inv_hours"] * fleet["pmax"] * fleet["hr_fl"] * fleet["count"]).values
    state = dict(U=np.zeros(G), P=np.zeros(G),
                 s_hist=[np.zeros(24) for _ in range(G)], sd_hist=[np.zeros(24) for _ in range(G)],
                 soc=(0.5 * st["sto"]["mwh"]).values.astype(float),
                 oil_inv=oil_max.copy(), oil_refill=0.20 * oil_max)
    # warm start: nuclear/CC base online
    for g, u in fleet.iterrows():
        if u.tech == "CC":
            state["U"][g] = u["count"]
            state["P"][g] = u.pmin * u["count"]
        state["s_hist"][g] = np.zeros(24)

    Z = list(st["zones"]["zone"])
    hourly, daily, unit_rows = [], [], []
    t0 = time.time()
    for d in range(start_day, start_day + days):
        a = 24 * d
        H = 24 + LOOKAHEAD if d < ndays - 1 else 24
        b = a + H
        dd1 = min(d + 1, ndays - 1)
        winter = lambda dd: (D["w"]["month"].values[24 * dd] in (11, 12, 1, 2, 3))
        ts = dict(
            load={z: _slice(D["load"][z].values, a, b) for z in Z},
            solar=_slice(D["pv"], a, b),
            wind_on=_slice(D["wind"]["wind_on"], a, b),
            wind_off=_slice(D["wind"]["wind_off"], a, b),
            hydro_cf=_slice(D["hydro_cf"], a, b),
            nuc_avail={k: _slice(v, a, b) for k, v in D["nuc"].items()},
            imp_offer={k: _slice(v, a, b) for k, v in D["imp_offer"].items()},
            exp_bid={k: _slice(v, a, b) for k, v in D["exp_bid"].items()},
            imp_avail={k: _slice(v, a, b) for k, v in D["imp_avail"].items()},
            gas_steps=[I.gas_supply_steps(sc, D["hh"][d], winter(d)),
                       I.gas_supply_steps(sc, D["hh"][dd1], winter(dd1))],
            ldc_mmbtu=[D["ldc"][d], D["ldc"][dd1]],
        )
        px = dict(oil=sc.oil_price, co2=sc.co2_price)
        R = solve_horizon(st, ts, state, px, H)
        x, dual = R["x"], R["dual"]

        # ---------- extract committed day (first 24 h) ----------
        h = np.arange(24)
        rec = {"hour": a + h}
        for zi_, z in enumerate(Z):
            rec[f"lmp_{z}"] = dual[R["bal"][zi_, h]]
            rec[f"load_{z}"] = D["load"][z].values[a:a + 24]
            rec[f"uns_{z}"] = x[R["uns"][zi_, h]]
        rec["res_price"] = dual[np.array(R["res_rows"])[h]]
        rec["res_short"] = x[R["rshort"][h]]
        gen_by = {}
        gas_burn_d = 0.0
        oil_burn_d = 0.0
        co2 = np.zeros(24)
        for un in R["units"]:
            g = un["g"]
            u = fleet.loc[g]
            for m, pv_ in un["p_mode"].items():
                p = x[pv_[h]]
                key = {"nuc": "Nuclear", "bio": "Biomass/Refuse"}.get(m)
                if key is None:
                    key = ("Gas " + un["tech"]) if m == "gas" else "Oil"
                gen_by[key] = gen_by.get(key, 0) + p
                if m in ("gas", "oil"):
                    fuel = x[un["U"][m][h]] * u.nl_fuel + p * u.ihr
                    if m == "gas":
                        gas_burn_d += fuel.sum()
                        co2 += fuel * CO2_GAS
                    else:
                        oil_burn_d += fuel.sum()
                        co2 += fuel * CO2_OIL
                    unit_rows.append((d, u["name"], u.tech, u.zone, m, p.sum(),
                                      fuel.sum(), (p * rec[f"lmp_{u.zone}"]).sum()))
                else:
                    unit_rows.append((d, u["name"], u.tech, u.zone, m, p.sum(), 0.0,
                                      (p * rec[f"lmp_{u.zone}"]).sum()))
            # state carry
            if "U" in un:
                Ut = sum(x[un["U"][m][23]] for m in un["U"])
                Pt = sum(x[un["p_mode"][m][23]] for m in un["p_mode"])
                state["U"][g] = round(Ut)
                state["P"][g] = Pt
                state["s_hist"][g] = x[un["s"][h]]
                state["sd_hist"][g] = x[un["sd"][h]]
                if "oil" in un["U"]:
                    burn = (x[un["U"]["oil"][h]] * u.nl_fuel + x[un["p_mode"]["oil"][h]] * u.ihr).sum()
                    state["oil_inv"][g] = min(oil_max[g], state["oil_inv"][g] - burn
                                              + state["oil_refill"][g])
        for name, typ, zone, p, avail in R["ren"]:
            key = {"wind_on": "Onshore Wind", "wind_off": "Offshore Wind",
                   "solar": "Utility Solar", "hydro": "Hydro"}[typ]
            gen_by[key] = gen_by.get(key, 0) + x[p[h]]
            if typ != "hydro":
                rec[f"curt_{name}"] = avail[h] - x[p[h]]
        sto_net = np.zeros(24)
        for k, (name, ch, dis, soc) in enumerate(R["sto"]):
            sto_net += x[dis[h]] - x[ch[h]]
            state["soc"][k] = x[soc[23]]
        gen_by["Storage (net)"] = sto_net
        for tie, xi, e in R["imp"]:
            gen_by[f"Imports {tie}"] = x[xi[h]] - (x[e[h]] if e is not None else 0)
        for k, v in gen_by.items():
            rec[f"gen_{k}"] = v
        for l, f in R["flows"].items():
            rec[f"flow_{l}"] = x[f[h]]
        for name, rows in R["if_rows"].items():
            rec[f"ifprice_{name}"] = -dual[np.array(rows)[h]]
        rec["co2_tons"] = co2
        hourly.append(pd.DataFrame(rec))

        gas_px = dual[R["gas_rows"][0]]
        y0 = x[R["gas_y"][0]]
        steps = ts["gas_steps"][0]
        daily.append(dict(day=d, date=D["w"].index[a].date(), hdd=D["hdd"][d], hh=D["hh"][d],
                          gas_price=gas_px, ldc_bcf=D["ldc"][d] / I.MMBTU_PER_BCF,
                          power_burn_bcf=gas_burn_d / I.MMBTU_PER_BCF,
                          pipeline_util=y0[:6].sum()
                          / (sc.pipeline_bcfd * I.MMBTU_PER_BCF),
                          lng_bcf=y0[6] / I.MMBTU_PER_BCF, peakshave_bcf=y0[7] / I.MMBTU_PER_BCF,
                          oil_burn_mmbtu=oil_burn_d,
                          oil_inventory_pct=float(state["oil_inv"].sum() / max(oil_max.sum(), 1))))
        if verbose and (d % 30 == 0 or d == start_day + days - 1):
            print(f"  [{sc.name}] day {d:3d}  gas ${gas_px:6.2f}  "
                  f"LMP NEMA ${rec['lmp_NEMA'].mean():6.1f}  elapsed {time.time()-t0:5.0f}s",
                  flush=True)
    hourly = pd.concat(hourly, ignore_index=True)
    hourly.index = D["w"].index[hourly["hour"].values]
    units = pd.DataFrame(unit_rows, columns=["day", "unit", "tech", "zone", "mode", "mwh",
                                             "fuel_mmbtu", "energy_revenue"])
    return dict(hourly=hourly, daily=pd.DataFrame(daily), units=units, prep=D)


def save(out: dict, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    out["hourly"].to_csv(folder / "hourly.csv.gz", compression="gzip")
    out["daily"].to_csv(folder / "daily.csv", index=False)
    out["units"].to_csv(folder / "units.csv.gz", index=False, compression="gzip")
