"""One rolling-horizon step: security-constrained unit commitment & economic dispatch for
New England's 8 load zones, co-optimized with a daily New England gas balance.

Formulation highlights (what PLEXOS/PROMOD do, written out explicitly):
  * Clustered integer commitment per unit group (online count 0..N), start-up/shut-down,
    min up / min down, ramp limits, no-load + incremental heat-rate curves.
  * Dual-fuel units choose gas or oil mode each hour; oil burn limited by on-site inventory.
  * Gas is NOT an exogenous price: power burn + firm LDC demand must clear a stepwise
    Algonquin-citygate supply stack (pipeline basis bands -> LNG -> LDC peak-shaving).
    The gas-balance shadow price is the endogenous daily citygate gas price.
  * Zonal transport network with ISO-NE style interface limits (North-South, Boston Import,
    SEMA/RI Export, CT Import); zonal LMPs are balance-constraint duals.
  * 30-minute operating reserve requirement with an RCPF-style shortage penalty.
  * Pumped hydro / batteries with state of charge; conventional hydro with energy budgets;
    imports from Quebec, New Brunswick and New York (bidirectional).
"""
from __future__ import annotations

import numpy as np

from .lp import LP


class Row(dict):
    """dict with attribute access (much faster than pandas row lookups in tight loops)."""
    __getattr__ = dict.__getitem__

CO2_GAS = 0.0585   # short tons CO2 / MMBtu
CO2_OIL = 0.0805
CO2_BIO = 0.0      # treated as biogenic / RGGI-exempt
VOLL = 5000.0      # $/MWh unserved energy
RCPF = 1000.0      # $/MWh reserve shortage
RES_REQ = 2300.0   # MW, 30-min operating reserve
OIL_DERATE = 0.95


def solve_horizon(st: dict, ts: dict, state: dict, px: dict, H: int) -> dict:
    """st: static tables; ts: hourly/daily arrays for this horizon; state: carry-over;
    px: prices ($/MMBtu, $/ton); H: horizon hours (first 24 are committed/reported)."""
    fleet, zones, lines, ifs = st["fleet"], st["zones"], st["lines"], st["ifs"]
    Z = list(zones["zone"])
    zi = {z: i for i, z in enumerate(Z)}
    lp = LP()
    T = np.arange(H)
    inj = [[[] for _ in range(H)] for _ in Z]          # zonal injection terms per hour
    res_terms = [[] for _ in range(H)]
    gas_burn_terms = {0: [], 1: []}                     # per gas day
    day_of = (T >= 24).astype(int)
    frac1 = max(H - 24, 0) / 24.0
    units = []
    fast_start = []

    # ---------------- thermal & firm units ----------------
    for g, u in enumerate(map(Row, fleet.to_dict('records'))):
        z = zi[u.zone]
        N = int(u["count"])
        rec = {"g": g, "name": u["name"], "tech": u.tech, "zone": u.zone}
        if u.tech in ("NUC", "BIO"):
            avail = ts["nuc_avail"][u["name"]] if u.tech == "NUC" else np.ones(H)
            cap = u.pmax_eff * N * avail
            lo = 0.92 * cap if u.tech == "NUC" else 0.0
            fuel_px = 0.65 if u.tech == "NUC" else 2.5
            p = lp.var(H, lb=lo, ub=cap, cost=u.vom + fuel_px * u.hr_fl)
            for t in T:
                inj[z][t].append((int(p[t]), 1.0))
            rec["p_mode"] = {"nuc" if u.tech == "NUC" else "bio": p}
            units.append(rec)
            continue

        modes = []
        if u.gas_capable:
            modes.append("gas")
        if u.oil_capable:
            modes.append("oil")
        U, P = {}, {}
        for m in modes:
            pmax_m = u.pmax_eff * (OIL_DERATE if (m == "oil" and u.fuel == "gas") else 1.0)
            U[m] = lp.var(H, lb=0, ub=N, integer=True)
            P[m] = lp.var(H, lb=0, ub=pmax_m * N, cost=u.vom)
            fuel_cost_nl = 0.0
            if m == "oil":
                # oil fuel and CO2 cost carried on the unit (gas cost comes via the gas node)
                c_oil = px["oil"] + px["co2"] * CO2_OIL
                lp.c[-1] = lp.c[-1] + c_oil * u.ihr          # P cost
                lp.c[-2] = lp.c[-2] + c_oil * u.nl_fuel      # U cost (no-load)
            else:
                c_g = px["co2"] * CO2_GAS
                lp.c[-1] = lp.c[-1] + c_g * u.ihr
                lp.c[-2] = lp.c[-2] + c_g * u.nl_fuel
                for t in T:
                    gas_burn_terms[day_of[t]] += [(U[m][t], u.nl_fuel), (P[m][t], u.ihr)]
            for t in T:
                lp.con([(P[m][t], 1.0), (U[m][t], -u.pmin)], lo=0)          # Pmin
                lp.con([(P[m][t], 1.0), (U[m][t], -pmax_m)], hi=0)          # Pmax (per mode)
            del fuel_cost_nl
        s = lp.var(H, cost=u.start_cost)
        sd = lp.var(H)
        r_on = lp.var(H)
        r_off = lp.var(H) if u.tech == "CT" else None
        Up, Pp = state["U"][g], state["P"][g]
        hs, hsd = state["s_hist"][g], state["sd_hist"][g]
        for t in T:
            Ut = [(U[m][t], 1.0) for m in modes]
            Pt = [(P[m][t], 1.0) for m in modes]
            lp.con(Ut, hi=N)
            # commitment transition
            if t == 0:
                lp.con(Ut + [(s[t], -1), (sd[t], 1)], lo=Up, hi=Up)
            else:
                Ut1 = [(U[m][t - 1], -1.0) for m in modes]
                lp.con(Ut + Ut1 + [(s[t], -1), (sd[t], 1)], lo=0, hi=0)
            # capacity incl. reserve headroom (use full-capacity mode as reference)
            pm = [(U[m][t], -(u.pmax_eff * (OIL_DERATE if (m == "oil" and u.fuel == "gas") else 1)))
                  for m in modes]
            lp.con(Pt + [(r_on[t], 1.0)] + pm, hi=0)
            lp.con([(r_on[t], 1.0)] + [(U[m][t], -0.5 * u.ramp_frac * u.pmax) for m in modes], hi=0)
            if r_off is not None:
                lp.con([(r_off[t], 1.0)] + [(U[m][t], u.pmax_eff) for m in modes],
                       hi=u.pmax_eff * N)
                res_terms[t].append((r_off[t], 1.0))
            res_terms[t].append((r_on[t], 1.0))
            # ramping
            R = u.ramp_frac * u.pmax
            if t == 0:
                lp.con(Pt + [(U[m][t], -R) for m in modes] + [(s[t], -u.pmax)], hi=Pp)
                lp.con([(a, -1.0) for a, _ in Pt] + [(U[m][t], -R) for m in modes]
                       + [(sd[t], -u.pmax)], hi=-Pp)
            else:
                Pt1 = [(P[m][t - 1], -1.0) for m in modes]
                lp.con(Pt + Pt1 + [(U[m][t], -R) for m in modes] + [(s[t], -u.pmax)], hi=0)
                lp.con([(a, -1.0) for a, _ in Pt] + [(a, 1.0) for a, _ in Pt1]
                       + [(U[m][t], -R) for m in modes] + [(sd[t], -u.pmax)], hi=0)
            # min up / min down (with history from previous day)
            MU, MD = int(u.min_up), int(u.min_down)
            if MU > 1:
                win = [(s[tau], 1.0) for tau in range(max(0, t - MU + 1), t + 1)]
                hist = hs[len(hs) - max(0, MU - 1 - t):] if t < MU - 1 else []
                lp.con(win + [(U[m][t], -1.0) for m in modes], hi=-float(np.sum(hist)))
            if MD > 1:
                win = [(sd[tau], 1.0) for tau in range(max(0, t - MD + 1), t + 1)]
                hist = hsd[len(hsd) - max(0, MD - 1 - t):] if t < MD - 1 else []
                lp.con(win + [(U[m][t], 1.0) for m in modes], hi=N - float(np.sum(hist)))
            for a, _ in Pt:
                inj[z][t].append((a, 1.0))
        # oil inventory
        if u.oil_capable:
            oil_terms = [(U["oil"], u.nl_fuel), (P["oil"], u.ihr)]
            d0 = [(U["oil"][:24], u.nl_fuel), (P["oil"][:24], u.ihr)]
            lp.con(d0, hi=state["oil_inv"][g])
            lp.con(oil_terms, hi=state["oil_inv"][g] + frac1 * state["oil_refill"][g])
        if u.tech == "CT":
            for m in modes:
                fast_start += list(U[m])
        rec.update(U=U, p_mode=P, s=s, sd=sd)
        units.append(rec)

    # ---------------- renewables & hydro ----------------
    ren_vars = []
    for r in map(Row, st["ren"].to_dict("records")):
        z = zi[r.zone]
        if r.type == "hydro":
            p = lp.var(H, lb=0.12 * r.mw * ts["hydro_cf"], ub=r.mw, cost=1.0)
            lp.con([(p, 1.0)], hi=float(np.sum(ts["hydro_cf"] * r.mw)))
        else:
            avail = r.mw * ts[r.type]
            p = lp.var(H, lb=0, ub=avail, cost=0.0)
        for t in T:
            inj[z][t].append((p[t], 1.0))
        ren_vars.append((r["name"], r.type, r.zone, p, r.mw * (ts[r.type] if r.type != "hydro" else 0)))

    # ---------------- storage ----------------
    sto_vars = []
    for k, s_ in enumerate(map(Row, st["sto"].to_dict("records"))):
        z = zi[s_.zone]
        ch = lp.var(H, ub=s_.mw)
        dis = lp.var(H, ub=s_.mw, cost=0.5)
        soc = lp.var(H, ub=s_.mwh)
        rs = lp.var(H, ub=s_.mw)
        soc0 = state["soc"][k]
        for t in T:
            prev = [(soc[t - 1], -1.0)] if t > 0 else []
            lp.con([(soc[t], 1.0), (ch[t], -s_.rte), (dis[t], 1.0)] + prev,
                   lo=soc0 if t == 0 else 0, hi=soc0 if t == 0 else 0)
            lp.con([(rs[t], 1.0), (dis[t], 1.0)], hi=s_.mw)
            lp.con([(rs[t], 1.0), (soc[t], -1.0)], hi=0)
            inj[z][t] += [(dis[t], 1.0), (ch[t], -1.0)]
            res_terms[t].append((rs[t], 1.0))
        lp.con([(soc[H - 1], 1.0)], lo=min(soc0, 0.98 * s_.mwh))
        sto_vars.append((s_["name"], ch, dis, soc))

    # ---------------- imports / exports ----------------
    imp_vars = []
    for im in map(Row, st["imports"].to_dict("records")):
        z = zi[im.zone]
        cap = im.cap_mw * ts["imp_avail"][im.tie]
        x = lp.var(H, ub=cap, cost=ts["imp_offer"][im.tie])
        e = None
        for t in T:
            inj[z][t].append((x[t], 1.0))
        if im.can_export:
            e = lp.var(H, ub=im.cap_mw, cost=-ts["exp_bid"][im.tie])
            for t in T:
                inj[z][t].append((e[t], -1.0))
                lp.con([(x[t], 1.0), (e[t], 1.0)], hi=im.cap_mw)
        imp_vars.append((im.tie, x, e))

    # ---------------- network ----------------
    flows = {}
    for l in map(Row, lines.to_dict("records")):
        f = lp.var(H, lb=-l.limit_mw, ub=l.limit_mw)
        flows[l.link] = f
        for t in T:
            inj[zi[l["from"]]][t].append((f[t], -1.0))
            inj[zi[l["to"]]][t].append((f[t], 1.0))
    if_rows = {}
    for it in map(Row, ifs.to_dict("records")):
        if_rows[it.interface] = [lp.con([(flows[k][t], c) for k, c in it.terms.items()],
                                        hi=it.limit_mw) for t in T]

    # ---------------- zonal balance ----------------
    uns = lp.var((len(Z), H), cost=VOLL)
    dump = lp.var((len(Z), H), cost=0.0)
    bal = np.zeros((len(Z), H), dtype=int)
    for z in range(len(Z)):
        load = ts["load"][Z[z]]
        for t in T:
            bal[z, t] = lp.con(inj[z][t] + [(uns[z, t], 1.0), (dump[z, t], -1.0)],
                               lo=load[t], hi=load[t])

    # ---------------- reserves ----------------
    rshort = lp.var(H, cost=RCPF)
    res_rows = [lp.con(res_terms[t] + [(rshort[t], 1.0)], lo=RES_REQ) for t in T]

    # ---------------- gas balance (endogenous citygate price) ----------------
    gas_rows, gas_y = [], []
    for k in (0, 1):
        if k == 1 and frac1 == 0:
            break
        scale = 1.0 if k == 0 else frac1
        steps = ts["gas_steps"][k]
        y = lp.var(len(steps), ub=[c * scale for c, _ in steps], cost=[p for _, p in steps])
        burn = [(v, -c) for v, c in gas_burn_terms[k]]
        gas_rows.append(lp.con([(y, 1.0)] + burn, lo=ts["ldc_mmbtu"][k] * scale))
        gas_y.append(y)

    # ---------------- solve ----------------
    x_mip, obj = lp.solve_mip()
    # A relative MIP gap can hide small amounts of avoidable shortfall (VOLL is a tiny share
    # of a winter day's cost); tighten and re-solve whenever shortfall appears.
    if x_mip[uns.ravel()].sum() + x_mip[rshort].sum() > 1.0:
        x_mip, obj = lp.solve_mip(gap=1e-4, time_limit=60, rebuild=False)
    x, dual = lp.solve_lp_fixed(x_mip, relax=fast_start)

    return dict(x=x, dual=dual, units=units, ren=ren_vars, sto=sto_vars, imp=imp_vars,
                flows=flows, if_rows=if_rows, bal=bal, uns=uns, dump=dump,
                res_rows=res_rows, rshort=rshort, gas_rows=gas_rows, gas_y=gas_y,
                obj=obj, zones=Z)
