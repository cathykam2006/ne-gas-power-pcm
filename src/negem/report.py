"""Post-processing: scenario KPIs, calibration table and publication figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RES, FIG = ROOT / "results", ROOT / "figures"
ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA"]
FUEL_GROUPS = {
    "Nuclear": ["gen_Nuclear"], "Natural gas": ["gen_Gas CC", "gen_Gas CT", "gen_Gas ST"],
    "Oil": ["gen_Oil"], "Hydro": ["gen_Hydro"], "Wind": ["gen_Onshore Wind", "gen_Offshore Wind"],
    "Solar": ["gen_Utility Solar"], "Biomass/Refuse": ["gen_Biomass/Refuse"],
    "Net imports": ["gen_Imports HQ Phase II", "gen_Imports Highgate",
                    "gen_Imports New Brunswick", "gen_Imports NYISO"],
    "Storage (net)": ["gen_Storage (net)"]}
COLORS = {"Nuclear": "#7b5ea7", "Natural gas": "#e07b39", "Oil": "#8c564b", "Hydro": "#3a7dc1",
          "Wind": "#4bb3a5", "Solar": "#f2c14e", "Biomass/Refuse": "#6b8e23",
          "Net imports": "#9aa5b1", "Storage (net)": "#d16ba5"}


def load(name):
    f = RES / name
    h = pd.read_csv(f / "hourly.csv.gz", index_col=0, parse_dates=True)
    d = pd.read_csv(f / "daily.csv", parse_dates=["date"])
    u = pd.read_csv(f / "units.csv.gz")
    return h, d, u


def kpis(name, label):
    h, d, u = load(name)
    loadcols = [f"load_{z}" for z in ZONES]
    L = h[loadcols].to_numpy()
    P = h[[f"lmp_{z}" for z in ZONES]].to_numpy()
    sys_lmp = (L * P).sum() / L.sum()
    winter = h.index.month.isin([12, 1, 2])
    lw = (L[winter] * P[winter]).sum() / L[winter].sum()
    mix = {k: h[[c for c in v if c in h]].sum().sum() / 1e6 for k, v in FUEL_GROUPS.items()}
    # unit economics: CC fleet energy margin (revenue - fuel - CO2 - VOM)
    gas_px = d.set_index("day")["gas_price"]
    u["gas_px"] = u["day"].map(gas_px)
    out = dict(
        scenario=name, label=label,
        load_weighted_lmp=round(sys_lmp, 2), winter_lmp=round(lw, 2),
        lmp_NEMA=round(h["lmp_NEMA"].mean(), 2), lmp_ME=round(h["lmp_ME"].mean(), 2),
        lmp_SEMA=round(h["lmp_SEMA"].mean(), 2),
        avg_gas_price=round(d["gas_price"].mean(), 2),
        winter_gas_price=round(d.loc[d["date"].dt.month.isin([12, 1, 2]), "gas_price"].mean(), 2),
        max_gas_price=round(d["gas_price"].max(), 2),
        avg_henry_hub=round(d["hh"].mean(), 2),
        power_burn_bcf=round(d["power_burn_bcf"].sum(), 1),
        lng_bcf=round(d["lng_bcf"].sum(), 1),
        oil_gen_gwh=round(h["gen_Oil"].sum() / 1e3, 1),
        min_oil_inventory_pct=round(100 * d["oil_inventory_pct"].min(), 1),
        co2_mtons=round(h["co2_tons"].sum() / 1e6, 2),
        unserved_mwh=round(h[[f"uns_{z}" for z in ZONES]].sum().sum(), 0),
        reserve_short_hours=int((h["res_short"] > 1).sum()),
        curtailment_gwh=round(h.filter(like="curt_").sum().sum() / 1e3, 1),
        energy_cost_busd=round((L * P).sum() / 1e9, 2),
        load_twh=round(L.sum() / 1e6, 1),
        **{f"mix_{k}": round(v, 2) for k, v in mix.items()},
    )
    for c in h.filter(like="ifprice_").columns:
        nm = c.replace("ifprice_", "")
        out[f"bind_hours_{nm}"] = int((h[c] > 0.01).sum())
        out[f"cong_rent_{nm}"] = round(h[c].mean(), 2)
    return out, h, d, u


def event_window(h, d, start="2027-01-20", end="2027-01-28"):
    w = h.loc[start:end]
    L = w[[f"load_{z}" for z in ZONES]].to_numpy()
    P = w[[f"lmp_{z}" for z in ZONES]].to_numpy()
    gen = w.filter(like="gen_").clip(lower=0).sum().sum()
    dd = d.set_index("date").loc[start:end]
    return dict(event_lmp=round((L * P).sum() / L.sum(), 1),
                event_peak_lmp=round(w["lmp_NEMA"].max(), 1),
                event_gas_price=round(dd["gas_price"].mean(), 2),
                event_oil_share_pct=round(100 * w["gen_Oil"].sum() / gen, 1),
                event_gas_share_pct=round(100 * w.filter(like="gen_Gas").sum().sum() / gen, 1),
                event_energy_cost_musd=round((L * P).sum() / 1e6, 0))


def unit_margins(u, d, cfg):
    """Annual energy margin by unit (revenue - fuel - CO2 - VOM), $/kW-yr: asset-valuation view."""
    fleet = pd.read_csv(ROOT / "data" / "fleet.csv").set_index("name")
    gas_px = d.set_index("day")["gas_price"]
    co2 = cfg.get("co2_price", 22.0)
    oil = cfg.get("oil_price", 19.0)
    u = u.copy()
    px = np.where(u["mode"] == "gas", u["day"].map(gas_px), np.where(u["mode"] == "oil", oil, 0.0))
    ef = np.where(u["mode"] == "gas", 0.0585, np.where(u["mode"] == "oil", 0.0805, 0.0))
    other = np.where(u["mode"] == "nuc", 0.65, np.where(u["mode"] == "bio", 2.5, 0.0))
    u["cost"] = (u["fuel_mmbtu"] * (px + co2 * ef) + u["mwh"] * u["unit"].map(fleet["vom"])
                 + u["mwh"] * u["unit"].map(fleet["hr_fl"]) * other)
    g = u.groupby(["unit", "tech", "zone"], as_index=False)[["mwh", "energy_revenue", "cost"]].sum()
    cap = (fleet["pmax"] * fleet["count"]).rename("mw")
    g = g.merge(cap, left_on="unit", right_index=True)
    g["capacity_factor_pct"] = (100 * g["mwh"] / (g["mw"] * 8760)).round(1)
    g["energy_margin_kw_yr"] = ((g["energy_revenue"] - g["cost"]) / (g["mw"] * 1000)).round(1)
    g["realized_price"] = (g["energy_revenue"] / g["mwh"].replace(0, np.nan)).round(1)
    return g.sort_values("energy_margin_kw_yr", ascending=False)


def figures(all_runs: dict, summary: pd.DataFrame):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # 1. monthly LMP vs gas price, base
    base = all_runs.get("base_2027")
    if base is not None:
        h, d = base[0], base[1]
        m_lmp = h["lmp_NEMA"].resample("MS").mean()
        m_gas = d.set_index("date")["gas_price"].resample("MS").mean()
        m_hh = d.set_index("date")["hh"].resample("MS").mean()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(m_lmp.index.strftime("%b"), m_lmp.values, color="#3a7dc1", alpha=.8, label="NEMA LMP ($/MWh)")
        ax2 = ax.twinx()
        ax2.plot(m_gas.index.strftime("%b"), m_gas.values, "o-", color="#e07b39", label="Citygate gas ($/MMBtu)")
        ax2.plot(m_hh.index.strftime("%b"), m_hh.values, "--", color="#555", label="Henry Hub ($/MMBtu)")
        ax.set_ylabel("$/MWh"); ax2.set_ylabel("$/MMBtu")
        ax.set_title("2027 Base: monthly LMP tracks the endogenous citygate gas price")
        fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(.5, .02), frameon=False)
        fig.tight_layout(); fig.savefig(FIG / "base_monthly_lmp_gas.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    # 2. cold snap dispatch stack
    for nm in ["cold_snap", "cold_snap_lng_loss"]:
        if nm not in all_runs:
            continue
        h, d = all_runs[nm][0], all_runs[nm][1]
        win = h.loc["2027-01-16":"2027-02-01"]
        stack = pd.DataFrame({k: win[[c for c in v if c in win]].sum(axis=1)
                              for k, v in FUEL_GROUPS.items()})
        pos = stack.clip(lower=0)
        fig, (ax, axp) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]})
        ax.stackplot(pos.index, pos.T.values / 1e3, labels=pos.columns,
                     colors=[COLORS[c] for c in pos.columns], alpha=.92)
        ax.plot(win.index, win[[f"load_{z}" for z in ZONES]].sum(axis=1) / 1e3, color="k", lw=1.2,
                label="Load (net of BTM PV)")
        ax.set_ylabel("GW"); ax.set_ylim(0, 31)
        ax.legend(ncol=5, fontsize=8, loc="upper left", frameon=False)
        ax.set_title(f"{summary.set_index('scenario').loc[nm, 'label']}: hourly dispatch, Jan 16 - Feb 1")
        axp.plot(win.index, win["lmp_NEMA"], color="#3a7dc1", label="NEMA LMP $/MWh")
        dd = d.set_index("date")["gas_price"].loc["2027-01-16":"2027-02-01"]
        axp2 = axp.twinx()
        axp2.step(dd.index, dd.values, where="post", color="#e07b39", label="Gas $/MMBtu")
        axp.set_ylabel("NEMA LMP $/MWh", color="#3a7dc1"); axp2.set_ylabel("Gas $/MMBtu", color="#e07b39")
        import matplotlib.dates as mdates
        axp.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.tight_layout(); fig.savefig(FIG / f"{nm}_dispatch.png", dpi=160); plt.close(fig)
    # 3. gas balance during cold snap
    if "cold_snap" in all_runs:
        d = all_runs["cold_snap"][1].set_index("date").loc["2027-01-10":"2027-02-05"]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(d.index, d["ldc_bcf"], label="Firm LDC demand", color="#9aa5b1")
        ax.bar(d.index, d["power_burn_bcf"], bottom=d["ldc_bcf"], label="Power-sector burn", color="#e07b39")
        cap = all_runs["cold_snap"][2]
        ax.axhline(3.45, color="k", ls="--", lw=1, label="Firm pipeline capacity")
        ax.axhline(3.45 + 0.70, color="#7b5ea7", ls=":", lw=1, label="+ LNG send-out")
        ax.set_ylabel("Bcf/d"); ax.set_ylim(0, 5.4)
        ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left")
        ax.set_title("Cold snap: power burn is squeezed out as LDC demand fills the pipes")
        fig.tight_layout(); fig.savefig(FIG / "cold_snap_gas_balance.png", dpi=160); plt.close(fig)
    # 4. scenario comparison
    s = summary.set_index("label")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, col, ttl in zip(axes, ["load_weighted_lmp", "co2_mtons", "oil_gen_gwh"],
                            ["Load-weighted LMP ($/MWh)", "CO2 (million short tons)", "Oil-fired generation (GWh)"]):
        ax.barh(s.index, s[col], color="#3a7dc1")
        ax.set_title(ttl); ax.invert_yaxis()
        for i, v in enumerate(s[col]):
            ax.text(v, i, f" {v:,.1f}", va="center", fontsize=8)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout(); fig.savefig(FIG / "scenario_comparison.png", dpi=160); plt.close(fig)
    # 5. price duration curves
    fig, ax = plt.subplots(figsize=(8, 4))
    for nm, (h, *_rest) in all_runs.items():
        v = np.sort(h["lmp_NEMA"].values)[::-1]
        ax.plot(np.arange(len(v)), v, lw=1.2, label=summary.set_index("scenario").loc[nm, "label"])
    ax.set_yscale("symlog", linthresh=100); ax.set_xlabel("Hours (sorted)"); ax.set_ylabel("NEMA LMP $/MWh")
    ax.set_title("Price duration curves"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "price_duration.png", dpi=160); plt.close(fig)


def main():
    rows, runs = [], {}
    for f in sorted((ROOT / "scenarios").glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text())
        if not (RES / cfg["name"] / "daily.csv").exists():
            continue
        k, h, d, u = kpis(cfg["name"], cfg["label"])
        k.update(event_window(h, d))
        k["description"] = cfg.get("description", "")
        rows.append(k)
        m = unit_margins(u, d, cfg)
        m.to_csv(RES / cfg["name"] / "unit_margins.csv", index=False)
        runs[cfg["name"]] = (h, d, u)
    summary = pd.DataFrame(rows)
    summary.to_csv(RES / "summary.csv", index=False)
    (RES / "summary.json").write_text(json.dumps(rows, indent=1, default=float))
    figures(runs, summary)
    print(summary.T.to_string())
    return summary, runs


if __name__ == "__main__":
    main()
