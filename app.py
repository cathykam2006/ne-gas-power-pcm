"""New England Gas-Electric Production Cost Model - Streamlit app.

Explore the pre-run scenarios or run the model live for a custom window.
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from negem.inputs import Scenario  # noqa: E402
from negem.report import FUEL_GROUPS, ZONES  # noqa: E402

RES = ROOT / "results"
REPO = "https://github.com/cathykam2006/ne-gas-power-pcm"
FUEL_COLORS = {"Nuclear": "#7b5ea7", "Natural gas": "#e07b39", "Oil": "#8c564b", "Hydro": "#3a7dc1",
               "Wind": "#4bb3a5", "Solar": "#f2c14e", "Biomass/Refuse": "#6b8e23",
               "Net imports": "#9aa5b1", "Storage (net)": "#d16ba5"}
PIPELINE_BCFD = 3.45
REPORTED_2025 = {"lmp_da": 71.81, "lmp_rt": 65.89, "gas": 6.27}  # ISO-NE IMM 2025 annual results

st.set_page_config(page_title="NE Gas-Electric Production Cost Model", page_icon="⚡", layout="wide")


# ======================================================================================
# Data
# ======================================================================================
@st.cache_data
def scenario_meta() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "scenarios").glob("*.yaml")):
        c = yaml.safe_load(f.read_text())
        rows.append({"scenario": c["name"], "label": c["label"], "description": c.get("description", ""),
                     "file": f.name})
    return pd.DataFrame(rows)


@st.cache_data
def summary() -> pd.DataFrame:
    s = pd.read_csv(RES / "summary.csv")
    order = list(scenario_meta()["scenario"])
    s["order"] = s["scenario"].map({k: i for i, k in enumerate(order)})
    return s.sort_values("order").reset_index(drop=True)


@st.cache_data
def hourly(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / name / "hourly.csv.gz", index_col=0, parse_dates=True)


@st.cache_data
def daily(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / name / "daily.csv", parse_dates=["date"])


@st.cache_data
def margins(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / name / "unit_margins.csv")


LIVE_RES = RES / "live"
LIVE_DATA = ROOT / "data" / "live"


def _stamp() -> float:
    """Changes whenever the daily job commits new live files, so cached loads refresh."""
    return max((p.stat().st_mtime for p in [LIVE_RES / "meta.json", LIVE_RES / "track_record.csv"] if p.exists()),
               default=0.0)


@st.cache_data
def live_files(stamp: float) -> dict | None:
    if not (LIVE_RES / "meta.json").exists():
        return None
    import json
    out = {"meta": json.loads((LIVE_RES / "meta.json").read_text()),
           "daily": pd.read_csv(LIVE_RES / "daily.csv", parse_dates=["date"]),
           "inp": pd.read_csv(LIVE_RES / "inputs_hourly.csv.gz", index_col=0, parse_dates=True)}
    tr = LIVE_RES / "track_record.csv"
    out["track"] = pd.read_csv(tr, parse_dates=["run_date", "target_date"]) if tr.exists() else pd.DataFrame()
    fm = LIVE_DATA / "isone_fuelmix.csv"
    out["fuelmix"] = pd.read_csv(fm, parse_dates=["time"]) if fm.exists() else None
    return out


def fuel_stack(h: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({k: h[[c for c in v if c in h]].sum(axis=1) for k, v in FUEL_GROUPS.items()}, index=h.index)


def load_weighted(h: pd.DataFrame) -> pd.Series:
    L = h[[f"load_{z}" for z in ZONES]].to_numpy()
    P = h[[f"lmp_{z}" for z in ZONES]].to_numpy()
    return pd.Series((L * P).sum(1) / L.sum(1), index=h.index)


S = summary()
META = scenario_meta()
LABEL = dict(zip(S["scenario"], S["label"]))
BY_LABEL = dict(zip(S["label"], S["scenario"]))
base = S.set_index("scenario").loc["base_2027"]
back = S.set_index("scenario").loc["backcast_2025"]


def delta(name, col):
    return float(S.set_index("scenario").loc[name, col] - base[col])


# ======================================================================================
# Header
# ======================================================================================
st.title("New England Gas-Electric Production Cost Model")
st.markdown(
    "Chronological **unit commitment and economic dispatch** for ISO New England, co-optimized with the "
    "**New England natural gas system** so the citygate gas price is solved inside the model. "
    "An open-source implementation of the simulation logic used in PLEXOS and PROMOD, built in Python "
    f"with the HiGHS solver. [Code and methodology on GitHub]({REPO})."
)

with st.sidebar:
    st.header("About the model")
    st.markdown(
        "- 365 daily MIP steps, 24 h committed + 12 h look-ahead\n"
        "- ~40 clustered unit groups: start costs, min up/down, ramps, heat-rate curves\n"
        "- Dual-fuel gas/oil with on-site oil inventory\n"
        "- Daily gas balance: firm LDC demand + power burn vs. pipeline, LNG and peak-shaving\n"
        "- 8 ISO-NE load zones, 4 interface limits, 30-min reserves\n"
        "- LMPs from an LP re-solve with fast-start pricing"
    )
    st.caption("Fleet aggregated from public plant data. Scenario tabs use a synthetic but calibrated weather "
               "year; the Live tab runs on real weather, load and prices, refreshed every morning.")
    st.markdown(f"[GitHub repository]({REPO})")
    st.caption("Built by Cathy Kam")

tabs = st.tabs(["Overview", "Live (real data)", "Scenarios", "Cold snap", "Prices & network", "Asset economics",
                "Run a scenario", "Methodology"])

# ======================================================================================
# Overview
# ======================================================================================
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Load-weighted LMP, 2027 base", f"${base.load_weighted_lmp:,.2f}/MWh",
                f"{base.load_weighted_lmp - back.load_weighted_lmp:+.2f} vs 2025 backcast", delta_color="inverse")
    c[1].metric("Citygate gas, 2027 base", f"${base.avg_gas_price:,.2f}/MMBtu",
                f"{base.avg_gas_price - back.avg_gas_price:+.2f} vs 2025", delta_color="inverse")
    c[2].metric("Winter gas (Dec-Feb)", f"${base.winter_gas_price:,.2f}/MMBtu")
    c[3].metric("Wholesale energy cost", f"${base.energy_cost_busd:,.2f}B")
    c[4].metric("Power-sector CO2", f"{base.co2_mtons:,.2f} Mt",
                f"{base.co2_mtons - back.co2_mtons:+.2f} vs 2025", delta_color="inverse")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Key findings")
        cs = S.set_index("scenario")
        st.markdown(
            f"1. **The binding constraint is the gas pipe, not the wires.** Electric interfaces bind for only "
            f"{int(cs.loc['base_2027'].filter(like='bind_hours').sum())} hours in the base year; on cold days "
            f"firm LDC demand fills the pipelines and generators become the marginal gas buyer.\n"
            f"2. **A 9-day cold snap more than doubles that week's energy cost** "
            f"(${cs.loc['cold_snap','event_energy_cost_musd']/1e3:,.2f}B vs "
            f"${cs.loc['base_2027','event_energy_cost_musd']/1e3:,.2f}B), with LMPs near "
            f"${cs.loc['cold_snap','event_lmp']:,.0f}/MWh and oil at "
            f"{cs.loc['cold_snap','event_oil_share_pct']:.0f}% of generation.\n"
            f"3. **Losing LNG barely moves price but raises oil dependence**: annual oil generation rises "
            f"{100*(cs.loc['cold_snap_lng_loss','oil_gen_gwh']/cs.loc['cold_snap','oil_gen_gwh']-1):.0f}% "
            f"because oil is already setting the price.\n"
            f"4. **+0.5 Bcf/d of pipeline** cuts the LMP by ${-delta('pipeline_expansion','load_weighted_lmp'):.2f}/MWh; "
            f"**2.4 GW offshore wind + 1 GW storage** cuts it by ${-delta('offshore_wind','load_weighted_lmp'):.2f}/MWh "
            f"and CO2 by {100*(1-cs.loc['offshore_wind','co2_mtons']/base.co2_mtons):.0f}%.\n"
            f"5. **Energy margins alone don't carry new gas plants**: combined cycles earn tens of $/kW-yr, "
            f"peakers close to zero, the 'missing money' capacity markets fill."
        )
    with right:
        st.subheader("Calibration: 2025 backcast")
        cal = pd.DataFrame({
            "Metric": ["Average energy price ($/MWh)", "Average gas price ($/MMBtu)", "Net energy for load (TWh)"],
            "Model": [back.load_weighted_lmp, back.avg_gas_price, back.load_twh],
            "ISO-NE reported": [REPORTED_2025["lmp_da"], REPORTED_2025["gas"], 118.0],
            "Note": [f"Day-ahead; real-time ${REPORTED_2025['lmp_rt']}", "New England average", "Implied by IMM $/MWh"],
        })
        st.dataframe(cal.style.format({"Model": "{:,.2f}", "ISO-NE reported": "{:,.2f}"}),
                     hide_index=True, width="stretch")
        st.caption("Reported: ISO-NE External Market Monitor 2025 annual results.")

    st.subheader("2027 base: monthly price follows the endogenous gas price")
    h, d = hourly("base_2027"), daily("base_2027")
    m = pd.DataFrame({"LMP ($/MWh)": load_weighted(h).resample("MS").mean(),
                      "Citygate gas ($/MMBtu)": d.set_index("date")["gas_price"].resample("MS").mean(),
                      "Henry Hub ($/MMBtu)": d.set_index("date")["hh"].resample("MS").mean()}).reset_index(names="month")
    bars = alt.Chart(m).mark_bar(color="#3a7dc1", opacity=.8, size=26).encode(
        x=alt.X("month:T", title=None, axis=alt.Axis(format="%b")),
        y=alt.Y("LMP ($/MWh):Q"), tooltip=[alt.Tooltip("month:T", format="%B"), alt.Tooltip("LMP ($/MWh):Q", format=".1f")])
    gl = m.melt("month", ["Citygate gas ($/MMBtu)", "Henry Hub ($/MMBtu)"], "series", "price")
    lines = alt.Chart(gl).mark_line(point=True).encode(
        x="month:T", y=alt.Y("price:Q", title="$/MMBtu"),
        color=alt.Color("series:N", scale=alt.Scale(range=["#e07b39", "#555"]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["series", alt.Tooltip("price:Q", format=".2f")])
    st.altair_chart(alt.layer(bars, lines).resolve_scale(y="independent").properties(height=320), width="stretch")

# ======================================================================================
# Live (real data)
# ======================================================================================
with tabs[1]:
    L = live_files(_stamp())
    if L is None:
        st.info("The live run has not produced results yet. It runs every morning via GitHub Actions "
                "(workflow: Live model daily refresh).")
    else:
        m, dl, inp, tr = L["meta"], L["daily"], L["inp"], L["track"]
        run_day = pd.Timestamp(m["run_date"])
        st.subheader("The model on real data, re-solved every morning")
        st.markdown(
            "Each morning the model pulls **real New England weather** (actual and forecast), **ISO-NE load and "
            "day-ahead prices**, and **Henry Hub and oil prices**, then re-solves unit commitment, dispatch and the "
            "gas market for the **past 7 days** (scored against ISO-NE's published day-ahead prices) and the "
            f"**next 7 days** (forecast). Last run: {m['run_at']}.")
        mt = m.get("metrics", {})
        fc = dl[dl["kind"] == "forecast"]
        tmr = fc[fc["date"] == run_day + pd.Timedelta(days=1)]
        last_act = dl.dropna(subset=["lmp_hub_actual"])
        k = st.columns(4)
        if len(tmr):
            r = tmr.iloc[0]
            k[0].metric(f"Model forecast, {r.date:%a %b %-d}", f"${r.lmp_hub_model:,.2f}/MWh",
                        help="Internal Hub day-ahead price, daily average (load-weighted across the 8 zones).")
        if len(last_act):
            r = last_act.iloc[-1]
            k[1].metric(f"ISO-NE actual, {r.date:%a %b %-d}", f"${r.lmp_hub_actual:,.2f}/MWh",
                        delta=f"model {r.lmp_hub_model - r.lmp_hub_actual:+.2f}", delta_color="off")
        if "daily_mae" in mt:
            k[2].metric("Past 7 days: avg daily error", f"${mt['daily_mae']:,.2f}/MWh",
                        help=f"Mean absolute error of the model's daily hub price vs. ISO-NE day-ahead "
                             f"(actual average ${mt['actual_avg']:.2f}/MWh). Hourly error ${mt['hourly_mae']:.2f}.")
        if len(tmr):
            r = tmr.iloc[0]
            k[3].metric(f"Model citygate gas, {r.date:%a %b %-d} (Henry Hub {r.hh:,.2f})", f"${r.gas_price:,.2f}/MMBtu",
                        help="New England citygate gas price the model clears, from the Henry Hub input "
                             "(NYMEX front month) plus pipeline basis, LNG and peak-shaving supply.")

        # ---- hourly hub price: model vs ISO-NE day-ahead, with the forecast zone shaded
        st.markdown("**Internal Hub price, hourly: model vs. ISO-NE day-ahead**")
        ph = inp[["lmp_hub_model"] + (["da_lmp_HUB"] if "da_lmp_HUB" in inp else [])].copy()
        ph.columns = ["Model"] + (["ISO-NE day-ahead (actual)"] if "da_lmp_HUB" in inp else [])
        pl = ph.reset_index(names="time").melt("time", var_name="series", value_name="$/MWh").dropna()
        zone = pd.DataFrame({"a": [run_day], "b": [inp.index.max()]})
        shade = alt.Chart(zone).mark_rect(color="#3a7dc1", opacity=0.07).encode(x="a:T", x2="b:T")
        split = alt.Chart(zone).mark_rule(strokeDash=[3, 3], color="#555").encode(x="a:T")
        lab = alt.Chart(pd.DataFrame({"t": [run_day], "txt": ["  FORECAST →"]})).mark_text(
            align="left", baseline="top", dy=4, color="#3a7dc1", fontWeight="bold").encode(x="t:T", y=alt.value(4), text="txt:N")
        lines = alt.Chart(pl).mark_line(strokeWidth=2).encode(
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%a %b %d", labelAngle=0, tickCount=7)),
            y=alt.Y("$/MWh:Q", title="$/MWh"),
            color=alt.Color("series:N", scale=alt.Scale(domain=["ISO-NE day-ahead (actual)", "Model"], range=["#222", "#3a7dc1"]),
                            legend=alt.Legend(orient="top", title=None)),
            strokeDash=alt.condition(alt.datum.series == "Model", alt.value([6, 3]), alt.value([1, 0])),
            tooltip=[alt.Tooltip("time:T", format="%a %b %d %H:00"), "series", alt.Tooltip("$/MWh:Q", format=".2f")])
        st.altair_chart((shade + split + lines + lab).properties(height=320), width="stretch")
        st.caption("Shaded: the next 7 days, a forecast from forecast weather and the NYMEX Henry Hub price. "
                   "Before it: the model re-run on actual weather, load and gas prices, next to ISO-NE's published "
                   "day-ahead Internal Hub price.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Daily: model vs. actual hub price**")
            dd = dl.melt("date", ["lmp_hub_model", "lmp_hub_actual"], "series", "$/MWh").dropna()
            dd["series"] = dd["series"].map({"lmp_hub_model": "Model", "lmp_hub_actual": "ISO-NE day-ahead (actual)"})
            bars = alt.Chart(dd).mark_bar().encode(
                x=alt.X("yearmonthdate(date):O", title=None, axis=alt.Axis(format="%b %d", labelAngle=-45)),
                xOffset="series:N", y=alt.Y("$/MWh:Q"),
                color=alt.Color("series:N", scale=alt.Scale(domain=["ISO-NE day-ahead (actual)", "Model"], range=["#222", "#3a7dc1"]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("date:T", format="%a %b %d"), "series", alt.Tooltip("$/MWh:Q", format=".2f")])
            st.altair_chart(bars.properties(height=260), width="stretch")
        with c2:
            st.markdown("**Temperature and load**")
            ld = inp[["load_model_mw"] + (["load_actual_mw"] if inp["load_actual_mw"].notna().any() else [])] / 1e3
            ld.columns = ["Model (from temperature)"] + (["ISO-NE actual (EIA-930)"] if ld.shape[1] > 1 else [])
            ll = ld.reset_index(names="time").melt("time", var_name="series", value_name="GW").dropna()
            lc = alt.Chart(ll).mark_line(strokeWidth=1.8).encode(
                x=alt.X("time:T", title=None, axis=alt.Axis(format="%b %d", labelAngle=0)), y=alt.Y("GW:Q", title="Load (GW)"),
                color=alt.Color("series:N", scale=alt.Scale(domain=["ISO-NE actual (EIA-930)", "Model (from temperature)"],
                                                            range=["#222", "#3a7dc1"]), legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("time:T", format="%a %b %d %H:00"), "series", alt.Tooltip("GW:Q", format=".2f")])
            tc = alt.Chart(inp.reset_index(names="time")).mark_line(color="#e07b39", strokeWidth=1.2, opacity=0.8).encode(
                x="time:T", y=alt.Y("temp_f:Q", title="Temperature (°F, orange)"),
                tooltip=[alt.Tooltip("time:T", format="%a %b %d %H:00"), alt.Tooltip("temp_f:Q", title="°F", format=".0f")])
            st.altair_chart(alt.layer(lc, tc).resolve_scale(y="independent").properties(height=260), width="stretch")
            if "load_mape_pct" in mt:
                st.caption(f"Model load vs. actual over the past 7 days: {mt['load_mape_pct']:.1f}% average error "
                           f"(scaled by {m['load_ratio']:.3f} to match the last 30 days of actual demand).")
            else:
                st.caption("Actual ISO-NE load appears once the EIA API key is added to the repository.")

        st.markdown("**Day-ahead track record**: forecasts saved each morning and scored when ISO-NE publishes the price")
        if len(tr) and "actual_hub" in tr and tr["actual_hub"].notna().any():
            t1 = tr[(tr["horizon_days"] == 1)].dropna(subset=["actual_hub"]).sort_values("target_date", ascending=False)
            t1 = t1.assign(error=(t1["model_hub"] - t1["actual_hub"]).round(2))
            st.dataframe(t1[["target_date", "run_date", "model_hub", "actual_hub", "error", "model_gas", "hh"]].rename(columns={
                "target_date": "Delivery day", "run_date": "Forecast made", "model_hub": "Model hub ($/MWh)",
                "actual_hub": "ISO-NE day-ahead ($/MWh)", "error": "Error", "model_gas": "Model citygate gas ($/MMBtu)",
                "hh": "Henry Hub input"}), hide_index=True, width="stretch",
                column_config={"Delivery day": st.column_config.DateColumn(format="ddd MMM D"),
                               "Forecast made": st.column_config.DateColumn(format="MMM D")})
            if "track_day_ahead_mae" in mt:
                n_ = mt["track_days_scored"]
                st.caption(f"{n_} day-ahead forecast{'s' if n_ != 1 else ''} scored so far; average error "
                           f"${mt['track_day_ahead_mae']:.2f}/MWh. The record grows by one day each morning.")
        else:
            st.caption("The first forecasts are scored the day after they are made.")

        st.markdown("**Data sources**")
        use = {"weather (Open-Meteo)": "Temperature (load, gas demand), solar radiation and 100 m wind speed",
               "ISO-NE day-ahead LMPs": "Actual prices the model is scored against",
               "NYMEX Henry Hub futures": "Henry Hub gas price for forecast days",
               "ISO-NE load and fuel mix (EIA-930)": "Actual load for past days; calibrates the load forecast",
               "Henry Hub spot (EIA)": "Henry Hub gas price for past days",
               "Heating oil / ULSD spot (EIA)": "Oil-unit fuel cost"}
        rows = [{"Source": k_, "Status": "✅ updated" if v.get("ok") else ("⚠️ " + v.get("error", "failed")),
                 "Last successful update": v.get("last_ok") or "–", "Used for": use.get(k_, "")}
                for k_, v in m.get("sources", {}).items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Limits: the fleet is aggregated and nuclear refueling outages are not tracked live; New England "
                   "citygate (Algonquin) gas prices are not freely published, so the model's gas price is not scored. "
                   "Hourly price shape is less accurate than the daily level.")

# ======================================================================================
# Scenarios
# ======================================================================================
with tabs[2]:
    st.subheader("Scenario comparison against the 2027 base")
    with st.expander("Scenario definitions"):
        st.dataframe(META[["label", "description", "file"]], hide_index=True, width="stretch")
    metrics = {"Load-weighted LMP ($/MWh)": "load_weighted_lmp", "Winter LMP ($/MWh)": "winter_lmp",
               "Avg citygate gas ($/MMBtu)": "avg_gas_price", "Winter gas ($/MMBtu)": "winter_gas_price",
               "Energy cost ($B)": "energy_cost_busd", "CO2 (Mt)": "co2_mtons", "Oil generation (GWh)": "oil_gen_gwh",
               "Power burn (Bcf)": "power_burn_bcf", "LNG send-out (Bcf)": "lng_bcf"}
    mlabel = st.selectbox("Metric", list(metrics), index=0)
    col = metrics[mlabel]
    dfm = S[["label", "scenario", col]].copy()
    dfm["vs base"] = dfm[col] - base[col]
    dfm["is_base"] = dfm["scenario"].eq("base_2027")
    ch = alt.Chart(dfm).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("label:N", sort=list(S["label"]), title=None),
        x=alt.X(f"{col}:Q", title=mlabel),
        color=alt.condition("datum.is_base", alt.value("#3a7dc1"), alt.value("#b8c2cc")),
        tooltip=["label", alt.Tooltip(f"{col}:Q", format=",.2f"), alt.Tooltip("vs base:Q", format="+,.2f")])
    txt = ch.mark_text(align="left", dx=4, fontSize=11).encode(text=alt.Text(f"{col}:Q", format=",.2f"), color=alt.value("#555"))
    st.altair_chart((ch + txt).properties(height=300), width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("**Generation mix (TWh)**")
        mix_cols = [c for c in S.columns if c.startswith("mix_")]
        mx = S[["label"] + mix_cols].melt("label", var_name="fuel", value_name="TWh")
        mx["fuel"] = mx["fuel"].str.replace("mix_", "")
        mx = mx[mx["TWh"] > 0]
        st.altair_chart(alt.Chart(mx).mark_bar().encode(
            y=alt.Y("label:N", sort=list(S["label"]), title=None), x=alt.X("TWh:Q", stack="zero"),
            color=alt.Color("fuel:N", scale=alt.Scale(domain=list(FUEL_COLORS), range=list(FUEL_COLORS.values())),
                            legend=alt.Legend(orient="bottom", columns=4, title=None)),
            tooltip=["label", "fuel", alt.Tooltip("TWh:Q", format=".1f")]).properties(height=320), width="stretch")
    with b:
        st.markdown("**Change vs base**")
        tbl = S[["label", "load_weighted_lmp", "avg_gas_price", "energy_cost_busd", "co2_mtons", "oil_gen_gwh"]].copy()
        for c_ in tbl.columns[1:]:
            tbl[c_] = tbl[c_] - base[c_]
        tbl.columns = ["Scenario", "LMP $/MWh", "Gas $/MMBtu", "Energy $B", "CO2 Mt", "Oil GWh"]
        st.dataframe(tbl.style.format({c_: "{:+,.2f}" for c_ in tbl.columns[1:]})
                     .background_gradient(cmap="RdYlGn_r", subset=["LMP $/MWh", "Energy $B"], vmin=-10, vmax=10),
                     hide_index=True, width="stretch")

    st.markdown("**All results**")
    show = ["label", "load_weighted_lmp", "winter_lmp", "avg_gas_price", "winter_gas_price", "max_gas_price",
            "energy_cost_busd", "co2_mtons", "oil_gen_gwh", "power_burn_bcf", "lng_bcf", "min_oil_inventory_pct",
            "unserved_mwh", "event_lmp", "event_gas_price", "event_oil_share_pct", "event_energy_cost_musd"]
    st.dataframe(S[show].set_index("label"), width="stretch")
    st.download_button("Download summary CSV", S.to_csv(index=False), "summary.csv", "text/csv")

# ======================================================================================
# Cold snap
# ======================================================================================
with tabs[3]:
    st.subheader("Inside a winter week")
    cc = st.columns([1.3, 1])
    sc_label = cc[0].selectbox("Scenario", list(S["label"]), index=list(S["scenario"]).index("cold_snap_lng_loss"), key="cs_sc")
    sc = BY_LABEL[sc_label]
    h, d = hourly(sc), daily(sc)
    start, end = cc[1].slider("Window", min_value=h.index.min().date(), max_value=h.index.max().date(),
                              value=(pd.Timestamp("2027-01-16").date(), pd.Timestamp("2027-02-01").date()), key="cs_win")
    w = h.loc[str(start):str(end)]
    dw = d[(d.date.dt.date >= start) & (d.date.dt.date <= end)]

    k = st.columns(5)
    Lw = w[[f"load_{z}" for z in ZONES]].to_numpy()
    lw_price = (Lw * w[[f"lmp_{z}" for z in ZONES]].to_numpy()).sum() / Lw.sum()
    gen = w.filter(like="gen_").clip(lower=0).sum().sum()
    k[0].metric("Load-weighted LMP", f"${lw_price:,.1f}/MWh")
    k[1].metric("Peak NEMA LMP", f"${w['lmp_NEMA'].max():,.0f}/MWh")
    k[2].metric("Avg citygate gas", f"${dw['gas_price'].mean():,.2f}/MMBtu")
    k[3].metric("Oil share of generation", f"{100*w['gen_Oil'].sum()/gen:.1f}%")
    k[4].metric("Energy cost", f"${(Lw * w[[f'lmp_{z}' for z in ZONES]].to_numpy()).sum()/1e6:,.0f}M")

    stack = fuel_stack(w).clip(lower=0) / 1e3
    sl = stack.reset_index(names="time").melt("time", var_name="fuel", value_name="GW")
    load = (w[[f"load_{z}" for z in ZONES]].sum(axis=1) / 1e3).rename("GW").rename_axis("time").reset_index()
    area = alt.Chart(sl).mark_area(opacity=.9).encode(
        x=alt.X("time:T", title=None), y=alt.Y("GW:Q", stack="zero", title="Generation (GW)"),
        color=alt.Color("fuel:N", scale=alt.Scale(domain=list(FUEL_COLORS), range=list(FUEL_COLORS.values())),
                        legend=alt.Legend(orient="top", columns=9, title=None)),
        order=alt.Order("fuel_order:Q"), tooltip=[alt.Tooltip("time:T", format="%b %d %H:00"), "fuel", alt.Tooltip("GW:Q", format=".2f")]
    ).transform_calculate(fuel_order=f"indexof({list(FUEL_COLORS)}, datum.fuel)")
    lline = alt.Chart(load).mark_line(color="black", strokeWidth=1.4).encode(x="time:T", y="GW:Q",
            tooltip=[alt.Tooltip("time:T", format="%b %d %H:00"), alt.Tooltip("GW:Q", title="Load GW", format=".2f")])
    st.altair_chart((area + lline).properties(height=360), width="stretch")

    pr = pd.DataFrame({"time": w.index, "NEMA LMP ($/MWh)": w["lmp_NEMA"].values})
    gp = dw[["date", "gas_price"]].rename(columns={"date": "time", "gas_price": "Citygate gas ($/MMBtu)"})
    p1 = alt.Chart(pr).mark_line(color="#3a7dc1", strokeWidth=2).encode(x=alt.X("time:T", title=None), y="NEMA LMP ($/MWh):Q",
            tooltip=[alt.Tooltip("time:T", format="%b %d %H:00"), alt.Tooltip("NEMA LMP ($/MWh):Q", format=".1f")])
    p2 = alt.Chart(gp).mark_line(color="#e07b39", interpolate="step-after", strokeWidth=2).encode(x="time:T", y="Citygate gas ($/MMBtu):Q",
            tooltip=[alt.Tooltip("time:T", format="%b %d"), alt.Tooltip("Citygate gas ($/MMBtu):Q", format=".2f")])
    st.altair_chart(alt.layer(p1, p2).resolve_scale(y="independent").properties(height=200), width="stretch")

    st.markdown("**New England gas balance (Bcf/d)**: firm LDC demand fills the pipes first; power plants get what is left.")
    gb = dw.melt("date", ["ldc_bcf", "power_burn_bcf"], "use", "Bcf/d").replace({"ldc_bcf": "Firm LDC demand", "power_burn_bcf": "Power-sector burn"})
    gbar = alt.Chart(gb).mark_bar().encode(
        x=alt.X("yearmonthdate(date):O", title=None, axis=alt.Axis(format="%b %d", labelAngle=-45)),
        y=alt.Y("Bcf/d:Q", stack="zero"),
        color=alt.Color("use:N", scale=alt.Scale(range=["#9aa5b1", "#e07b39"]), legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("use:N", sort="descending"),
        tooltip=[alt.Tooltip("date:T", format="%b %d"), "use", alt.Tooltip("Bcf/d:Q", format=".2f")])
    rule = alt.Chart(pd.DataFrame({"y": [PIPELINE_BCFD]})).mark_rule(strokeDash=[5, 4], color="black").encode(y="y:Q")
    st.altair_chart((gbar + rule).properties(height=260), width="stretch")
    st.caption(f"Dashed line: {PIPELINE_BCFD} Bcf/d firm pipeline capacity (scenario input; the pipeline scenario raises it).")

# ======================================================================================
# Prices & network
# ======================================================================================
with tabs[4]:
    st.subheader("Prices")
    pick = st.multiselect("Scenarios", list(S["label"]),
                          default=["2025 backcast (calibration)", "2027 Base", "Cold snap + LNG import loss", "Pipeline +0.5 Bcf/d"])
    if pick:
        a, b = st.columns(2)
        mon, pdc = [], []
        for lab in pick:
            hh = hourly(BY_LABEL[lab])
            lwp = load_weighted(hh)
            mon.append(pd.DataFrame({"month": lwp.resample("MS").mean().index, "LMP": lwp.resample("MS").mean().values, "scenario": lab}))
            v = np.sort(hh["lmp_NEMA"].values)[::-1]
            idx = np.unique(np.linspace(0, len(v) - 1, 400).astype(int))
            pdc.append(pd.DataFrame({"hours": idx, "LMP": v[idx], "scenario": lab}))
        with a:
            st.markdown("**Monthly load-weighted LMP ($/MWh)**")
            st.altair_chart(alt.Chart(pd.concat(mon)).mark_line(point=True).encode(
                x=alt.X("month:T", title=None, axis=alt.Axis(format="%b")), y=alt.Y("LMP:Q", title="$/MWh"),
                color=alt.Color("scenario:N", legend=alt.Legend(orient="bottom", columns=2, title=None)),
                tooltip=["scenario", alt.Tooltip("month:T", format="%B"), alt.Tooltip("LMP:Q", format=".1f")]
            ).properties(height=320), width="stretch")
        with b:
            st.markdown("**Price duration curve, NEMA (log scale)**")
            st.altair_chart(alt.Chart(pd.concat(pdc)).mark_line().encode(
                x=alt.X("hours:Q", title="Hours, sorted high to low"),
                y=alt.Y("LMP:Q", title="$/MWh", scale=alt.Scale(type="symlog")),
                color=alt.Color("scenario:N", legend=alt.Legend(orient="bottom", columns=2, title=None)),
                tooltip=["scenario", "hours", alt.Tooltip("LMP:Q", format=".1f")]).properties(height=320), width="stretch")

    st.subheader("Zonal prices and interface congestion")
    zl = st.selectbox("Scenario", list(S["label"]), index=1, key="zone_sc")
    hz = hourly(BY_LABEL[zl])
    a, b = st.columns(2)
    with a:
        zt = pd.DataFrame({"Zone": ZONES, "Avg LMP ($/MWh)": [hz[f"lmp_{z}"].mean() for z in ZONES],
                           "Max LMP": [hz[f"lmp_{z}"].max() for z in ZONES],
                           "Load (TWh)": [hz[f"load_{z}"].sum() / 1e6 for z in ZONES]})
        st.dataframe(zt.style.format({"Avg LMP ($/MWh)": "{:.2f}", "Max LMP": "{:.0f}", "Load (TWh)": "{:.1f}"}),
                     hide_index=True, width="stretch")
    with b:
        it = pd.DataFrame({"Interface": [c.replace("ifprice_", "") for c in hz.filter(like="ifprice_").columns],
                           "Binding hours": [(hz[c] > 0.01).sum() for c in hz.filter(like="ifprice_").columns],
                           "Avg shadow price ($/MWh)": [hz[c].mean() for c in hz.filter(like="ifprice_").columns]})
        st.dataframe(it.style.format({"Avg shadow price ($/MWh)": "{:.2f}"}), hide_index=True, width="stretch")
        st.caption("New England's bulk electric system is lightly congested in the model, consistent with "
                   "ISO-NE's history; scarcity shows up in the gas system instead.")
    flows = hz.filter(like="flow_")
    fl = pd.DataFrame({"Link": [c.replace("flow_", "") for c in flows.columns],
                       "Avg flow (MW)": flows.mean().values, "Max |flow| (MW)": flows.abs().max().values})
    with st.expander("Average transmission flows"):
        st.dataframe(fl.style.format({"Avg flow (MW)": "{:,.0f}", "Max |flow| (MW)": "{:,.0f}"}), hide_index=True, width="stretch")

# ======================================================================================
# Asset economics
# ======================================================================================
with tabs[5]:
    st.subheader("Unit energy margins")
    st.markdown("Energy revenue at zonal LMPs minus fuel, CO2 and variable O&M, in $/kW-yr: the "
                "energy-only view used in asset valuation. Capacity and ancillary revenues are not included.")
    al = st.selectbox("Scenario", list(S["label"]), index=1, key="asset_sc")
    mg = margins(BY_LABEL[al])
    techs = st.multiselect("Technology", sorted(mg["tech"].unique()), default=[t for t in ["CC", "CT", "ST", "NUC"] if t in mg["tech"].unique()])
    mg = mg[mg["tech"].isin(techs)]
    a, b = st.columns([1, 1.1])
    with a:
        st.altair_chart(alt.Chart(mg).mark_circle(size=90, opacity=.85).encode(
            x=alt.X("capacity_factor_pct:Q", title="Capacity factor (%)"),
            y=alt.Y("energy_margin_kw_yr:Q", title="Energy margin ($/kW-yr)", scale=alt.Scale(type="symlog")),
            color=alt.Color("tech:N", legend=alt.Legend(orient="top", title=None)),
            tooltip=["unit", "tech", "zone", alt.Tooltip("mw:Q", format=","), alt.Tooltip("capacity_factor_pct:Q", format=".1f"),
                     alt.Tooltip("realized_price:Q", format=".1f"), alt.Tooltip("energy_margin_kw_yr:Q", format=".1f")]
        ).properties(height=380), width="stretch")
    with b:
        st.dataframe(mg[["unit", "tech", "zone", "mw", "capacity_factor_pct", "realized_price", "energy_margin_kw_yr"]]
                     .rename(columns={"unit": "Unit", "tech": "Type", "zone": "Zone", "mw": "MW", "capacity_factor_pct": "CF %",
                                      "realized_price": "Realized $/MWh", "energy_margin_kw_yr": "Margin $/kW-yr"}),
                     hide_index=True, width="stretch", height=380)

    st.markdown("**How scenarios move one asset's margin**")
    unit = st.selectbox("Unit", list(margins("base_2027").query("tech in ['CC','CT','ST']")["unit"]), index=0)
    um = []
    for nm in S["scenario"]:
        r = margins(nm).set_index("unit")
        if unit in r.index:
            um.append({"Scenario": LABEL[nm], "Margin $/kW-yr": r.loc[unit, "energy_margin_kw_yr"],
                       "CF %": r.loc[unit, "capacity_factor_pct"]})
    um = pd.DataFrame(um)
    st.altair_chart(alt.Chart(um).mark_bar(cornerRadiusEnd=3, color="#e07b39").encode(
        y=alt.Y("Scenario:N", sort=list(S["label"]), title=None), x="Margin $/kW-yr:Q",
        tooltip=["Scenario", alt.Tooltip("Margin $/kW-yr:Q", format=".1f"), alt.Tooltip("CF %:Q", format=".1f")]
    ).properties(height=260), width="stretch")

# ======================================================================================
# Run a scenario (live model)
# ======================================================================================
@st.cache_data(show_spinner=False, max_entries=24)
def run_window(params: dict, start_day: int, days: int):
    from negem.simulate import run
    sc = Scenario(**params)
    out = run(sc, days=days, start_day=start_day, verbose=False)
    h, d = out["hourly"], out["daily"]
    L = h[[f"load_{z}" for z in ZONES]].to_numpy()
    P = h[[f"lmp_{z}" for z in ZONES]].to_numpy()
    gen = h.filter(like="gen_").clip(lower=0).sum().sum()
    kp = {"Load-weighted LMP ($/MWh)": (L * P).sum() / L.sum(),
          "Peak LMP ($/MWh)": h["lmp_NEMA"].max(),
          "Avg citygate gas ($/MMBtu)": d["gas_price"].mean(),
          "Energy cost ($M)": (L * P).sum() / 1e6,
          "Oil share (%)": 100 * h["gen_Oil"].sum() / gen,
          "Power burn (Bcf)": d["power_burn_bcf"].sum(),
          "CO2 (kt)": h["co2_tons"].sum() / 1e3,
          "Unserved energy (MWh)": h.filter(like="uns_").sum().sum()}
    return kp, h[["lmp_NEMA"]].assign(**{k: fuel_stack(h)[k] for k in FUEL_GROUPS}), d


with tabs[6]:
    st.subheader("Run the model live")
    st.markdown("Set the levers, pick a window, and the app re-solves the rolling unit commitment and gas balance "
                "for those days, then compares the result with the 2027 base over the same window and weather. "
                "Each day is a mixed-integer solve, so a week takes roughly 20-60 seconds.")
    base_sc = Scenario.from_yaml(ROOT / "scenarios" / "02_base_2027.yaml")
    with st.form("custom"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Fuel & carbon**")
            hh_shift = st.slider("Henry Hub shift ($/MMBtu, all months)", -2.0, 4.0, 0.0, 0.25)
            oil = st.slider("Delivered oil price ($/MMBtu)", 12.0, 30.0, base_sc.oil_price, 0.5)
            co2 = st.slider("CO2 price ($/short ton)", 0.0, 100.0, base_sc.co2_price, 1.0)
        with c2:
            st.markdown("**Gas infrastructure**")
            pipe = st.slider("Firm pipeline into New England (Bcf/d)", 2.8, 4.6, base_sc.pipeline_bcfd, 0.05)
            lng = st.slider("LNG send-out capacity (Bcf/d)", 0.0, 1.2, base_sc.lng_bcfd, 0.05)
            lngp = st.slider("Delivered LNG price ($/MMBtu)", 8.0, 30.0, base_sc.lng_price, 0.5)
        with c3:
            st.markdown("**Supply & demand**")
            osw = st.slider("Added offshore wind (MW)", 0, 6000, 0, 200)
            batt = st.slider("Added 4-hour batteries (MW)", 0, 3000, 0, 100)
            growth = st.slider("Load growth vs 2025 (x)", 0.95, 1.20, base_sc.load_growth, 0.005)
            hq = st.slider("Quebec import availability (x)", 0.3, 1.3, base_sc.hq_mult, 0.05)
            cold = st.checkbox("Weather: late-January arctic cold snap (applied to both runs)", value=True)
        w1, w2 = st.columns(2)
        start_date = w1.date_input("Window start", pd.Timestamp("2027-01-18").date(),
                                   min_value=pd.Timestamp("2027-01-01").date(), max_value=pd.Timestamp("2027-12-25").date())
        ndays = w2.slider("Days to simulate", 1, 10, 7)
        go = st.form_submit_button("Run model", type="primary")

    if go:
        start_day = (pd.Timestamp(start_date) - pd.Timestamp("2027-01-01")).days
        ndays = min(ndays, 365 - start_day)
        bp = dict(base_sc.__dict__, cold_snap=cold)
        cp = dict(bp, name="custom", label="Custom",
                  hh_monthly=[max(0.5, x + hh_shift) for x in base_sc.hh_monthly], oil_price=oil, co2_price=co2,
                  pipeline_bcfd=pipe, lng_bcfd=lng, lng_price=lngp, extra_offshore_mw=float(osw),
                  extra_battery_mw=float(batt), load_growth=growth, hq_mult=hq)
        with st.spinner(f"Solving {ndays} days of unit commitment for the base case..."):
            kb, hb, db = run_window(bp, start_day, ndays)
        with st.spinner(f"Solving {ndays} days for your scenario..."):
            kc, hc, dc = run_window(cp, start_day, ndays)
        st.session_state["custom_result"] = (kb, hb, db, kc, hc, dc)
        st.caption("Each window starts from a warm-start fleet state, so the first day can differ slightly "
                   "from the full-year run.")

    if "custom_result" in st.session_state:
        kb, hb, db, kc, hc, dc = st.session_state["custom_result"]
        cols = st.columns(4)
        for i, k_ in enumerate(kb):
            better_low = k_ != "Power burn (Bcf)"
            cols[i % 4].metric(k_, f"{kc[k_]:,.2f}", f"{kc[k_]-kb[k_]:+,.2f} vs base",
                               delta_color="inverse" if better_low else "off")
        cmp = pd.concat([hb["lmp_NEMA"].rename("2027 Base"), hc["lmp_NEMA"].rename("Your scenario")], axis=1).reset_index(names="time")
        st.altair_chart(alt.Chart(cmp.melt("time", var_name="case", value_name="LMP")).mark_line(strokeWidth=2).encode(
            x=alt.X("time:T", title=None), y=alt.Y("LMP:Q", title="NEMA LMP ($/MWh)"),
            color=alt.Color("case:N", scale=alt.Scale(range=["#9aa5b1", "#3a7dc1"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["case", alt.Tooltip("time:T", format="%b %d %H:00"), alt.Tooltip("LMP:Q", format=".1f")]
        ).properties(height=280), width="stretch")
        stk = (hc[list(FUEL_GROUPS)].clip(lower=0) / 1e3).reset_index(names="time").melt("time", var_name="fuel", value_name="GW")
        st.markdown("**Your scenario: dispatch**")
        st.altair_chart(alt.Chart(stk).mark_area(opacity=.9).encode(
            x=alt.X("time:T", title=None), y=alt.Y("GW:Q", stack="zero"),
            color=alt.Color("fuel:N", scale=alt.Scale(domain=list(FUEL_COLORS), range=list(FUEL_COLORS.values())),
                            legend=alt.Legend(orient="top", columns=9, title=None)),
            tooltip=[alt.Tooltip("time:T", format="%b %d %H:00"), "fuel", alt.Tooltip("GW:Q", format=".2f")]
        ).properties(height=300), width="stretch")
        gd = pd.concat([db.assign(case="2027 Base"), dc.assign(case="Your scenario")])
        st.markdown("**Daily citygate gas price**")
        st.altair_chart(alt.Chart(gd).mark_line(point=True).encode(
            x=alt.X("date:T", title=None), y=alt.Y("gas_price:Q", title="$/MMBtu"),
            color=alt.Color("case:N", scale=alt.Scale(range=["#9aa5b1", "#e07b39"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["case", alt.Tooltip("date:T", format="%b %d"), alt.Tooltip("gas_price:Q", format=".2f"),
                     alt.Tooltip("power_burn_bcf:Q", format=".2f"), alt.Tooltip("lng_bcf:Q", format=".2f")]
        ).properties(height=240), width="stretch")

# ======================================================================================
# Methodology
# ======================================================================================
with tabs[7]:
    st.markdown((ROOT / "docs" / "methodology.md").read_text())
