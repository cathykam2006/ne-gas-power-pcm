"""Streamlit explorer for the New England gas-electric production cost model results.

Run locally:  streamlit run app.py   (after `python run_scenarios.py` and `python -m negem.report`)
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from negem.report import FUEL_GROUPS, ZONES  # noqa: E402

st.set_page_config(page_title="NE Gas-Electric PCM", layout="wide")
st.title("New England Gas-Electric Production Cost Model")
st.caption("Chronological unit commitment & dispatch for ISO-NE, co-optimized with the "
           "New England gas system. Open-source analogue of a PLEXOS/PROMOD study.")

summary = pd.read_csv(ROOT / "results" / "summary.csv")
labels = dict(zip(summary["label"], summary["scenario"]))
sel = st.sidebar.selectbox("Scenario", list(labels), index=1 if len(labels) > 1 else 0)
cmp_ = st.sidebar.selectbox("Compare against", list(labels), index=0)
name, ref = labels[sel], labels[cmp_]


@st.cache_data
def load(n):
    h = pd.read_csv(ROOT / "results" / n / "hourly.csv.gz", index_col=0, parse_dates=True)
    d = pd.read_csv(ROOT / "results" / n / "daily.csv", parse_dates=["date"])
    return h, d


h, d = load(name)
s = summary.set_index("scenario")
c = st.columns(5)
for col, (k, lab, fmt) in zip(c, [("load_weighted_lmp", "Load-weighted LMP", "${:,.2f}/MWh"),
                                  ("avg_gas_price", "Avg citygate gas", "${:,.2f}/MMBtu"),
                                  ("max_gas_price", "Peak-day gas", "${:,.2f}/MMBtu"),
                                  ("co2_mtons", "CO2", "{:,.2f} Mt"),
                                  ("oil_gen_gwh", "Oil generation", "{:,.0f} GWh")]):
    col.metric(lab, fmt.format(s.loc[name, k]),
               delta=None if name == ref else f"{s.loc[name, k] - s.loc[ref, k]:+,.2f} vs {cmp_}")

st.subheader("Prices")
m = pd.DataFrame({"NEMA LMP ($/MWh)": h["lmp_NEMA"].resample("D").mean(),
                  "Citygate gas x 10 ($/MMBtu)": d.set_index("date")["gas_price"] * 10})
st.line_chart(m)

st.subheader("Dispatch")
start, end = st.slider("Window", min_value=h.index.min().to_pydatetime(),
                       max_value=h.index.max().to_pydatetime(),
                       value=(pd.Timestamp("2027-01-16").to_pydatetime(),
                              pd.Timestamp("2027-01-30").to_pydatetime()))
w = h.loc[start:end]
stack = pd.DataFrame({k: w[[c for c in v if c in w]].sum(axis=1) for k, v in FUEL_GROUPS.items()})
st.area_chart(stack.clip(lower=0))

st.subheader("Gas balance (Bcf/d)")
st.bar_chart(d.set_index("date")[["ldc_bcf", "power_burn_bcf"]])

st.subheader("Zonal LMPs and congestion")
st.dataframe(pd.DataFrame({"Avg LMP": [h[f"lmp_{z}"].mean() for z in ZONES]}, index=ZONES)
             .style.format("${:.2f}"))
st.dataframe(summary.set_index("label").filter(like="bind_hours"))
st.subheader("All scenarios")
st.dataframe(summary.set_index("label").T)
