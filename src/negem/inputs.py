"""Input preparation: fleet, network, weather-driven load, renewables and the gas supply curve.

All time series are hourly for a non-leap 8,760-hour study year. Weather is synthetic but
seeded and shaped to Boston climatology so every scenario sees the *same* weather year
(only the scenario levers change). Swap `build_weather()` for real ISO-NE / NOAA data
without touching the optimization code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DATA = Path(__file__).resolve().parents[2] / "data"
HOURS = 8760
MMBTU_PER_BCF = 1.037e6


# --------------------------------------------------------------------------------------
# Scenario definition
# --------------------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    label: str
    description: str = ""
    hh_monthly: list = field(default_factory=lambda: [4.9, 4.6, 3.9, 3.5, 3.4, 3.5,
                                                       3.8, 3.8, 3.6, 3.7, 4.2, 4.8])
    oil_price: float = 19.0           # $/MMBtu, ULSD delivered
    co2_price: float = 22.0           # $/short ton (RGGI-like)
    load_growth: float = 1.00         # scales annual energy
    heat_elec_mult: float = 1.00      # heat-pump electrification: scales winter heating slope
    cold_snap: bool = False           # inject a 9-day arctic event in late January
    extra_offshore_mw: float = 0.0    # added SEMA/RI offshore wind
    extra_battery_mw: float = 0.0
    pipeline_bcfd: float = 3.45       # firm interstate pipeline delivery into New England
    lng_bcfd: float = 0.70            # regasified LNG send-out (Everett/Canaport via M&N)
    lng_price: float = 15.5           # $/MMBtu delivered, global-LNG linked
    ldc_peakshave_bcfd: float = 1.10  # LDC on-system LNG/propane vaporization
    hq_mult: float = 1.00             # Quebec import availability (hydro conditions)
    ny_basis: float = 1.0             # NY Zone G gas basis over Henry Hub, $/MMBtu
    retire: list = field(default_factory=list)
    drop_ren: list = field(default_factory=list)

    @staticmethod
    def from_yaml(path: Path) -> "Scenario":
        cfg = yaml.safe_load(Path(path).read_text())
        return Scenario(**cfg)


# --------------------------------------------------------------------------------------
# Static data
# --------------------------------------------------------------------------------------
def load_static(sc: Scenario) -> dict:
    fleet = pd.read_csv(DATA / "fleet.csv")
    fleet = fleet[~fleet["name"].isin(sc.retire)].reset_index(drop=True)
    fleet["commit"] = fleet["tech"].isin(["CC", "CT", "ST"]).astype(int)
    fleet["pmax_eff"] = fleet["pmax"] * (1 - fleet["efor"])
    fleet["pmin"] = fleet["pmax"] * fleet["pmin_frac"]
    # Heat-rate curve: no-load fuel + incremental heat rate (full-load HR preserved)
    fleet["nl_fuel"] = 0.18 * fleet["pmax"] * fleet["hr_fl"]          # MMBtu/h per unit
    fleet["ihr"] = 0.82 * fleet["hr_fl"]                              # MMBtu/MWh
    fleet["oil_capable"] = ((fleet["fuel"] == "oil") | (fleet["dual"] == 1)).astype(int)
    fleet["gas_capable"] = ((fleet["fuel"] == "gas") |
                            ((fleet["fuel"] == "oil") & (fleet["dual"] == 1))).astype(int)

    ren = pd.read_csv(DATA / "renewables.csv")
    ren = ren[~ren["name"].isin(sc.drop_ren)].reset_index(drop=True)
    if sc.extra_offshore_mw:
        add = pd.DataFrame([
            {"name": "New Offshore Wind SEMA", "zone": "SEMA", "type": "wind_off",
             "mw": sc.extra_offshore_mw * 0.6},
            {"name": "New Offshore Wind RI", "zone": "RI", "type": "wind_off",
             "mw": sc.extra_offshore_mw * 0.4}])
        ren = pd.concat([ren, add], ignore_index=True)
    sto = pd.read_csv(DATA / "storage.csv")
    if sc.extra_battery_mw:
        sto = pd.concat([sto, pd.DataFrame([{"name": "New Batteries", "zone": "SEMA",
                                             "mw": sc.extra_battery_mw,
                                             "mwh": 4 * sc.extra_battery_mw, "rte": 0.86}])],
                        ignore_index=True)
    zones = pd.read_csv(DATA / "zones.csv")
    lines = pd.read_csv(DATA / "transmission.csv")
    ifs = pd.read_csv(DATA / "interfaces.csv")
    ifs["terms"] = ifs["terms"].apply(
        lambda s: {k: float(v) for k, v in (t.split(":") for t in s.split(";"))})
    imports = pd.read_csv(DATA / "imports.csv")
    return dict(fleet=fleet, ren=ren, sto=sto, zones=zones, lines=lines,
                ifs=ifs, imports=imports)


# --------------------------------------------------------------------------------------
# Weather, load and renewable profiles
# --------------------------------------------------------------------------------------
def build_weather(sc: Scenario, seed: int = 2027) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2027-01-01", periods=HOURS, freq="h")
    doy = idx.dayofyear.values
    hod = idx.hour.values
    # daily mean temperature (F): Boston-like seasonal cycle + AR(1) synoptic noise
    nd = 365
    noise = np.zeros(nd)
    for d in range(1, nd):
        noise[d] = 0.72 * noise[d - 1] + rng.normal(0, 5.2)
    days = np.arange(1, nd + 1)
    tmean_d = 51.5 - 22.5 * np.cos(2 * np.pi * (days - 22) / 365) + noise
    if sc.cold_snap:  # 9-day arctic outbreak, Jan 20-28 (cf. Winter Storm Elliott / Jan-2018)
        snap = np.array([18, 9, 2, -3, -1, 4, 10, 16, 22], dtype=float)
        tmean_d[19:28] = np.minimum(tmean_d[19:28], snap)
    tmean = tmean_d[doy - 1]
    diurnal = 7.5 * np.cos(2 * np.pi * (hod - 15) / 24)
    temp = tmean + diurnal
    cloud = np.clip(np.repeat(rng.beta(2.2, 1.6, nd), 24), 0.05, 1)
    return pd.DataFrame({"temp": temp, "tmean": tmean, "doy": doy, "hod": hod,
                         "dow": idx.dayofweek.values, "month": idx.month.values,
                         "clear": cloud}, index=idx)


def solar_profile(w: pd.DataFrame) -> np.ndarray:
    """Normalized (0-1) PV output from solar geometry at 42.4N and daily cloudiness."""
    lat = np.radians(42.4)
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + w["doy"].values) / 365)
    ha = np.radians(15 * (w["hod"].values + 0.5 - 12.5))
    elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    raw = np.clip(elev, 0, None) ** 1.15 * (0.35 + 0.65 * w["clear"].values)
    return raw / 0.92


def wind_profiles(w: pd.DataFrame, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    month = w["month"].values
    seas = 1 + 0.32 * np.cos(2 * np.pi * (month - 1.3) / 12)  # windier winters
    z = np.zeros(HOURS)
    z2 = np.zeros(HOURS)
    for t in range(1, HOURS):
        z[t] = 0.965 * z[t - 1] + rng.normal(0, 0.26)
        z2[t] = 0.965 * z2[t - 1] + rng.normal(0, 0.26)
    def cf(base, zz):
        x = base * seas * np.exp(0.55 * zz - 0.15)
        return np.clip(x, 0, 0.97)
    on = cf(0.30, z)
    off = cf(0.44, 0.85 * z + 0.15 * z2)
    return {"wind_on": on, "wind_off": off}


def gross_load_shape(sc: Scenario, w: pd.DataFrame) -> np.ndarray:
    """Unscaled gross system load from temperature, hour of day and day of week."""
    t = w["temp"].values
    hod, dow = w["hod"].values, w["dow"].values
    shape = (0.80 + 0.12 * np.exp(-((hod - 18.5) ** 2) / 10) + 0.08 * np.exp(-((hod - 9) ** 2) / 8)
             - 0.14 * np.exp(-((hod - 3.5) ** 2) / 6))
    wkday = np.where(dow >= 5, 0.93, 1.0)
    heat = 155.0 * sc.heat_elec_mult * np.clip(52 - t, 0, None)
    cool = 30.0 * np.clip(t - 64, 0, None) ** 1.45
    return 11400 * shape * wkday + heat + cool


def load_scale(sc: Scenario, zones: pd.DataFrame) -> float:
    """Factor that calibrates gross_load_shape to ~118 TWh/yr of net energy for load in the
    reference weather year. Also used to drive the live model from real temperatures."""
    w = build_weather(sc)
    gross, pv = gross_load_shape(sc, w), solar_profile(w)
    btm_mwh = zones["btm_solar_mw"].sum() * pv.sum()
    return (118.0e6 * sc.load_growth + btm_mwh) / gross.sum()


def build_load(sc: Scenario, w: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    """Gross zonal load (MW) and BTM-solar-netted load. Calibrated to ~118 TWh/yr, ~24 GW peak."""
    gross = gross_load_shape(sc, w)
    pv = solar_profile(w)
    # calibrate so ISO-NE net energy for load (after BTM PV) is ~118 TWh in the 2025 weather
    btm_mwh = zones["btm_solar_mw"].sum() * pv.sum()
    gross *= (118.0e6 * sc.load_growth + btm_mwh) / gross.sum()
    out = {}
    for _, z in zones.iterrows():
        out[z.zone] = gross * z.load_share - z.btm_solar_mw * pv
    df = pd.DataFrame(out, index=w.index)
    df.attrs["gross_system"] = gross
    return df


def hydro_budget(month: np.ndarray) -> np.ndarray:
    """Monthly conventional-hydro capacity factor (spring freshet peak)."""
    cf = np.array([0.42, 0.40, 0.52, 0.68, 0.60, 0.40, 0.30, 0.26, 0.24, 0.30, 0.40, 0.44])
    return cf[month - 1]


def hq_availability(month: np.ndarray, mult: float) -> np.ndarray:
    base = np.array([0.80, 0.80, 0.70, 0.55, 0.55, 0.75, 0.85, 0.85, 0.70, 0.55, 0.60, 0.75])
    return np.clip(base[month - 1] * mult, 0, 1)


# --------------------------------------------------------------------------------------
# Gas system: New England citygate supply stack and LDC demand
# --------------------------------------------------------------------------------------
def ldc_demand_bcfd(tmean_daily: np.ndarray) -> np.ndarray:
    """Firm LDC (res/com/industrial) demand into New England, Bcf/d, from daily HDD."""
    hdd = np.clip(65 - tmean_daily, 0, None)
    return 0.62 + 0.0405 * hdd


def gas_supply_steps(sc: Scenario, hh: float, winter: bool) -> list[tuple[float, float]]:
    """(capacity MMBtu/d, price $/MMBtu) tranches of the Algonquin-citygate supply curve.

    Pipeline capacity is split into utilization bands priced as Henry Hub + basis; the basis
    widens convexly as pipes approach full (the New England winter 'basis blowout').
    LNG send-out and LDC peak-shaving sit at the top of the stack.
    """
    cap = sc.pipeline_bcfd * MMBTU_PER_BCF
    bands = [(0.55, 0.45), (0.20, 1.10), (0.10, 2.80), (0.07, 6.50), (0.05, 12.0), (0.03, 21.0)]
    adder = 0.35 if winter else 0.0
    steps = [(cap * share, hh + b + adder) for share, b in bands]
    steps.append((sc.lng_bcfd * MMBTU_PER_BCF, max(sc.lng_price, hh + 8.0)))
    steps.append((sc.ldc_peakshave_bcfd * MMBTU_PER_BCF, max(sc.lng_price + 6.0, hh + 14)))
    return steps
