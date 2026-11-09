"""Live model run: the production cost model driven by real data, re-solved every morning.

Window: the past 7 days (backcast, scored against ISO-NE's published day-ahead prices) and the
next 7 days (forecast). Inputs come from data/live/ (see live_data.py):

  temperature   Open-Meteo actual + forecast, weighted by ISO-NE zone load shares
  load          past hours: ISO-NE actual demand (EIA-930); future hours: the model's
                temperature-driven load, scaled by its ratio to actual demand over the last 30 days
  solar, wind   capacity factors from Open-Meteo solar radiation and 100 m wind speed
  Henry Hub     EIA daily spot for past days, NYMEX front month for future days
  oil           NY Harbor ULSD spot (EIA) + delivery adder, when available

Each run also appends its day-ahead forecasts to results/live/track_record.csv, which is scored
against the actual day-ahead hub price as it is published, building an out-of-sample record.

    python -m negem.live
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import inputs as I
from . import simulate as S
from .live_data import LIVE, STATIONS, TZ

OUT = Path(__file__).resolve().parents[2] / "results" / "live"
PAST_DAYS, AHEAD_DAYS = 7, 7
ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA"]
LIVE_SCENARIO = dict(name="live", label="Live (real data)", load_growth=1.0, heat_elec_mult=1.10,
                     description="Current system, real weather, load and fuel prices")


# ------------------------------------------------------------------------------------------
def _read(name: str, **kw) -> pd.DataFrame | None:
    p = LIVE / name
    return pd.read_csv(p, **kw) if p.exists() else None


def wind_cf(v: np.ndarray, rated: float, losses: float) -> np.ndarray:
    """Generic turbine power curve: cut-in 3 m/s, rated speed, cut-out 25 m/s; less losses."""
    v = np.nan_to_num(v)
    cf = np.clip((v - 3.0) / (rated - 3.0), 0, 1) ** 3
    cf[v >= 25] = 0
    return cf * (1 - losses)


def weather_frame(idx: pd.DatetimeIndex, zones: pd.DataFrame) -> pd.DataFrame:
    act, fc = _read("weather_actual.csv"), _read("weather_forecast.csv")
    if act is None and fc is None:
        raise FileNotFoundError("no weather data: run `python -m negem.live_data` first")
    wx = pd.concat([d for d in (act, fc) if d is not None], ignore_index=True)
    wx["time"] = pd.to_datetime(wx["time"])
    wx = wx.drop_duplicates(["time", "site"], keep="last")
    t = wx.pivot_table(index="time", columns="site", values="temp_f")
    g = wx.pivot_table(index="time", columns="site", values="ghi_wm2")
    s = wx.pivot_table(index="time", columns="site", values="wind100_ms")
    share = zones.set_index("zone")["load_share"]
    sites = [z for z, *_ in STATIONS if z in t]
    temp = (t[sites] * share[sites]).sum(axis=1) / share[sites].sum()
    ghi = g[sites].mean(axis=1)
    out = pd.DataFrame({"temp": temp, "ghi": ghi,
                        "ws_on": s.get("wind_on"), "ws_off": s.get("wind_off")}).sort_index()
    out = out[~out.index.duplicated()].reindex(idx).interpolate(limit=6).ffill().bfill()
    out["src"] = np.where(out.index < pd.Timestamp.now(TZ).tz_localize(None).normalize(), "actual", "forecast")
    return out


def prepare_live(sc: I.Scenario, start: pd.Timestamp, days: int) -> tuple[dict, pd.DataFrame]:
    """Inputs in the same shape as simulate.prepare(), for `days` days from `start` (+1 day of look-ahead)."""
    st = I.load_static(sc)
    zones = st["zones"]
    idx = pd.date_range(start, periods=(days + 1) * 24, freq="h")
    wx = weather_frame(idx, zones)

    w = pd.DataFrame({"temp": wx["temp"].values, "doy": idx.dayofyear, "hod": idx.hour, "dow": idx.dayofweek,
                      "month": idx.month, "clear": 1.0}, index=idx)
    w["tmean"] = w["temp"].groupby(idx.date).transform("mean").values
    pv = np.clip(wx["ghi"].values / 920.0, 0, 1.05)
    wind = {"wind_on": wind_cf(wx["ws_on"].values, 12.0, 0.20),
            "wind_off": wind_cf(wx["ws_off"].values, 11.5, 0.10)}

    # --- load: model from temperature, bias-corrected to actual ISO-NE demand
    k = I.load_scale(sc, zones)
    btm = zones["btm_solar_mw"].sum()
    model_sys = I.gross_load_shape(sc, w) * k - btm * pv
    act = _read("isone_load.csv")
    actual = pd.Series(np.nan, index=idx)
    ratio = 1.0
    if act is not None and "load_mw" in act:
        a = act.assign(time=pd.to_datetime(act["time"])).set_index("time")["load_mw"]
        a = a[~a.index.duplicated()]
        actual = a.reindex(idx)
        # ratio over the last 30 days of overlap between the model and actual demand
        idx30 = pd.date_range(pd.Timestamp.now(TZ).tz_localize(None).normalize() - pd.Timedelta(days=30),
                              periods=30 * 24, freq="h")
        try:
            wx30 = weather_frame(idx30, zones)
            w30 = pd.DataFrame({"temp": wx30["temp"].values, "hod": idx30.hour, "dow": idx30.dayofweek}, index=idx30)
            m30 = pd.Series(I.gross_load_shape(sc, w30) * k - btm * np.clip(wx30["ghi"].values / 920, 0, 1.05),
                            index=idx30)
            both = pd.concat([m30, a.reindex(idx30)], axis=1).dropna()
            if len(both) > 24 * 5:
                ratio = float(both.iloc[:, 1].sum() / both.iloc[:, 0].sum())
        except Exception:
            pass
    model_adj = model_sys * ratio
    sys_load = np.where(actual.notna().values, actual.values, model_adj)
    load = pd.DataFrame({z.zone: sys_load * z.load_share for _, z in zones.iterrows()}, index=idx)

    # --- Henry Hub per day: EIA spot where published, otherwise the latest NYMEX front month
    dates = pd.date_range(start, periods=days + 1, freq="D")
    spot, front = _read("henry_hub_spot.csv"), _read("nymex_front.csv")
    hh = pd.Series(np.nan, index=dates)
    if spot is not None:
        sp = spot.assign(date=pd.to_datetime(spot["date"])).set_index("date")["hh_spot"].sort_index()
        hh = hh.fillna(sp.reindex(dates))
    fut_px = None
    if front is not None and len(front):
        fut_px = float(front.sort_values("date")["front"].iloc[-1])
    today = pd.Timestamp.now(TZ).tz_localize(None).normalize()
    if fut_px is not None:
        hh[hh.index >= today] = fut_px
    hh = hh.ffill().bfill().fillna(np.mean(sc.hh_monthly))
    hh_daily = hh.values

    tmean_daily = w["tmean"].values[::24]
    hdd = np.clip(65 - tmean_daily, 0, None)
    ldc = I.ldc_demand_bcfd(tmean_daily) * I.MMBTU_PER_BCF
    nyb = sc.ny_basis + 0.12 * np.clip(hdd - 25, 0, None)
    n = len(idx)
    hh_h, nyb_h = np.repeat(hh_daily, 24), np.repeat(nyb, 24)
    hq = I.hq_availability(w["month"].values, sc.hq_mult)
    D = dict(sc=sc, st=st, w=w, load=load, pv=pv, wind=wind, hdd=hdd, hh=hh_daily, ldc=ldc,
             imp_offer={"HQ Phase II": np.full(n, 22.0), "Highgate": np.full(n, 24.0),
                        "New Brunswick": 8.0 * (hh_h + 2.5) + 3, "NYISO": 7.4 * (hh_h + nyb_h) + 6},
             exp_bid={"NYISO": 7.0 * (hh_h + nyb_h) - 6},
             imp_avail={"HQ Phase II": hq, "Highgate": hq, "New Brunswick": np.full(n, 0.8), "NYISO": np.ones(n)},
             nuc={k_: np.ones(n) for k_ in ["Seabrook", "Millstone 2", "Millstone 3"]},
             hydro_cf=I.hydro_budget(w["month"].values), load_ratio=ratio)
    inputs = pd.DataFrame({"temp_f": w["temp"].values, "weather": wx["src"].values,
                           "load_actual_mw": actual.values, "load_model_mw": model_adj,
                           "solar_cf": pv, "wind_on_cf": wind["wind_on"], "wind_off_cf": wind["wind_off"],
                           "hh": hh_h}, index=idx)
    return D, inputs


# ------------------------------------------------------------------------------------------
def _hub(h: pd.DataFrame) -> pd.Series:
    L = h[[f"load_{z}" for z in ZONES]].to_numpy()
    P = h[[f"lmp_{z}" for z in ZONES]].to_numpy()
    return pd.Series((L * P).sum(1) / np.maximum(L.sum(1), 1), index=h.index)


def run_live(verbose: bool = True) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    sc = I.Scenario(**LIVE_SCENARIO)
    oil = _read("ulsd_spot.csv")
    if oil is not None and len(oil):
        sc.oil_price = round(float(oil.sort_values("date")["ulsd_mmbtu"].iloc[-1]) + 2.0, 2)  # + delivery
    today = pd.Timestamp.now(TZ).tz_localize(None).normalize()
    start = today - pd.Timedelta(days=PAST_DAYS)
    days = PAST_DAYS + AHEAD_DAYS
    D, inp = prepare_live(sc, start, days)
    out = S.run(sc, days=days, start_day=0, verbose=verbose, D=D)

    h = out["hourly"].copy()
    h.index = D["w"].index[: len(h)]
    h["lmp_hub_model"] = _hub(h)
    inp = inp.iloc[: len(h)]
    da = _read("isone_da_lmp.csv")
    if da is not None:
        da = da.assign(time=pd.to_datetime(da["time"])).drop(columns=["date"]).set_index("time")
        da = da[~da.index.duplicated()]
        inp = inp.join(da, how="left")
    inp = inp.join(h[["lmp_hub_model"] + [f"lmp_{z}" for z in ZONES]])

    daily = out["daily"].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    g = inp.groupby(inp.index.normalize())
    daily = daily.set_index("date").join(pd.DataFrame({
        "lmp_hub_model": g["lmp_hub_model"].mean(),
        "lmp_hub_actual": g["da_lmp_HUB"].mean() if "da_lmp_HUB" in inp else np.nan,
        "peak_load_model": g["load_model_mw"].max(),
        "peak_load_actual": g["load_actual_mw"].max(),
        "temp_mean": g["temp_f"].mean()})).reset_index()
    daily["kind"] = np.where(daily["date"] < today, "backcast", "forecast")

    # accuracy over the backcast days that have published day-ahead prices
    past = inp[(inp.index < today)]
    m = {}
    if "da_lmp_HUB" in past and past["da_lmp_HUB"].notna().sum() >= 24:
        e = (past["lmp_hub_model"] - past["da_lmp_HUB"]).dropna()
        dd = daily[(daily["kind"] == "backcast")].dropna(subset=["lmp_hub_actual"])
        m = {"hours": int(e.size), "hourly_mae": round(float(e.abs().mean()), 2),
             "hourly_bias": round(float(e.mean()), 2),
             "hourly_corr": round(float(past[["lmp_hub_model", "da_lmp_HUB"]].dropna().corr().iloc[0, 1]), 3),
             "daily_mae": round(float((dd["lmp_hub_model"] - dd["lmp_hub_actual"]).abs().mean()), 2),
             "actual_avg": round(float(past["da_lmp_HUB"].mean()), 2)}
    lp = past.dropna(subset=["load_actual_mw"])
    if len(lp):
        m["load_mape_pct"] = round(float(((lp["load_model_mw"] - lp["load_actual_mw"]).abs()
                                          / lp["load_actual_mw"]).mean() * 100), 2)

    # day-ahead track record: forecasts made today for the next 7 days, scored as actuals arrive
    tr_p = OUT / "track_record.csv"
    fc = daily[daily["kind"] == "forecast"]
    new = pd.DataFrame({"run_date": today.strftime("%Y-%m-%d"), "target_date": fc["date"].dt.strftime("%Y-%m-%d"),
                        "horizon_days": (fc["date"] - today).dt.days, "model_hub": fc["lmp_hub_model"].round(2),
                        "model_gas": fc["gas_price"].round(2), "model_peak_load": fc["peak_load_model"].round(0),
                        "hh": fc["hh"].round(3)})
    tr = pd.concat([pd.read_csv(tr_p), new], ignore_index=True) if tr_p.exists() else new
    tr = tr.drop_duplicates(["run_date", "target_date"], keep="last")
    if da is not None and "da_lmp_HUB" in da:
        act = da["da_lmp_HUB"].groupby(da.index.normalize()).mean()
        act.index = act.index.strftime("%Y-%m-%d")
        tr["actual_hub"] = tr["target_date"].map(act).round(2)
    tr.to_csv(tr_p, index=False)
    scored = tr.dropna(subset=["actual_hub"]) if "actual_hub" in tr else tr.iloc[0:0]
    if len(scored):
        d1 = scored[scored["horizon_days"] == 1]
        m["track_days_scored"] = int(len(d1))
        if len(d1):
            m["track_day_ahead_mae"] = round(float((d1["model_hub"] - d1["actual_hub"]).abs().mean()), 2)

    h.to_csv(OUT / "hourly.csv.gz", compression="gzip")
    daily.to_csv(OUT / "daily.csv", index=False)
    inp.to_csv(OUT / "inputs_hourly.csv.gz", compression="gzip")
    status = json.loads((LIVE / "status.json").read_text()) if (LIVE / "status.json").exists() else {}
    meta = {"run_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "run_date": today.strftime("%Y-%m-%d"),
            "window": [start.strftime("%Y-%m-%d"), (start + pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")],
            "load_ratio": round(float(D["load_ratio"]), 4), "oil_price": sc.oil_price,
            "metrics": m, "sources": status}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    if verbose:
        print(json.dumps(meta["metrics"], indent=1))
    return meta


if __name__ == "__main__":
    run_live()
