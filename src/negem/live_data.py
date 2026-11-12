"""Live data connectors: pull real New England power and gas data into data/live/.

Sources (all public):
  * Open-Meteo        hourly temperature, solar radiation and 100 m wind speed, actual (past
                      31 days) and forecast (next 10 days), for seven New England cities and two
                      wind sites. No key.
  * EIA-930 (EIA API) ISO-NE hourly demand, EIA's day-ahead demand forecast and net generation
                      by fuel. Needs EIA_API_KEY (free at https://www.eia.gov/opendata/).
  * ISO-NE            day-ahead hourly LMPs for the Internal Hub and the eight load zones
                      (public ISO Express CSV reports). No key.
  * NYMEX via Yahoo   Henry Hub front month and the next 12 monthly contracts. No key.
  * EIA (EIA API)     daily Henry Hub spot price and New York Harbor ultra-low-sulfur diesel
                      spot price (the oil-unit fuel cost).

Every source is fetched independently: a failure is recorded in data/live/status.json and the
previous file is kept, so one outage never blocks the daily model run.

    python -m negem.live_data            # fetch everything
"""
from __future__ import annotations

import io
import json
import re
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

LIVE = Path(__file__).resolve().parents[2] / "data" / "live"
TZ = "America/New_York"
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/130.0.0.0 Safari/537.36"), "Accept-Language": "en-US,en;q=0.9"}

# weather stations: (zone, name, lat, lon). Zone load shares weight the temperature.
STATIONS = [("NEMA", "Boston", 42.36, -71.06), ("SEMA", "New Bedford", 41.64, -70.93),
            ("RI", "Providence", 41.82, -71.41), ("CT", "Hartford", 41.76, -72.68),
            ("WCMA", "Worcester", 42.26, -71.80), ("NH", "Manchester", 42.99, -71.46),
            ("ME", "Portland", 43.66, -70.26), ("VT", "Burlington", 44.48, -73.21)]
WIND_SITES = [("wind_on", "Western Maine ridgelines", 45.05, -70.30),
              ("wind_off", "Southern New England lease area", 41.05, -70.55)]

ISONE_ZONES = {".H.INTERNAL_HUB": "HUB", ".Z.MAINE": "ME", ".Z.NEWHAMPSHIRE": "NH", ".Z.VERMONT": "VT",
               ".Z.CONNECTICUT": "CT", ".Z.RHODEISLAND": "RI", ".Z.SEMASS": "SEMA", ".Z.WCMASS": "WCMA",
               ".Z.NEMASSBOST": "NEMA"}
MONTH_CODES = "FGHJKMNQUVXZ"


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def redact(text: str) -> str:
    """Remove API keys from error text before it is printed or saved (the repo is public)."""
    text = re.sub(r"(api_key=)[^&\s'\"]+", r"\1***", str(text))
    key = os.environ.get("EIA_API_KEY", "").strip()
    return text.replace(key, "***") if key else text


RETRY_WAIT = 20      # seconds before the 2nd attempt; doubled before the 3rd


def _request(url: str, params=None, timeout: int = 90, attempts: int = 3) -> requests.Response:
    """requests.get with retries on timeouts, dropped connections and 5xx/429 responses.
    Free public APIs occasionally stall for a minute; one slow response should not fail the day."""
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and i < attempts - 1:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            if i == attempts - 1:
                raise
            wait = RETRY_WAIT * 2 ** i
            print(f"    retrying in {wait}s after {type(e).__name__}")
            time.sleep(wait)


def _get(url: str, params: dict | None = None, timeout: int = 90) -> requests.Response:
    """GET with browser headers and retries; on HTTP 403 retry with curl_cffi (Chrome TLS
    fingerprint), since some public sites block plain HTTP clients from cloud servers."""
    r = _request(url, params=params, timeout=timeout)
    if r.status_code == 403:
        try:
            from curl_cffi import requests as cf
            r = cf.get(url, params=params, impersonate="chrome", timeout=timeout)
        except ImportError:
            pass
    r.raise_for_status()
    return r


def _merge_save(new: pd.DataFrame, path: Path, keys: list[str], keep_days: int | None = None,
                time_col: str | None = None) -> pd.DataFrame:
    """Append new rows to a CSV history, newest value wins, optionally trimmed to keep_days."""
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new.astype({c: old[c].dtype for c in keys if c in old})], ignore_index=True)
    new = new.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
    if keep_days and time_col:
        cutoff = (pd.Timestamp.now(TZ).tz_localize(None) - pd.Timedelta(days=keep_days)).strftime("%Y-%m-%d")
        new = new[new[time_col].astype(str) >= cutoff]
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def _local_from_utc(s: pd.Series) -> pd.Series:
    """UTC timestamps -> New York local wall-clock (naive). The repeated fall-back hour is kept once."""
    return pd.to_datetime(s, utc=True).dt.tz_convert(TZ).dt.tz_localize(None)


# ------------------------------------------------------------------------------------------
# Open-Meteo: weather actuals and forecast
# ------------------------------------------------------------------------------------------
def fetch_weather(past_days: int = 31, forecast_days: int = 10) -> pd.DataFrame:
    pts = STATIONS + WIND_SITES
    params = {"latitude": ",".join(str(p[2]) for p in pts), "longitude": ",".join(str(p[3]) for p in pts),
              "hourly": "temperature_2m,shortwave_radiation,wind_speed_100m",
              "temperature_unit": "fahrenheit", "wind_speed_unit": "ms", "timezone": TZ,
              "past_days": past_days, "forecast_days": forecast_days}
    js = _get("https://api.open-meteo.com/v1/forecast", params).json()
    js = js if isinstance(js, list) else [js]
    frames = []
    for (key, name, _, _), loc in zip(pts, js):
        h = loc["hourly"]
        frames.append(pd.DataFrame({"time": pd.to_datetime(h["time"]), "site": key,
                                    "temp_f": h["temperature_2m"], "ghi_wm2": h["shortwave_radiation"],
                                    "wind100_ms": h["wind_speed_100m"]}))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["time", "site"])
    today = pd.Timestamp.now(TZ).tz_localize(None).normalize()
    df["kind"] = np.where(df["time"] < today, "actual", "forecast")
    df["issued"] = str(today.date())
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M")
    _merge_save(df[df["kind"] == "actual"], LIVE / "weather_actual.csv", ["time", "site"], 400, "time")
    df[df["kind"] == "forecast"].to_csv(LIVE / "weather_forecast.csv", index=False)
    return df


# ------------------------------------------------------------------------------------------
# EIA-930: ISO-NE demand, EIA demand forecast and generation by fuel
# ------------------------------------------------------------------------------------------
def _eia(route: str, facets: dict, start: str, key: str, frequency: str = "hourly") -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        q = [("api_key", key), ("frequency", frequency), ("data[0]", "value"), ("start", start),
             ("sort[0][column]", "period"), ("sort[0][direction]", "asc"), ("offset", offset), ("length", 5000)]
        for k, vals in facets.items():
            q += [(f"facets[{k}][]", v) for v in vals]
        r = _request(f"https://api.eia.gov/v2/{route}/data/", params=q, timeout=90)
        r.raise_for_status()
        resp = r.json()["response"]
        rows += resp["data"]
        offset += 5000
        if offset >= int(resp.get("total", 0)):
            break
        time.sleep(0.3)
    return pd.DataFrame(rows)


def fetch_eia930(key: str, days: int = 35) -> None:
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT00")
    d = _eia("electricity/rto/region-data", {"respondent": ["ISNE"], "type": ["D", "DF"]}, start, key)
    if d.empty:
        raise ValueError("EIA-930 returned no ISO-NE demand rows")
    d["time"] = _local_from_utc(d["period"] + ":00").dt.strftime("%Y-%m-%d %H:%M")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    wide = d.pivot_table(index="time", columns="type", values="value", aggfunc="first").reset_index()
    wide = wide.rename(columns={"D": "load_mw", "DF": "eia_forecast_mw"})
    _merge_save(wide, LIVE / "isone_load.csv", ["time"], 400, "time")

    f = _eia("electricity/rto/fuel-type-data", {"respondent": ["ISNE"]}, start, key)
    if len(f):
        f["time"] = _local_from_utc(f["period"] + ":00").dt.strftime("%Y-%m-%d %H:%M")
        f["value"] = pd.to_numeric(f["value"], errors="coerce")
        fw = f.pivot_table(index="time", columns="fueltype", values="value", aggfunc="first").reset_index()
        fw.columns = ["time"] + [f"gen_{c}" for c in fw.columns[1:]]
        _merge_save(fw, LIVE / "isone_fuelmix.csv", ["time"], 400, "time")


def fetch_hh_spot(key: str) -> None:
    start = (date.today() - timedelta(days=120)).isoformat()
    d = _eia("natural-gas/pri/fut", {"series": ["RNGWHHD"]}, start, key, frequency="daily")
    if d.empty:
        raise ValueError("EIA returned no Henry Hub spot prices")
    d = d.rename(columns={"period": "date", "value": "hh_spot"})[["date", "hh_spot"]]
    d["hh_spot"] = pd.to_numeric(d["hh_spot"], errors="coerce")
    _merge_save(d.dropna(), LIVE / "henry_hub_spot.csv", ["date"], 800, "date")


def fetch_ulsd(key: str) -> None:
    start = (date.today() - timedelta(days=120)).isoformat()
    d = _eia("petroleum/pri/spt", {"series": ["EER_EPD2DXL0_PF4_Y35NY_DPG"]}, start, key, frequency="daily")
    if d.empty:
        raise ValueError("EIA returned no ULSD spot prices")
    d = d.rename(columns={"period": "date", "value": "ulsd_gal"})[["date", "ulsd_gal"]]
    d["ulsd_gal"] = pd.to_numeric(d["ulsd_gal"], errors="coerce")
    d["ulsd_mmbtu"] = (d["ulsd_gal"] / 0.1374).round(3)          # 137,381 Btu per gallon
    _merge_save(d.dropna(), LIVE / "ulsd_spot.csv", ["date"], 800, "date")


# ------------------------------------------------------------------------------------------
# ISO-NE: day-ahead LMPs (hub + load zones)
# ------------------------------------------------------------------------------------------
def _parse_isone_csv(text: str) -> pd.DataFrame:
    """ISO Express CSV reports: 'C' comment rows, 'H' header rows, 'D' data rows."""
    import csv
    rows = list(csv.reader(io.StringIO(text)))
    head = next(r for r in rows if r and r[0] == "H")
    data = [r for r in rows if r and r[0] == "D"]
    return pd.DataFrame([r[1:] for r in data], columns=head[1:])


def fetch_da_lmp(days_back: int = 10) -> None:
    today = pd.Timestamp.now(TZ).tz_localize(None).normalize()
    have = set()
    p = LIVE / "isone_da_lmp.csv"
    if p.exists():
        have = set(pd.read_csv(p, usecols=["date"])["date"].astype(str))
    frames, missing = [], []
    for k in range(-days_back, 2):           # through tomorrow (published ~1:30 pm the day before)
        d = today + pd.Timedelta(days=k)
        if d.strftime("%Y-%m-%d") in have and k < 0:
            continue
        url = f"https://www.iso-ne.com/static-transform/csv/histRpts/da-lmp/WW_DALMP_ISO_{d:%Y%m%d}.csv"
        try:
            df = _parse_isone_csv(_get(url).text)
        except Exception as e:  # tomorrow's report is not out yet before ~1:30 pm
            missing.append(f"{d:%Y-%m-%d} ({type(e).__name__})")
            continue
        df = df[df["Location Name"].isin(ISONE_ZONES)]
        he = df["Hour Ending"].astype(str)
        df = df[~he.str.contains("X")]                       # repeated DST hour
        df["time"] = (pd.to_datetime(df["Date"]) + pd.to_timedelta(he.str.extract(r"(\d+)")[0].astype(int) - 1,
                                                                    unit="h")).dt.strftime("%Y-%m-%d %H:%M")
        df["zone"] = df["Location Name"].map(ISONE_ZONES)
        df["lmp"] = pd.to_numeric(df["Locational Marginal Price"], errors="coerce")
        w = df.pivot_table(index="time", columns="zone", values="lmp", aggfunc="first").reset_index()
        w.columns = ["time"] + [f"da_lmp_{c}" for c in w.columns[1:]]
        w.insert(1, "date", w["time"].str[:10])
        frames.append(w)
        time.sleep(0.5)
    if frames:
        _merge_save(pd.concat(frames, ignore_index=True), p, ["time"], 800, "time")
    if not frames and not p.exists():
        raise ValueError("no ISO-NE day-ahead LMP reports could be downloaded: " + "; ".join(missing[:3]))


# ------------------------------------------------------------------------------------------
# NYMEX Henry Hub futures (Yahoo Finance)
# ------------------------------------------------------------------------------------------
def fetch_nymex(months: int = 13) -> None:
    import yfinance as yf
    today = pd.Timestamp.now(TZ).tz_localize(None).normalize()
    tickers, labels = [], []
    m = today.to_period("M") + 1
    for i in range(months + 1):
        per = m + i
        tickers.append(f"NG{MONTH_CODES[per.month - 1]}{per.year % 100:02d}.NYM")
        labels.append(str(per))
    px = yf.download(tickers + ["NG=F"], period="10d", interval="1d", progress=False, auto_adjust=False,
                     group_by="column", threads=False)
    close = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px[["Close"]]
    rows = []
    for t, lab in zip(tickers, labels):
        if t in close and close[t].dropna().size:
            s = close[t].dropna()
            if s.index[-1] >= today - pd.Timedelta(days=7) and 0.5 < s.iloc[-1] < 50:
                rows.append({"contract": lab, "settle": round(float(s.iloc[-1]), 4),
                             "as_of": s.index[-1].strftime("%Y-%m-%d")})
    if not rows:
        raise ValueError("no NYMEX contracts returned")
    strip = pd.DataFrame(rows)
    strip.to_csv(LIVE / "nymex_strip.csv", index=False)
    if "NG=F" in close and close["NG=F"].dropna().size:
        f = close["NG=F"].dropna().rename("front").reset_index()
        f.columns = ["date", "front"]
        f["date"] = pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d")
        _merge_save(f, LIVE / "nymex_front.csv", ["date"], 800, "date")


# ------------------------------------------------------------------------------------------
def fetch_all() -> dict:
    LIVE.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("EIA_API_KEY", "").strip()
    jobs = [("weather (Open-Meteo)", fetch_weather),
            ("ISO-NE day-ahead LMPs", fetch_da_lmp),
            ("NYMEX Henry Hub futures", fetch_nymex)]
    if key:
        jobs += [("ISO-NE load and fuel mix (EIA-930)", lambda: fetch_eia930(key)),
                 ("Henry Hub spot (EIA)", lambda: fetch_hh_spot(key)),
                 ("Heating oil / ULSD spot (EIA)", lambda: fetch_ulsd(key))]
    status = {}
    for name, fn in jobs:
        try:
            fn()
            status[name] = {"ok": True, "at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
            print(f"  ok    {name}")
        except Exception as e:
            msg = redact(f"{type(e).__name__}: {e}")
            if "403" in msg and "eia.gov" in msg:
                msg = "EIA refused the API key (HTTP 403): check the EIA_API_KEY secret"
            status[name] = {"ok": False, "error": msg[:300],
                            "at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
            print(f"  FAIL  {name}: {msg}")
            if os.environ.get("GITHUB_ACTIONS"):
                print(f"::warning title=Live data: {name}::{msg[:250]}")
    if not key:
        status["ISO-NE load and fuel mix (EIA-930)"] = {"ok": False, "error": "EIA_API_KEY not set"}
        status["Henry Hub spot (EIA)"] = {"ok": False, "error": "EIA_API_KEY not set"}
        status["Heating oil / ULSD spot (EIA)"] = {"ok": False, "error": "EIA_API_KEY not set"}
    old = json.loads((LIVE / "status.json").read_text()) if (LIVE / "status.json").exists() else {}
    for k, v in status.items():          # keep the last successful time for each source
        v["last_ok"] = v["at"] if v.get("ok") else old.get(k, {}).get("last_ok")
    (LIVE / "status.json").write_text(json.dumps(status, indent=1))
    return status


def weather_age_hours() -> float | None:
    """Hours since the saved weather forecast was issued (None if there is none)."""
    p = LIVE / "weather_forecast.csv"
    if not p.exists():
        return None
    issued = pd.read_csv(p, usecols=["issued"])["issued"].max()
    return (pd.Timestamp.now(TZ).tz_localize(None) - pd.Timestamp(issued)).total_seconds() / 3600


if __name__ == "__main__":
    st = fetch_all()
    if "--strict" in sys.argv and not st.get("weather (Open-Meteo)", {}).get("ok"):
        age = weather_age_hours()
        if age is None or age > 36:
            sys.exit("Weather download failed and the saved forecast is missing or more than 36 hours old; "
                     "the live model cannot run without it.")
        print(f"  Weather download failed; running on the saved forecast ({age:.0f} hours old).")
