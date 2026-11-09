"""Live-data helpers: ISO-NE report parsing, wind power curve, and the live run on stand-in data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from negem import live_data as LD  # noqa: E402
from negem import live as L  # noqa: E402

ISONE_SAMPLE = """"C","Day-Ahead Energy Market Hourly LMP Report"
"C","For Report Date: 11/06/2026"
"H","Date","Hour Ending","Location ID","Location Name","Location Type","Locational Marginal Price","Energy Component","Congestion Component","Marginal Loss Component"
"D","11/06/2026","01","4000",".H.INTERNAL_HUB","HUB","35.10","35.00","0.00","0.10"
"D","11/06/2026","01","4008",".Z.NEMASSBOST","LOAD ZONE","35.70","35.00","0.30","0.40"
"D","11/06/2026","01","321","UN.FRNKLNSQ13.810CC","NETWORK NODE","34.00","35.00","0.00","-1.00"
"T","3"
"""


def test_parse_isone_csv_keeps_data_rows_only():
    df = LD._parse_isone_csv(ISONE_SAMPLE)
    assert len(df) == 3 and list(df["Location Name"])[:2] == [".H.INTERNAL_HUB", ".Z.NEMASSBOST"]
    assert df["Locational Marginal Price"].astype(float).iloc[0] == 35.10


def test_wind_power_curve():
    v = np.array([0, 2.9, 3, 7.5, 12, 20, 25, 30, np.nan])
    cf = L.wind_cf(v, rated=12.0, losses=0.1)
    assert cf[0] == cf[1] == cf[2] == 0 and cf[-1] == 0          # below cut-in, missing
    assert np.isclose(cf[4], 0.9) and np.isclose(cf[5], 0.9)     # rated, minus losses
    assert cf[6] == cf[7] == 0                                     # cut-out
    assert 0 < cf[3] < cf[4]


def test_live_run_on_stand_in_data(tmp_path, monkeypatch):
    """Full live run (14 days) on synthetic stand-in files: outputs exist and prices are sane."""
    live, out = tmp_path / "live", tmp_path / "out"
    live.mkdir()
    monkeypatch.setattr(LD, "LIVE", live)
    monkeypatch.setattr(L, "LIVE", live)
    monkeypatch.setattr(L, "OUT", out)
    rng = np.random.default_rng(0)
    today = pd.Timestamp.now(L.TZ).tz_localize(None).normalize()
    idx = pd.date_range(today - pd.Timedelta(days=31), today + pd.Timedelta(days=10), freq="h", inclusive="left")
    temp = 45 + 8 * np.cos(2 * np.pi * (idx.hour.values - 15) / 24)
    ghi = np.clip(np.sin(np.pi * (idx.hour.values - 6.5) / 11), 0, None) * 400
    wx = pd.concat([pd.DataFrame({"time": idx.strftime("%Y-%m-%d %H:%M"), "site": z, "temp_f": temp,
                                  "ghi_wm2": ghi, "wind100_ms": 8.0}) for z, *_ in LD.STATIONS + LD.WIND_SITES])
    past = pd.to_datetime(wx["time"]) < today
    wx[past].to_csv(live / "weather_actual.csv", index=False)
    wx[~past].to_csv(live / "weather_forecast.csv", index=False)
    da = idx[(idx >= today - pd.Timedelta(days=8)) & (idx < today)]
    pd.DataFrame({"time": da.strftime("%Y-%m-%d %H:%M"), "date": da.strftime("%Y-%m-%d"),
                  "da_lmp_HUB": 40 + rng.normal(0, 5, len(da))}).to_csv(live / "isone_da_lmp.csv", index=False)
    pd.DataFrame({"date": [today.strftime("%Y-%m-%d")], "front": [3.2]}).to_csv(live / "nymex_front.csv", index=False)
    meta = L.run_live(verbose=False)
    d = pd.read_csv(out / "daily.csv")
    assert len(d) == L.PAST_DAYS + L.AHEAD_DAYS
    assert d["lmp_hub_model"].between(5, 500).all() and (d["gas_price"] >= d["hh"] - 1e-6).all()
    assert "hourly_mae" in meta["metrics"] and (out / "track_record.csv").exists()
