"""Sanity checks: energy balance closes, prices are economically consistent, gas price >= HH."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from negem.inputs import Scenario  # noqa: E402
from negem.simulate import run  # noqa: E402

ZONES = ["ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA"]


@pytest.fixture(scope="module")
def winter_run():
    return run(Scenario(name="test", label="test"), days=2, start_day=20, verbose=False)


def test_system_energy_balance(winter_run):
    h = winter_run["hourly"]
    supply = h.filter(like="gen_").sum(axis=1) + h[[f"uns_{z}" for z in ZONES]].sum(axis=1)
    load = h[[f"load_{z}" for z in ZONES]].sum(axis=1)
    assert np.all(supply >= load - 1.0)           # surplus only via dump (overgeneration)
    assert (supply - load).abs().mean() < 50.0


def test_gas_price_at_least_henry_hub(winter_run):
    d = winter_run["daily"]
    assert (d["gas_price"] >= d["hh"] - 1e-6).all()


def test_lmp_bounds(winter_run):
    h = winter_run["hourly"]
    lmp = h[[f"lmp_{z}" for z in ZONES]]
    assert lmp.min().min() >= -1e-6
    assert lmp.max().max() <= 5000 + 1e-6


def test_no_flow_exceeds_limits(winter_run):
    import pandas as pd
    lines = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "transmission.csv")
    h = winter_run["hourly"]
    for _, l in lines.iterrows():
        assert h[f"flow_{l.link}"].abs().max() <= l.limit_mw + 1e-3
