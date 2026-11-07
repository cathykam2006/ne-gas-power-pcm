"""Run all (or selected) scenarios: python run_scenarios.py [scenario_name ...] [--days N]"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from negem.inputs import Scenario  # noqa: E402
from negem.simulate import run, save  # noqa: E402


def _one(args):
    path, days = args
    sc = Scenario.from_yaml(path)
    t = time.time()
    out = run(sc, days=days)
    save(out, ROOT / "results" / sc.name)
    return sc.name, time.time() - t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    files = sorted((ROOT / "scenarios").glob("*.yaml"))
    if a.names:
        files = [f for f in files if Scenario.from_yaml(f).name in a.names]
    with Pool(a.workers) as pool:
        for name, secs in pool.imap_unordered(_one, [(f, a.days) for f in files]):
            print(f"finished {name} in {secs/60:.1f} min", flush=True)
