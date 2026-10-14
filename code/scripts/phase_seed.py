# SPDX-License-Identifier: GPL-3.0-or-later
"""Seed a phase (EC_PHASE=pseudo|final) with the 2025-only artifacts that do not depend on the test set.

The final-phase stages rebuild everything that sees the test rows (movements, features, test-side refits, rank-dependent
caches). What was computed from 2025 alone and is expensive or impossible to redo on this host is copied (real copies, never
links: a stage rewriting a file in place must not touch the base file):

  reports/overnight/results/*.json  -> reports/overnight_<phase>/results/   (configs and fold iterations that final.py reads)
  outputs/overnight/oof/            -> outputs/overnight_<phase>/oof/       (2025 OOF predictions of every member / blend)
  outputs/overnight/cache/{adsb,adsb2,adsb2m}/<2025 days>               stage-1 ADS-B products of 2025 days. Jan/Jul 2026 days
                                                                           are NOT seeded: the final file jitters take-off times
                                                                           +-12 s, so they are recomputed from fresh raw points.
  data/processed/ per-row ADS-B files of the raw-point scripts (adsbrev, adsbgeo2): the 2025 rows only; the phase run adds the
                                                                           new rows with EC_MERGE=1 (raw points exist only for
                                                                           the new days on this host).
  data/processed/opdi_{flights,events}_10apt.parquet                       OPDI extraction (pseudo only; final re-extracts)

  python scripts/phase_seed.py pseudo [--oof] [--dry]
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHES = ("adsb", "adsb2", "adsb2m")
ROWFILES = ("features_dep_adsbrev", "features_dep_adsbgeo2")


def cp(src, dst, dry):
    if dry:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["pseudo", "pseudo2", "pseudo3", "pseudo4", "pseudo5", "pseudo6", "final"])
    ap.add_argument("--oof", action="store_true", help="also copy the 2025 OOF predictions (5 GB)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--keep-days", default="2025", help="cache days seeded: those starting with this prefix")
    a = ap.parse_args()
    ph = a.phase
    n = 0
    for f in sorted((ROOT / "reports/overnight/results").glob("*.json")):
        cp(f, ROOT / f"reports/overnight_{ph}/results" / f.name, a.dry)
        n += 1
    print(f"results json: {n}")
    if a.oof:
        n = 0
        for f in sorted((ROOT / "outputs/overnight/oof").glob("*.parquet")):
            d = ROOT / f"outputs/overnight_{ph}/oof" / f.name
            if not d.exists():
                cp(f, d, a.dry)
            n += 1
        print(f"oof files: {n}")
    for c in CACHES:
        src = ROOT / "outputs/overnight/cache" / c
        n = 0
        for f in sorted(src.glob(f"{a.keep_days}*.parquet")) if src.is_dir() else []:
            d = ROOT / f"outputs/overnight_{ph}/cache" / c / f.name
            if not d.exists():
                cp(f, d, a.dry)
            n += 1
        print(f"cache {c}: {n} days")
    for f in ("arr_witness",):  # small 2025-only table the rules chain reads
        cp(ROOT / f"outputs/overnight/cache/{f}.parquet", ROOT / f"outputs/overnight_{ph}/cache/{f}.parquet", a.dry)
    # per-row raw-point features: keep the rows of 2025 days (MVT time < 2026) so Jan/Jul 2026 rows are rebuilt
    mv = pd.read_parquet(ROOT / "data/processed/features_dep.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"])
    ids = set(mv.MVT_ID_mvt[pd.to_datetime(mv.MVT_TIME_UTC_mvt, unit="s") < "2026-01-01"])
    for s in ROWFILES:
        src, dst = ROOT / f"data/processed/{s}.parquet", ROOT / f"data/processed_{ph}/{s}.parquet"
        if not src.exists():
            print(f"missing {src.name}")
            continue
        d = pd.read_parquet(src)
        d = d[d.MVT_ID_mvt.isin(ids)]
        print(f"{s}: {len(d):,} 2025 rows")
        if not a.dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            d.to_parquet(dst, index=False)
    if ph.startswith("pseudo"):
        for s in ("opdi_flights_10apt", "opdi_events_10apt"):
            cp(ROOT / f"data/processed/{s}.parquet", ROOT / f"data/processed_{ph}/{s}.parquet", a.dry)
        print("opdi extraction seeded")


if __name__ == "__main__":
    sys.exit(main())
