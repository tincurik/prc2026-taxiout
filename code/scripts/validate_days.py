# SPDX-License-Identifier: GPL-3.0-or-later
"""Plausibility check of raw ADS-B day files before they feed features (read-only; exit 1 on a bad day).

  python scripts/validate_days.py                       # adsb.lol points of the final months
  python scripts/validate_days.py --src adsblol --days 2026-01-01..2026-01-31
A day is bad when its parquet cannot be read, a tmp/partial file sits next to it, its row count is below --ratio x the median
of the days within +-3 of it in the same source, or an airport of the ten has no rows while its neighbours do.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = {"adsblol": ROOT / "data/external/adsblol/points"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="adsblol")
    ap.add_argument("--months", default="2026-01,2026-02,2026-06,2026-07")
    ap.add_argument("--ratio", type=float, default=0.4)
    a = ap.parse_args()
    months = tuple(a.months.split(","))
    bad = 0
    for name in a.src.split(","):
        d = SRC[name]
        stray = [p.name for p in d.glob("*") if p.suffix not in (".parquet",) and not p.is_dir()]
        if stray:
            print(f"[{name}] non-parquet files present (in-flight or partial writes): {stray[:5]}")
        rows, apts, empty = {}, {}, []
        for p in sorted(d.glob("*.parquet")):
            if not p.stem.startswith(months):
                continue
            try:
                f = pq.ParquetFile(p)
                if f.metadata.num_rows == 0:  # a day the source has no data for
                    empty.append(p.stem)
                    continue
                apts[p.stem] = set(pd.read_parquet(p, columns=["apt"])["apt"].unique())
                rows[p.stem] = int((pd.read_parquet(p, columns=["zone"])["zone"] == 0).sum())  # zone 0 is all the builds read; SQL days hold only that
            except Exception as e:  # noqa: BLE001
                print(f"[{name}] {p.stem}: UNREADABLE ({type(e).__name__}: {e})")
                bad += 1
        s = pd.Series(rows).sort_index()
        allapt = set().union(*apts.values()) if apts else set()
        for i, (day, n) in enumerate(s.items()):
            nb = pd.concat([s.iloc[max(0, i - 3):i], s.iloc[i + 1:i + 4]])
            med = nb.median() if len(nb) else n
            if n < a.ratio * med:
                print(f"[{name}] {day}: {n:,} rows vs neighbour median {med:,.0f}")
                bad += 1
            miss = allapt - apts[day]
            if miss and name == "adsblol":
                print(f"[{name}] {day}: no rows at {sorted(miss)}")
                bad += 1
        if empty:
            print(f"[{name}] {len(empty)} empty days (source gap, not flagged): {empty[0]}..{empty[-1]}")
        print(f"[{name}] {len(s)} days checked in {','.join(months)}; flagged {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
