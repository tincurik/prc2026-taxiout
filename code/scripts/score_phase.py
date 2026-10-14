# SPDX-License-Identifier: GPL-3.0-or-later
"""Score a rehearsal phase's prediction file against the held-out months' labels (data/<phase>/truth.parquet, written by
scripts/make_pseudo_final.py).  RMSE in seconds, overall and per airport.  Only for EC_PHASE=pseudoN; the test months have no labels here.

  EC_PHASE=pseudo2 python scripts/score_phase.py --source outputs/overnight_pseudo2/test/ar_129_prc2.parquet
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE = os.environ.get("EC_PHASE", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    a = ap.parse_args()
    if not PHASE.startswith("pseudo"):
        raise SystemExit("score_phase.py scores the rehearsal phases (EC_PHASE=pseudo2..6); the test months have no labels")
    truth = pd.read_parquet(ROOT / f"data/{PHASE}/truth.parquet").rename(columns={"TAXITIME_SEC_mvt": "y"})
    pred = pd.read_parquet(a.source).rename(columns={"TAXITIME_SEC_mvt": "p"})
    mv = pd.read_parquet(ROOT / f"data/processed_{PHASE}/movements.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"]).drop_duplicates("MVT_ID_mvt")
    d = truth.merge(pred, on="MVT_ID_mvt", how="left").merge(mv, on="MVT_ID_mvt", how="left")
    if d.p.isna().any():
        raise SystemExit(f"{int(d.p.isna().sum())} held-out departures have no prediction")
    e = (d.p - d.y) ** 2
    print(f"{PHASE}: {len(d):,} held-out departures, RMSE {np.sqrt(e.mean()):.2f} s; "
          f"constant-mean RMSE {np.sqrt(((d.y - d.y.mean()) ** 2).mean()):.2f} s")
    print(e.groupby(d.ADEP_mvt).mean().pow(0.5).round(1).to_string())


if __name__ == "__main__":
    main()
