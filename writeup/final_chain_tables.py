# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-month and per-airport scores of the final chain (the validation run of the shipped chain's steps, a separate build (33-45 s rms from the release draws), before the witness plan-window step and the bag; docs/results.md) on the twelve held-out months of 2025,
written to writeup/data/. Read-only: it scores predictions that exist, nothing is fitted.

Held-out rows: Jan/Jul 2025 out of fold (outputs/overnight_final/oof), the other ten months from the rehearsals (outputs/overnight_pseudo<N>/test against
data/pseudo<N>/truth.parquet), as overnight/month_weighted.py. Columns: rmse (all rows), rmse_le2h (label <= 7,200 s), bias_le2h (mean of prediction minus label on those rows).

  python writeup/final_chain_tables.py --root <full-run tree> --id <validation run id> [--out <dir>]

--root is the working tree of the full run (it holds outputs/ and data/; this repository does not), --id the name of the validation run's prediction files
(outputs/overnight_final/oof/<id>.parquet and outputs/overnight_pseudo<N>/test/<id>.parquet).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")   # the working tree of the full run (outputs/, data/), not this repository; set by --root
FOLDS = {"final": (1, 7), "pseudo2": (2, 6), "pseudo3": (3, 9), "pseudo4": (4, 10), "pseudo5": (5, 11), "pseudo6": (8, 12)}


def load(phase, i):
    f = pd.read_parquet(ROOT / f"data/processed_{phase}/features_dep.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "apt"])
    t = f.MVT_TIME_UTC_mvt
    sec = t.astype("int64") // 10**9 if np.issubdtype(t.dtype, np.datetime64) else t.astype("int64")
    f = pd.DataFrame({"MVT_ID_mvt": f.MVT_ID_mvt.astype("int64"), "apt": f.apt.to_numpy(), "month": pd.to_datetime(sec, unit="s").dt.month.to_numpy()})
    if phase == "final":
        d = pd.read_parquet(ROOT / f"outputs/overnight_final/oof/{i}.parquet", columns=["MVT_ID_mvt", "y", "pred"])
    else:
        d = pd.read_parquet(ROOT / f"data/{phase}/truth.parquet").rename(columns={"TAXITIME_SEC_mvt": "y"})
        o = pd.read_parquet(ROOT / f"outputs/overnight_{phase}/test/{i}.parquet", columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"]).rename(columns={"TAXITIME_SEC_mvt": "pred"})
        d = d.merge(o.assign(MVT_ID_mvt=o.MVT_ID_mvt.astype("int64")), on="MVT_ID_mvt", how="left")
    d["MVT_ID_mvt"] = d.MVT_ID_mvt.astype("int64")
    d = d.merge(f, on="MVT_ID_mvt", how="left")
    if d[["pred", "month"]].isna().any().any():
        raise SystemExit(f"{phase}: rows without a prediction or a month")
    d["phase"] = phase
    return d[d.month.isin(FOLDS[phase])]


def score(g):
    e = g.pred - g.y
    c = g.y <= 7200
    return pd.Series({"n": len(g), "rmse": np.sqrt((e**2).mean()), "rmse_le2h": np.sqrt((e[c]**2).mean()), "bias_le2h": e[c].mean()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="working tree of the full run (outputs/, data/)")
    ap.add_argument("--id", required=True, help="id of the validation run's prediction files")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data"))
    a = ap.parse_args()
    global ROOT
    ROOT = Path(a.root)
    d = pd.concat([load(p, a.id) for p in FOLDS], ignore_index=True)
    out = Path(a.out)
    m = d.groupby("month").apply(score, include_groups=False).reset_index()
    tot = score(d).to_frame().T.assign(month="all")
    m = pd.concat([m, tot], ignore_index=True)
    ap_ = d.groupby("apt").apply(score, include_groups=False).reset_index()
    ap_["share_sse"] = d.assign(se=(d.pred - d.y) ** 2).groupby("apt").se.sum().div((d.pred - d.y).pow(2).sum()).reindex(ap_.apt).to_numpy()
    ma = d.groupby(["month", "apt"]).apply(score, include_groups=False).reset_index()
    for df, name in ((m, "final_chain_month.csv"), (ap_, "final_chain_airport.csv"), (ma, "final_chain_month_airport.csv")):
        df.round(3).to_csv(out / name, index=False)
        print(name, len(df))
    print(m.round(2).to_string())
    print(ap_.round(3).to_string())


if __name__ == "__main__":
    main()
