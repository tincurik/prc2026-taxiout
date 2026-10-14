# SPDX-License-Identifier: GPL-3.0-or-later
"""LIRF BEL/AEZ near-schedule no-plan quirk as a post-step on a stacked chain output.

LIRF departures of BEL (Brussels) and AEZ (Aer Lingus Regional) with no AOBT_3 and MVT - SCHED < 3600 carry an almost
constant label: 17 of the 20 in 2025 lie in 7.7-8.3 k s (the BLOCK stamp sits ~50-80 min before the inbound's own
arrival, so it is not a real off-block event), over 9 different months; the models predict ~3.5 k there.
The rule sets these rows to the mean 2025 label of the group (MSE-optimal, keeps the three low outliers in).
OOF: the Jan / Jul rows get the mean of the other 11 months. Test (4 rows): the mean of all 2025.

  python lirf_bel.py --id ar_55_bel_on53 --base ar_53_lirf_far_w05_ok_on18
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from ocommon import OUTD, PROC, ROOT, report, rmse, save_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--airlines", default="BEL,AEZ")
    ap.add_argument("--dmin", type=float, default=-1e9, help="lower MVT - SCHED bound (0: skip early departures, none in 2025 train)")
    a = ap.parse_args()
    run = f"lirf-bel-{a.id}"
    report(run, 0, 2, "rows", purpose="Does a constant-label rule for LIRF BEL/AEZ near-schedule no-plan departures help?")
    f = pd.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "apt", "airline", "d_aobt", "d_sched", "split",
                                                                "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
    f = f[(f.apt == "LIRF") & f.d_aobt.isna() & (f.d_sched < 3600) & (f.d_sched >= a.dmin) & f.airline.isin(a.airlines.split(","))].copy()
    f["mo"] = pd.to_datetime(f.MVT_TIME_UTC_mvt, unit="s").dt.strftime("%y%m")
    tr = f[f.split == "train"]

    oof = pd.read_parquet(OUTD / "oof" / f"{a.base}.parquet")
    new = oof.pred.to_numpy().copy()
    idx = pd.Series(np.arange(len(oof)), index=oof.MVT_ID_mvt)
    info = {}
    for mo in ("2501", "2507"):
        te = tr[tr.mo == mo]
        c = float(tr[tr.mo != mo].TAXITIME_SEC_mvt.mean())
        new[idx.loc[te.MVT_ID_mvt].to_numpy()] = c
        info[mo] = dict(n=len(te), const=c)
    y = oof.y.to_numpy()
    res = {"id": a.id, "model": "lirf_bel", "base": a.base, "airlines": a.airlines, "dmin": a.dmin, "n_train": len(tr),
           "cv_rmse": rmse(new, y), "cv_trim_rmse": rmse(new[y <= 20000], y[y <= 20000]),
           "cv_clean_rmse": rmse(new[y <= 7200], y[y <= 7200]), "base_cv_rmse": rmse(oof.pred, y), "info": info, "folds": {}}
    for fo in ("jan", "jul"):
        k = (oof.fold == fo).to_numpy()
        res["folds"][fo] = {"rmse": rmse(new[k], y[k]), "base_rmse": rmse(oof.pred.to_numpy()[k], y[k]), "n": int(k.sum())}
    oof.assign(pred=new)[["MVT_ID_mvt", "fold", "y", "pred"]].to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)

    report(run, 1, 2, "test")
    t = pd.read_parquet(OUTD / "test" / f"{a.base}.parquet")
    c = float(tr.TAXITIME_SEC_mvt.mean())
    te = f[f.split != "train"]
    ti = pd.Series(np.arange(len(t)), index=t.MVT_ID_mvt.astype("int64")).loc[te.MVT_ID_mvt].to_numpy()
    before = t.TAXITIME_SEC_mvt.to_numpy()[ti].copy()
    t.loc[t.index[ti], "TAXITIME_SEC_mvt"] = c
    path = OUTD / "test" / f"{a.id}.parquet"
    t.to_parquet(path, index=False)
    sys.path.insert(0, str(ROOT / "scripts"))
    from baseline import validate_submission
    validate_submission(path)
    res["test"] = dict(n=len(te), const=c, before=[round(float(v)) for v in before])
    save_result(res)
    print(json.dumps(res, indent=1))
    report(run, 2, 2, f"CV {res['cv_rmse']:.2f} (base {res['base_cv_rmse']:.2f}); test {len(te)} rows -> {c:.0f}", finished=True)


if __name__ == "__main__":
    main()
