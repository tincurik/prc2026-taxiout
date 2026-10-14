# SPDX-License-Identifier: GPL-3.0-or-later
"""Duplicated departure records: the NM-matched half of a same-flight pair carries a planned-time BLOCK.

Some take-offs appear twice in the APDF departures: same airport, same flight number (or callsign), MVT within
seconds, often different stands. Almost all are at LFPG (115 train / 17 test pairs). The half WITH an NM AOBT has a
label that tracks MVT - LOBT (corr 0.94; 76% above 2,400 s), the half without NM has an ordinary taxi (median 1,084 s).
So the matched half's BLOCK is LOBT-like, not an actual off-block, and the base model under-predicts it (OOF SE 92.9M
on 26 Jan/Jul rows vs 43.0M for the rule). The step replaces the matched half's prediction by
  pred + w * (a + b * (MVT - LOBT) - pred),
a, b fitted by least squares on the 2025 pair rows outside the scored month (CV) or on all of them (test).

  python dup_pair.py --id ar_120_dup --base ar_119_dlyh [--w 1] [--win 300]
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from ocommon import OUTD, PROC, ROOT, report, rmse, save_result


def pairs(win):
    m = pd.read_parquet(PROC / "movements.parquet", columns=[
        "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt", "LOBT_flt",
        "AOBT_3_flt", "TAXITIME_SEC_mvt", "split"])
    d = m[m.PHASE_mvt == "DEP"].copy()
    d["key"] = d.FLIGHT_mvt.fillna(d.CALLSIGN_flt)
    d = d[d.key.notna()].sort_values(["ADEP_mvt", "key", "MVT_TIME_UTC_mvt"])
    g = d.groupby(["ADEP_mvt", "key"]).MVT_TIME_UTC_mvt
    near = (g.diff().abs() <= win) | (g.diff(-1).abs() <= win)
    d = d[near & d.AOBT_3_flt.notna() & d.LOBT_flt.notna()].copy()
    d["dlo"] = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).astype(float)
    d["mo"] = pd.to_datetime(d.MVT_TIME_UTC_mvt, unit="s").dt.month
    return d


def fit(t):
    A = np.c_[np.ones(len(t)), t.dlo]
    c, *_ = np.linalg.lstsq(A, t.TAXITIME_SEC_mvt.to_numpy(float), rcond=None)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--win", type=float, default=300, help="max |MVT difference| (s) between the two records")
    a = ap.parse_args()
    run = f"dup-pair-{a.id}"
    report(run, 0, 2, "oof", purpose="Does the NM-matched half of a duplicated departure record carry a LOBT-like BLOCK?")
    P = pairs(a.win)
    tr = P[P.split == "train"]
    oof = pd.read_parquet(OUTD / "oof" / f"{a.base}.parquet")
    new = oof.pred.to_numpy().copy()
    y = oof.y.to_numpy()
    idx = pd.Series(np.arange(len(oof)), index=oof.MVT_ID_mvt.to_numpy())
    coefs = {}
    for fo, mo in (("jan", 1), ("jul", 7)):
        c = fit(tr[tr.mo != mo])
        coefs[fo] = c.round(4).tolist()
        r = tr[(tr.mo == mo) & tr.MVT_ID_mvt.isin(idx.index)]
        i = idx[r.MVT_ID_mvt].to_numpy()
        new[i] = new[i] + a.w * (c[0] + c[1] * r.dlo.to_numpy() - new[i])
    ch = new != oof.pred.to_numpy()
    res = {"id": a.id, "model": "dup_pair", "base": a.base, "w": a.w, "win": a.win,
           "cv_rmse": rmse(new, y), "cv_trim_rmse": rmse(new[y <= 20000], y[y <= 20000]),
           "cv_clean_rmse": rmse(new[y <= 7200], y[y <= 7200]), "base_cv_rmse": rmse(oof.pred, y),
           "n_changed": int(ch.sum()), "train_pairs": int(len(tr)), "train_by_apt": tr.ADEP_mvt.value_counts().to_dict(),
           "coef_cv": coefs, "se_base_M": float(((oof.pred.to_numpy()[ch] - y[ch]) ** 2).sum() / 1e6),
           "se_new_M": float(((new[ch] - y[ch]) ** 2).sum() / 1e6), "folds": {}}
    for fo in ("jan", "jul"):
        k = (oof.fold == fo).to_numpy()
        res["folds"][fo] = {"rmse": rmse(new[k], y[k]), "base_rmse": rmse(oof.pred.to_numpy()[k], y[k]), "n": int(k.sum())}
    oof.assign(pred=new)[["MVT_ID_mvt", "fold", "y", "pred"]].to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)

    report(run, 1, 2, "test")
    c = fit(tr)
    t = pd.read_parquet(OUTD / "test" / f"{a.base}.parquet")
    ti = pd.Series(np.arange(len(t)), index=t.MVT_ID_mvt.astype("int64").to_numpy())
    r = P[(P.split == "rank") & P.MVT_ID_mvt.astype("int64").isin(ti.index)]
    i = ti[r.MVT_ID_mvt.astype("int64")].to_numpy()
    p = t.TAXITIME_SEC_mvt.to_numpy(float).copy()
    old = p[i].copy()
    p[i] = old + a.w * (c[0] + c[1] * r.dlo.to_numpy() - old)
    t["TAXITIME_SEC_mvt"] = p
    path = OUTD / "test" / f"{a.id}.parquet"
    t.to_parquet(path, index=False)
    sys.path.insert(0, str(ROOT / "scripts"))
    from baseline import validate_submission
    validate_submission(path)
    res["test"] = {"n_changed": int(len(i)), "coef": c.round(4).tolist(), "by_apt": r.ADEP_mvt.value_counts().to_dict(),
                   "mean_change": float((p[i] - old).mean()), "sum_d2_over_N": float(((p[i] - old) ** 2).sum() / len(t))}
    save_result(res)
    print(json.dumps(res, indent=1))
    report(run, 2, 2, f"CV {res['cv_rmse']:.3f} (base {res['base_cv_rmse']:.3f})", finished=True)


if __name__ == "__main__":
    main()
