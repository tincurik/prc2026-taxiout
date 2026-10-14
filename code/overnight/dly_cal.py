# SPDX-License-Identifier: GPL-3.0-or-later
"""Daily PRC pre-departure delay as a day-level calibration of ordinary rows.

PRC's daily `DLY_ALL_PRE_2` (all-cause pre-departure delay, minutes, per airport and UTC day of take-off) equals
sum over that day's departures of max(BLOCK - SCHED, 0) = max(MVT - SCHED - y, 0): on 2025 it matches our labels with
corr 0.9997 and median ratio 1.000. Given predictions, the day's shift delta on the ordinary rows (pred < --hp) solves
  sum_ord max(MVT - SCHED - (pred + delta), 0) + sum_hedged max(MVT - SCHED - pred, 0) = 60 * DLY
(the hedged / monster rows are held at their prediction so they do not steer the shift). The shift is applied with
weight --w to ordinary rows that enter the sum (MVT - SCHED > pred). No training; the daily file covers 2026-01 to 2026-08.

  python dly_cal.py --id ar_114_dly --base ar_103_fin
  python dly_cal.py --id ar_129_dlyh --base ar_129_lplan --mode mul --hedge 1
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from ocommon import OUTD, PROC, ROOT, report, rmse, save_result

PRC = ROOT / "data/external/prc"


def solve(m, yh, S, fixed, mul=False, sh=1.0):
    lo, hi = (-3.0, 3.0) if mul else (-1500.0, 1500.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        if np.maximum(m - (yh * np.exp(mid * sh) if mul else yh + mid), 0).sum() + fixed > S:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def shift(d, hp, w, mul=False, shape=(0.0, 0.0)):
    """d: MVT_ID_mvt, apt, day, m (MVT - SCHED, s), pred, S (s). Returns the per-row additive change."""
    out = np.zeros(len(d))
    info = []
    ordm = d.pred.to_numpy() < hp
    for (a, dy), ix in d.groupby(["apt", "day"]).indices.items():
        S = d.S.to_numpy()[ix[0]]
        if not np.isfinite(S):
            continue
        m, yh, o = d.m.to_numpy()[ix], d.pred.to_numpy()[ix], ordm[ix]
        ok = np.isfinite(m)
        fixed = np.maximum(m[ok & ~o] - yh[ok & ~o], 0).sum()
        k = ok & o
        # mul: row i scales by exp(d * s_i), s_i = (pred/1000)^g * (1 + lam * clip((MVT-SCHED-pred)/3600, 0, 3)):
        # the day's error sits more on long and late rows (fitted w 0.40 at pred<600 s .. 0.93 at 1.8-3.6 k)
        sh = (np.clip(yh, 1, None) / 1000) ** shape[0] * (1 + shape[1] * np.clip((m - yh) / 3600, 0, 3))
        dl = solve(m[k], yh[k], S, fixed, mul, sh[k])
        tgt = k & (m > yh)
        new = yh[tgt] * np.exp(w * dl * sh[tgt]) if mul else yh[tgt] + w * dl
        out[ix[tgt]] = np.maximum(new, np.minimum(yh[tgt], 120.0)) - yh[tgt]
        info.append((a, str(dy), dl))
    return out, info


def day_z(d):
    """Robust z of each airport-day's raw residual (implied sum from the base predictions minus PRC), per airport."""
    c = np.maximum(d.m - d.pred, 0).fillna(0)
    g = d.assign(c=c).groupby(["apt", "day"]).agg(c=("c", "sum"), S=("S", "first"))
    g["R"] = (g.c - g.S) / 60
    med = g.groupby(level="apt").R.transform("median")
    mad = g.groupby(level="apt").R.transform(lambda x: 1.4826 * np.median(np.abs(x - x.median())))
    g["z"] = (g.R - med) / mad
    return d[["apt", "day"]].merge(g.z.reset_index(), on=["apt", "day"], how="left").z.fillna(0).to_numpy()


def hedge_rows(d, thr):
    k = (d.pred > thr) & (d.m > d.pred + 1200)
    return k.to_numpy()


def load(split):
    f = pd.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "apt", "split", "MVT_TIME_UTC_mvt",
                                                                 "SCHED_TIME_UTC_mvt"])
    f = f[(f.split == "train") == (split == "train")]
    mvt = pd.to_datetime(f.MVT_TIME_UTC_mvt, unit="s") if np.issubdtype(f.MVT_TIME_UTC_mvt.dtype, np.number) else pd.to_datetime(f.MVT_TIME_UTC_mvt)
    sch = pd.to_datetime(f.SCHED_TIME_UTC_mvt, unit="s") if np.issubdtype(f.SCHED_TIME_UTC_mvt.dtype, np.number) else pd.to_datetime(f.SCHED_TIME_UTC_mvt)
    f = f.assign(day=mvt.dt.date.astype(str), m=(mvt - sch).dt.total_seconds())[["MVT_ID_mvt", "apt", "day", "m"]]
    p = pd.concat([pd.read_csv(PRC / f"all_pre_departure_delays_{y}.csv", usecols=["FLT_DATE", "APT_ICAO", "DLY_ALL_PRE_2"])
                   for y in (2025, 2026)])
    p = p.rename(columns={"FLT_DATE": "day", "APT_ICAO": "apt"}).drop_duplicates(["apt", "day"])
    p["S"] = p.DLY_ALL_PRE_2 * 60.0
    return f.merge(p[["apt", "day", "S"]], on=["apt", "day"], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--hp", type=float, default=3600, help="rows with pred >= hp are held fixed (hedged / monster)")
    ap.add_argument("--w", type=float, default=0.6)
    ap.add_argument("--hedge", type=float, default=0, help="1 = also move hedged rows (pred > --hthr, MVT-SCHED > pred + "
                    "1200) by -b * clip(z, +-2) * pred, z = robust per-airport z of the day residual; b fitted on the OOF "
                    "(cross-fold for the CV, both folds for test). Low-z days: hedged rows are over-predicted (CV -0.37 alone)")
    ap.add_argument("--hthr", type=float, default=4000)
    ap.add_argument("--shape", default="0,0", help="mul only: g,lam of the row scale s_i (1,1: CV -0.81 vs plain mul -0.72)")
    ap.add_argument("--mode", default="add", choices=["add", "mul"], help="mul: the day factor scales predictions (CV -0.72 vs add -0.60)")
    a = ap.parse_args()
    run = f"dly-cal-{a.id}"
    report(run, 0, 2, "oof", purpose="Does the daily PRC pre-departure delay total calibrate each airport-day's ordinary rows?")

    oof = pd.read_parquet(OUTD / "oof" / f"{a.base}.parquet")
    d = oof[["MVT_ID_mvt", "pred"]].merge(load("train"), on="MVT_ID_mvt", how="left")
    assert len(d) == len(oof) and (d.MVT_ID_mvt.to_numpy() == oof.MVT_ID_mvt.to_numpy()).all()
    adj, _ = shift(d, a.hp, a.w, a.mode == "mul", tuple(float(x) for x in a.shape.split(",")))
    new = oof.pred.to_numpy() + adj
    y = oof.y.to_numpy()
    b_all = None
    if a.hedge:
        z = np.clip(day_z(d), -2, 2)
        k = hedge_rows(d, a.hthr)
        P = oof.pred.to_numpy()
        x, e = z * P, P - y
        fold = oof.fold.to_numpy()
        betas = {}
        for fo in ("jan", "jul"):
            tr, te = k & (fold != fo), k & (fold == fo)
            betas[fo] = float((x[tr] * e[tr]).sum() / (x[tr] ** 2).sum())
            new[te] = P[te] - betas[fo] * x[te]
        b_all = float((x[k] * e[k]).sum() / (x[k] ** 2).sum())
    res = {"id": a.id, "model": "dly_cal", "base": a.base, "hp": a.hp, "w": a.w, "mode": a.mode, "shape": a.shape, "hedge": a.hedge, "hthr": a.hthr,
           "cv_rmse": rmse(new, y), "cv_trim_rmse": rmse(new[y <= 20000], y[y <= 20000]),
           "cv_clean_rmse": rmse(new[y <= 7200], y[y <= 7200]), "base_cv_rmse": rmse(oof.pred, y),
           "n_shifted": int((adj != 0).sum()), "folds": {}, "per_airport": {}}
    for fo in ("jan", "jul"):
        k = (oof.fold == fo).to_numpy()
        res["folds"][fo] = {"rmse": rmse(new[k], y[k]), "base_rmse": rmse(oof.pred.to_numpy()[k], y[k]), "n": int(k.sum())}
    for ap_, k in d.groupby("apt").indices.items():
        res["per_airport"][ap_] = round(rmse(new[k], y[k]) - rmse(oof.pred.to_numpy()[k], y[k]), 3)
    oof.assign(pred=new)[["MVT_ID_mvt", "fold", "y", "pred"]].to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)

    report(run, 1, 2, "test")
    t = pd.read_parquet(OUTD / "test" / f"{a.base}.parquet")
    ids = t.MVT_ID_mvt.astype("int64")
    d = pd.DataFrame({"MVT_ID_mvt": ids, "pred": t.TAXITIME_SEC_mvt.to_numpy()}).merge(load("test"), on="MVT_ID_mvt", how="left")
    assert len(d) == len(t)
    tadj, info = shift(d, a.hp, a.w, a.mode == "mul", tuple(float(x) for x in a.shape.split(",")))
    if a.hedge:
        zt = np.clip(day_z(d), -2, 2)
        kh = hedge_rows(d, a.hthr)
        P = d.pred.to_numpy()
        tadj[kh] = -b_all * zt[kh] * P[kh]
        res_h = {"b": b_all, "n": int(kh.sum()), "mean_change": float(tadj[kh].mean()),
                 "by_apt": d[kh].assign(ch=tadj[kh]).groupby("apt").ch.agg(["size", "mean"]).round(0).to_dict("index")}
    t["TAXITIME_SEC_mvt"] = t.TAXITIME_SEC_mvt.to_numpy() + tadj
    path = OUTD / "test" / f"{a.id}.parquet"
    t.to_parquet(path, index=False)
    sys.path.insert(0, str(ROOT / "scripts"))
    from baseline import validate_submission
    validate_submission(path)
    dl = pd.DataFrame(info, columns=["apt", "day", "delta"])
    if a.hedge:
        res["hedge"] = {**res_h, "betas_cv": betas}
    res["test"] = {"n_shifted": int((tadj != 0).sum()), "no_prc": int(d.S.isna().sum()),
                   "delta_by_apt": dl.groupby("apt").delta.agg(["mean", "std"]).round(1).to_dict("index"),
                   "mean_change": float(tadj.mean())}
    save_result(res)
    print(json.dumps(res, indent=1))
    report(run, 2, 2, f"CV {res['cv_rmse']:.3f} (base {res['base_cv_rmse']:.3f})", finished=True)


if __name__ == "__main__":
    main()
