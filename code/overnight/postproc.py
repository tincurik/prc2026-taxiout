# SPDX-License-Identifier: GPL-3.0-or-later
"""LIRF long-delay post-processing: cell-level component rates override the model's mixture for rare rows.

LIRF DEPs with no flight match and a large MVT - SCHED fall into three recording regimes:
  imp    block = SCHED                       -> y = d_sched
  shift  block = actual time-of-day, but on the scheduled date (only when the flight crosses local midnight)
                                             -> y = 86400 + real taxi
  normal correct block                       -> y = real taxi (model regressor)
Cells: d_sched bin x crosses-midnight. Rates are estimated per CV fold on that fold's TRAIN months only (no val leak),
then pred = r_imp*(d_sched+delta) + r_shift*(86400+shift_taxi) + r_norm*reg for rows in cells with >= min_n train rows.

python postproc.py --src mix_v2_fs2 --id pp_v2
"""
import argparse
import json

import numpy as np
import pandas as pd
import polars as pl

from ocommon import FOLDS, OUTD, PROC, REP, rmse, save_result

BINS = [3600, 7200, 14400, 28800, 56000, 1e9]


def cells(df):
    cross = ((df.MVT_TIME_UTC_mvt + 3600) // 86400 - (df.SCHED_TIME_UTC_mvt + 3600) // 86400).clip(0, 1)
    b = pd.cut(df.d_sched, BINS, labels=False)
    ok = (df.apt == "LIRF") & (df.has_flt == 0) & (df.d_sched > BINS[0])
    return np.where(ok, cross.astype(int).astype(str) + "_" + b.fillna(-1).astype(int).astype(str), "")


def load_meta():
    f = pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "apt", "has_flt", "d_sched", "month", "split"]).to_pandas()
    m = pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                                                             "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]).filter(pl.col("PHASE_mvt") == "DEP").drop("PHASE_mvt").to_pandas()
    d = f.merge(m, on="MVT_ID_mvt")
    d["cell"] = cells(d)
    y = d.TAXITIME_SEC_mvt
    d["kind"] = np.select([(d.BLOCK_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).abs() <= 30, y > 60000], ["imp", "shift"], "normal")
    return d


def rates(tr, min_n, max_n=10**9):
    g = tr[tr.cell != ""].groupby("cell")
    out = {}
    for c, x in g:
        if len(x) < min_n or len(x) >= max_n:
            continue
        out[c] = {"n": len(x), "imp": float((x.kind == "imp").mean()), "shift": float((x.kind == "shift").mean()),
                  "normal": float((x.kind == "normal").mean())}
    sh = tr[(tr.kind == "shift") & (tr.apt == "LIRF")]
    taxi = float(np.median(sh.TAXITIME_SEC_mvt - 86400)) if len(sh) else 1000.0
    return out, taxi


def apply(o, rt, taxi, delta, alpha=1.0, alpha_cross=1.0):
    pred = o.pred.to_numpy().copy()
    for c, r in rt.items():
        m = (o.cell == c).to_numpy()
        if not m.any():
            continue
        ov = r["imp"] * (o.d_sched.to_numpy()[m] + delta) + r["shift"] * (86400 + taxi) + r["normal"] * o.reg.to_numpy()[m]
        al = alpha_cross if c.startswith("1_") else alpha
        pred[m] = al * ov + (1 - al) * pred[m]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--min_n", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the cell override for non-crossing cells")
    ap.add_argument("--max_n", type=int, default=100, help="cells with this many train rows or more are left to the model")
    a = ap.parse_args()
    d = load_meta()
    cfg = json.loads((REP / "results" / f"{a.src}.json").read_text())
    o = pd.read_parquet(OUTD / "oof" / f"{a.src}.parquet").merge(d[["MVT_ID_mvt", "cell", "d_sched", "apt", "kind"]], on="MVT_ID_mvt")
    trd = d[d.split == "train"]
    out = {"id": a.id, "model": "postproc", "src": a.src, "min_n": a.min_n, "max_n": a.max_n, "alpha": a.alpha, "folds": {}, "note": f"LIRF long-delay cell override on {a.src}"}
    preds = np.zeros(len(o))
    for fname, fd in FOLDS.items():
        rt, taxi = rates(trd[trd.month.isin(fd["train"])], a.min_n, a.max_n)
        fm = (o.fold == fname).to_numpy()
        delta = cfg["folds"][fname].get("delta", 0.0)
        p = apply(o[fm], rt, taxi, delta, a.alpha)
        preds[fm] = p
        yv = o.y.to_numpy()[fm]
        before = o.pred.to_numpy()[fm]
        cm = (o.cell[fm] != "").to_numpy()
        out["folds"][fname] = {"rmse": rmse(p, yv), "rmse_before": rmse(before, yv), "clean_rmse": rmse(p[yv <= 7200], yv[yv <= 7200]),
                               "cell_rows": int(cm.sum()), "cell_rmse_before": rmse(before[cm], yv[cm]), "cell_rmse_after": rmse(p[cm], yv[cm]),
                               "rates": rt, "shift_taxi": taxi,
                               "per_airport": {k: rmse(p[(o.apt[fm] == k).to_numpy()], yv[(o.apt[fm] == k).to_numpy()]) for k in sorted(o.apt.unique())}}
        # per-cell diagnostics
        diag = {}
        for c in sorted(set(o.cell[fm]) - {""}):
            m = (o.cell[fm] == c).to_numpy()
            diag[c] = {"n": int(m.sum()), "sse_before": float(((before[m] - yv[m]) ** 2).sum()), "sse_after": float(((p[m] - yv[m]) ** 2).sum()),
                       "applied": c in rt}
        out["folds"][fname]["cells"] = diag
    y = o.y.to_numpy()
    out["cv_rmse"] = rmse(preds, y)
    out["cv_rmse_before"] = rmse(o.pred, y)
    out["cv_clean_rmse"] = rmse(preds[y <= 7200], y[y <= 7200])
    out["cv_trim_rmse"] = rmse(preds[y <= 20000], y[y <= 20000])
    o2 = o[["MVT_ID_mvt", "fold", "y", "p_imp", "reg"]].copy()
    o2["pred"] = preds
    o2.to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    save_result(out)
    print(json.dumps({k: out[k] for k in ("cv_rmse_before", "cv_rmse", "cv_trim_rmse", "cv_clean_rmse")}))
    for f, v in out["folds"].items():
        print(f, round(v["rmse_before"], 2), "->", round(v["rmse"], 2), "cells:")
        for c, x in v["cells"].items():
            print("   ", c, x["n"], f"{x['sse_before']:.3g} -> {x['sse_after']:.3g}", "applied" if x["applied"] else "")
        print("   rates:", {c: (r["n"], round(r["imp"], 2), round(r["shift"], 2)) for c, r in v["rates"].items()})


if __name__ == "__main__":
    main()
