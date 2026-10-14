# SPDX-License-Identifier: GPL-3.0-or-later
"""Blend regressor OOFs inside the mixture: pred = p_imp*(d_sched+delta) + (1-p_imp)*sum_i w_i reg_i.

p_imp and per-fold delta come from --p_from (a mix run). Weights: non-negative least squares on the pooled OOF of
NON-imputed rows (few weights, so the optimism of fitting on the validation folds is small; also reported:
equal-weight blend and per-fold cross-fitted weights, i.e. weights fit on Jan applied to Jul and vice versa).

python blend.py --id blend_v1 --p_from mix_v2_fs2 --regs mix_v2_fs2,mix_v5_cat_resid,dae_v1
"""
import argparse
import json

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import nnls

from ocommon import OUTD, PROC, REP, rmse, save_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--p_from", required=True)
    ap.add_argument("--regs", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--equal", type=int, default=0, help="1: equal weights everywhere (no fitting; the _cf file equals the main one)")
    ap.add_argument("--covw", type=int, default=0, help="1: fit the weights with coverage importance weights (covw_cv.py: 2026 ADS-B match mix)")
    a = ap.parse_args()
    regs = a.regs.split(",")
    base = pd.read_parquet(OUTD / "oof" / f"{a.p_from}.parquet")[["MVT_ID_mvt", "fold", "y", "p_imp"]]
    for r in regs:
        o = pd.read_parquet(OUTD / "oof" / f"{r}.parquet")[["MVT_ID_mvt", "reg"]].rename(columns={"reg": f"reg_{r}"})
        base = base.merge(o, on="MVT_ID_mvt", how="inner")
    meta = pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "d_sched", "SCHED_TIME_UTC_mvt", "apt"]).to_pandas()
    blk = pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "BLOCK_TIME_UTC_mvt"]).filter(
        pl.col("PHASE_mvt") == "DEP").drop("PHASE_mvt").to_pandas()
    d = base.merge(meta, on="MVT_ID_mvt").merge(blk, on="MVT_ID_mvt")
    cfg = json.loads((REP / "results" / f"{a.p_from}.json").read_text())
    tol = cfg.get("tol", 30)
    d["imp"] = (d.BLOCK_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).abs() <= tol
    d["delta"] = d.fold.map({f: v["delta"] for f, v in cfg["folds"].items()})
    R = d[[f"reg_{r}" for r in regs]].to_numpy(float)
    y = d.y.to_numpy(float)
    nimp = ~d.imp.to_numpy()

    def mix(w):
        reg = R @ w
        return d.p_imp * (d.d_sched + d.delta) + (1 - d.p_imp) * reg

    out = {"id": a.id, "model": "blend", "p_from": a.p_from, "regs": regs, "note": a.note, "tol": tol, "covw": a.covw}
    # individual
    out["single"] = {r: rmse(mix(np.eye(len(regs))[i]), y) for i, r in enumerate(regs)}
    # equal
    out["equal"] = rmse(mix(np.ones(len(regs)) / len(regs)), y)
    # nnls on non-imputed rows, clean target (y<=7200) so the tail does not set the weights
    m = nimp & (y <= 7200)
    sw = np.ones(len(d))
    if a.covw:
        from covw_cv import cov_weights
        sw = np.sqrt(cov_weights(d.MVT_ID_mvt))
    Rw, yw = R * sw[:, None], y * sw
    w, _ = nnls(Rw[m], yw[m])
    if a.equal:
        w = np.ones(len(regs)) / len(regs)
    out["weights"] = dict(zip(regs, w.round(4).tolist()))
    # cross-fitted weights
    pred_cf = np.zeros(len(d)); reg_cf = np.zeros(len(d))
    for f in d.fold.unique():
        fm = (d.fold == f).to_numpy()
        wf, _ = nnls(Rw[m & ~fm], yw[m & ~fm])
        if a.equal:
            wf = w
        pred_cf[fm] = mix(wf)[fm]; reg_cf[fm] = (R @ wf)[fm]
    pred = mix(w).to_numpy()
    out["cv_rmse"] = rmse(pred, y)
    out["cv_rmse_crossfit_weights"] = rmse(pred_cf, y)
    out["cv_clean_rmse_crossfit_weights"] = rmse(pred_cf[y <= 7200], y[y <= 7200])
    out["cv_trim_rmse_crossfit_weights"] = rmse(pred_cf[y <= 20000], y[y <= 20000])
    out["cv_clean_rmse"] = rmse(pred[y <= 7200], y[y <= 7200])
    out["cv_trim_rmse"] = rmse(pred[y <= 20000], y[y <= 20000])
    out["folds"] = {}
    for f in d.fold.unique():
        fm = (d.fold == f).to_numpy()
        out["folds"][f] = {"rmse": rmse(pred[fm], y[fm]), "clean_rmse": rmse(pred[fm & (y <= 7200)], y[fm & (y <= 7200)]),
                           "per_airport": {ap_: rmse(pred[fm & (d.apt == ap_).to_numpy()], y[fm & (d.apt == ap_).to_numpy()]) for ap_ in sorted(d.apt.unique())}}
    pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt, "fold": d.fold, "y": y, "pred": pred, "p_imp": d.p_imp, "reg": R @ w}).to_parquet(
        OUTD / "oof" / f"{a.id}.parquet", index=False)
    save_result(out)
    # cross-fitted twin (weights from the other fold) so post-processed comparisons are not optimistic
    pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt, "fold": d.fold, "y": y, "pred": pred_cf, "p_imp": d.p_imp, "reg": reg_cf}).to_parquet(
        OUTD / "oof" / f"{a.id}_cf.parquet", index=False)
    save_result({"id": f"{a.id}_cf", "model": "blend_crossfit", "p_from": a.p_from, "regs": regs, "note": f"{a.note} (cross-fitted weights)",
                 "cv_rmse": out["cv_rmse_crossfit_weights"], "cv_clean_rmse": out["cv_clean_rmse_crossfit_weights"],
                 "cv_trim_rmse": out["cv_trim_rmse_crossfit_weights"], "folds": {f: {"delta": cfg["folds"][f].get("delta", 0.0)} for f in cfg["folds"]}})
    print(json.dumps({k: v for k, v in out.items() if k != "folds"}, indent=1))


if __name__ == "__main__":
    main()
