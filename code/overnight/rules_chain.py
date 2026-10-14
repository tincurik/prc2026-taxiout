# SPDX-License-Identifier: GPL-3.0-or-later
"""Gate rules inside the full chain: blend mixture -> gate rules -> postproc (a0.5) -> clip (test) -> clockfix R1-R3(b).

Rule "LOBT cliff": BLOCK = SCHED never happens once the flight plan's LOBT is more than CLIFF s after SCHED (2025 train:
0 copies in 66,025 rows above 3,600 s; 69 in the 3,540-3,600 s bin just below, at all airports). p_imp = 0 there.
Rule "LOBT window": |BLOCK - LOBT| <= 3,606 s in all 2,062,577 train departures with a flight plan (flights were attached
only inside that window), so y is clipped to [MVT - LOBT - W, MVT - LOBT + W] after clockfix (--window W, 0 = off).
Optional gate2 q stacking (--q_cache, --wl, --wo) on delayed rows min_ds < d_sched <= max_ds, applied before the rules.

python rules_chain.py --id rc_v1 --cliff 3600 --nocross 1 --test_raw final_blend_v20
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import polars as pl

import clockfix
import postproc
from ocommon import FOLDS, OUTD, PROC, REP, ROOT, rmse, save_result, TEMPLATE


def chain(o, g, delta_by_row, rates_by_fold, c, pp_alpha, nocross, clip=False, window=0.0):
    tmp = o.copy()
    tmp["pred"] = g * (tmp.d_sched + delta_by_row) + (1 - g) * tmp.reg
    out = tmp.pred.to_numpy().copy()
    for f, (rt, taxi, delta) in rates_by_fold.items():
        fm = (tmp.fold == f).to_numpy()
        out[fm] = postproc.apply(tmp[fm], rt, taxi, delta, pp_alpha)
    if clip:
        out = np.clip(out, 0, None)
    pred, *_ = clockfix.apply(tmp[["MVT_ID_mvt"]], out, c, True, True, nocross)
    if window:
        ml = tmp.ml.to_numpy()
        pred = np.where(np.isfinite(ml), np.clip(pred, ml - window, ml + window), pred)
    return pred


def gate(df, p, a, q=None):
    g = p.copy()
    if q is not None:
        rng = ((df.d_sched > a.min_ds) & (df.d_sched <= a.max_ds) & df.q.notna()).to_numpy()
        w = np.where((df.apt == "LIRF").to_numpy(), a.wl, a.wo) * rng
        g = g + w * (df.q.fillna(0).to_numpy() - g)
    if a.witness:
        g[(df.w_sl.abs() <= a.cliff).to_numpy()] = 0.0
    cliff = ((df.ls > a.cliff) | (df.ls < -a.cliff)).to_numpy() if a.sym else (df.ls > a.cliff).to_numpy()
    g[cliff] = 0.0
    return g, int(cliff.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--blend", default="blend_v20_cf")
    ap.add_argument("--test_raw", default="")
    ap.add_argument("--ref", default="clk3", help="prefix of the earlier chain files to sanity-check the base chain against")
    ap.add_argument("--cliff", type=float, default=3600)
    ap.add_argument("--nocross", type=int, default=1)
    ap.add_argument("--window", type=float, default=0.0)
    ap.add_argument("--witness", type=int, default=0, help="no-plan rows whose ARR-row witness LOBT is within cliff of SCHED: p_imp = 0")
    ap.add_argument("--sym", type=int, default=0, help="cliff on both sides: |LOBT - SCHED| > cliff")
    ap.add_argument("--q_cache", default="")
    ap.add_argument("--wl", type=float, default=0.0)
    ap.add_argument("--wo", type=float, default=0.0)
    ap.add_argument("--min_ds", type=float, default=900)
    ap.add_argument("--max_ds", type=float, default=20000)
    ap.add_argument("--pp_alpha", type=float, default=0.5)
    a = ap.parse_args()
    ls = (pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "LOBT_flt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"])
          .filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", ls=(pl.col("LOBT_flt") - pl.col("SCHED_TIME_UTC_mvt")).cast(pl.Float64),
                                                         ml=(pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).cast(pl.Float64)).to_pandas())
    wit = pd.read_parquet(OUTD / "cache" / "arr_witness.parquet")[["MVT_ID_mvt", "w_lobt"]]
    sch = pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "SCHED_TIME_UTC_mvt"]).filter(pl.col("PHASE_mvt") == "DEP").to_pandas()
    wit = wit.merge(sch, on="MVT_ID_mvt")
    ls = ls.merge(wit.assign(w_sl=wit.SCHED_TIME_UTC_mvt - wit.w_lobt)[["MVT_ID_mvt", "w_sl"]], on="MVT_ID_mvt", how="left")
    bres = json.loads((REP / "results" / f"{a.blend}.json").read_text())
    meta = postproc.load_meta()
    trd = meta[meta.split == "train"]
    q = None
    if a.q_cache:
        q = pd.read_parquet(OUTD / "cache" / f"{a.q_cache}_q.parquet")
    o = (pd.read_parquet(OUTD / "oof" / f"{a.blend}.parquet").merge(meta[["MVT_ID_mvt", "cell", "d_sched", "apt"]], on="MVT_ID_mvt")
         .merge(ls, on="MVT_ID_mvt", how="left"))
    if q is not None:
        o = o.merge(q[q.split == "oof"][["MVT_ID_mvt", "q"]], on="MVT_ID_mvt", how="left")
    fold_delta = {f: bres["folds"][f]["delta"] for f in FOLDS}
    delta_row = o.fold.map(fold_delta).to_numpy()
    rates = {f: (*postproc.rates(trd[trd.month.isin(fd["train"])], 8, 100), fold_delta[f]) for f, fd in FOLDS.items()}
    c = clockfix.clocks()
    y = o.y.to_numpy()
    p = o.p_imp.to_numpy()
    base = chain(o, p, delta_row, rates, c, a.pp_alpha, False)
    ref_path = OUTD / "oof" / f"{a.ref}_{a.blend.replace('_cf', '')}.parquet"
    base_chk = None  # a new blend has no clockfix reference file; the check only guards the earlier chains
    if ref_path.is_file():
        ref = pd.read_parquet(ref_path).set_index("MVT_ID_mvt").pred
        base_chk = float(np.max(np.abs(ref.loc[o.MVT_ID_mvt].to_numpy() - base)))
    g, n_cliff = gate(o, p, a, None if q is None else True)
    pred = chain(o, g, delta_row, rates, c, a.pp_alpha, bool(a.nocross), window=a.window)
    res = {"id": a.id, "model": "rules_chain", "blend": a.blend, "cliff": a.cliff, "nocross": a.nocross, "window": a.window, "sym": a.sym, "witness": a.witness, "q_cache": a.q_cache, "wl": a.wl, "wo": a.wo,
           "n_cliff_oof": n_cliff, "cv_rmse_base_chain": rmse(base, y), "oof_base_chain_max_abs_diff": base_chk,
           "cv_rmse": rmse(pred, y), "cv_trim_rmse": rmse(pred[y <= 20000], y[y <= 20000]), "cv_clean_rmse": rmse(pred[y <= 7200], y[y <= 7200]), "folds": {},
           "note": f"{a.blend} chain with p_imp = 0 where LOBT - SCHED > {a.cliff}" + (f", gate2 q {a.q_cache} w LIRF {a.wl} other {a.wo}" if a.q_cache else "")
                   + (", clockfix rule 3b" if a.nocross else "") + (f", clip to MVT-LOBT +- {a.window}" if a.window else "")}
    for f in FOLDS:
        k = (o.fold == f).to_numpy()
        res["folds"][f] = {"rmse": rmse(pred[k], y[k]), "base_rmse": rmse(base[k], y[k]), "clean_rmse": rmse(pred[k & (y <= 7200)], y[k & (y <= 7200)]), "n": int(k.sum())}
    pd.DataFrame({"MVT_ID_mvt": o.MVT_ID_mvt, "fold": o.fold, "y": y, "pred": pred, "p_imp": g, "reg": o.reg}).to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    if a.test_raw:
        t = (pd.read_parquet(OUTD / "test" / f"{a.test_raw}_raw.parquet")[["MVT_ID_mvt", "p_imp", "reg"]]
             .merge(meta[["MVT_ID_mvt", "cell", "d_sched", "apt"]], on="MVT_ID_mvt", how="left").merge(ls, on="MVT_ID_mvt", how="left"))
        if q is not None:
            t = t.merge(q[q.split == "rank"][["MVT_ID_mvt", "q"]], on="MVT_ID_mvt", how="left")
        delta = float(np.mean(trd.TAXITIME_SEC_mvt[trd.kind == "imp"] - trd.d_sched[trd.kind == "imp"]))
        rt, taxi = postproc.rates(trd, 8, 100)
        t["fold"] = "all"
        tp = t.p_imp.to_numpy()
        tg, n_cliff_t = gate(t, tp, a, None if q is None else True)
        tpred = chain(t, tg, np.full(len(t), delta), {"all": (rt, taxi, delta)}, c, a.pp_alpha, bool(a.nocross), clip=True, window=a.window)
        base_t = chain(t, tp, np.full(len(t), delta), {"all": (rt, taxi, delta)}, c, a.pp_alpha, False, clip=True)
        sub_path = OUTD / "test" / f"{a.ref}_{a.test_raw}.parquet"
        if sub_path.is_file():
            sub = pd.read_parquet(sub_path)
            chk = sub.MVT_ID_mvt.astype("int64").map(pd.Series(base_t, index=t.MVT_ID_mvt.to_numpy()))
            res["test_base_chain_max_abs_diff_vs_ref"] = float(np.nanmax(np.abs(chk.to_numpy() - sub.TAXITIME_SEC_mvt.to_numpy())))
        res.update(n_cliff_test=n_cliff_t, test_rms_change=float(np.sqrt(np.mean((tpred - base_t) ** 2))), test_n_changed=int((np.abs(tpred - base_t) > 1e-6).sum()))
        template = pd.read_parquet(TEMPLATE)
        template["TAXITIME_SEC_mvt"] = template.MVT_ID_mvt.astype("int64").map(pd.Series(tpred, index=t.MVT_ID_mvt.to_numpy())).to_numpy()
        path = OUTD / "test" / f"{a.id}.parquet"
        template.to_parquet(path, index=False)
        # gated p_imp and the regressor per test row, for stackers that correct the regressor branch
        pd.DataFrame({"MVT_ID_mvt": t.MVT_ID_mvt, "p_imp": tg, "reg": t.reg}).to_parquet(OUTD / "test" / f"{a.id}_parts.parquet", index=False)
        sys.path.insert(0, str(ROOT / "scripts"))
        from baseline import validate_submission
        validate_submission(path)
    save_result(res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
