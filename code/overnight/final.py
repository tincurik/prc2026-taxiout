# SPDX-License-Identifier: GPL-3.0-or-later
"""Refit a mixture config on all 2025 departures and predict the 2026 ranking departures.

Iterations = mean CV best iteration x scale (more data). Output: outputs/overnight/test/<id>.parquet in exact
template order, validated like scripts/baseline.py.

python final.py --from mix_v2_fs2 [--scale 1.15]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from mix import load

BAG = int(os.environ.get("BAG_SEED") or 0)   # seed offset of a deliberate redraw (full-chain seed bag); 0 = the shipped seeds
from ocommon import OUTD, REP, ROOT, report, TEMPLATE, CB_TASK, CB_GPU

sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--scale", type=float, default=1.15)
    ap.add_argument("--p_src", default="", help="classifier config/iterations from this run (e.g. clfcat_fs2)")
    ap.add_argument("--postproc", type=int, default=1)
    ap.add_argument("--reg_clip", type=float, default=None, help="override (older result jsons did not record reg_clip)")
    ap.add_argument("--served", type=int, default=0, help="regressor trained without LIRF rows and y > 80k (served-population "
                    "base); LIRF ranking rows keep the regressor of the untagged refit <src>_raw")
    ap.add_argument("--tag", default="", help="suffix for the output ids (<src>_<tag>), so variants never overwrite a refit")
    a = ap.parse_args()
    cfg = json.loads((REP / "results" / f"{a.src}.json").read_text())
    out_id = a.src + (f"_{a.tag}" if a.tag else "")
    run = f"final-{out_id}" + (f"-{a.p_src}" if a.p_src else "")
    t0 = time.time()
    report(run, 0, 4, "loading", purpose=f"Refit {a.src} on all 2025 and predict the test months")
    df, feats = load(set(filter(None, cfg.get("drop", "").split(","))), fs3=bool(cfg.get("fs3", 0)), fs4=bool(cfg.get("fs4", 0)),
                     fs3_name=cfg.get("fs3_name", "features_dep_fs3"), extra=tuple(filter(None, cfg.get("extra", "").split(","))))
    y = df["TAXITIME_SEC_mvt"].to_numpy(float)
    imp = (np.abs(df["BLOCK_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]) <= cfg.get("tol", 30)).to_numpy()
    tr = (df["split"] == "train").to_numpy()
    te = (df["split"] == "rank").to_numpy()
    if cfg.get("te"):
        from te import add_te
        yn = np.where(imp, np.nan, np.clip(y, 0, 3600))
        rs = np.where(imp | df["d_aobt"].isna().to_numpy(), np.nan, np.clip(y - df["d_aobt"].to_numpy(float), -1800, 1800))
        new = add_te(df, tr, te, {"y": np.where(tr, yn, np.nan), "res": np.where(tr, rs, np.nan), "imp": np.where(tr, imp.astype(float), np.nan)})
        for k, v in new.items():
            df[k] = v
        feats = feats + list(new)
    folds = cfg["folds"].values()
    clf_iter = int(np.mean([f["clf_iter"] for f in folds]) * a.scale)
    reg_iter = int(np.mean([f["reg_iter"] for f in folds]) * a.scale)
    clf_p = dict(cfg["clf_params"]); reg_p = dict(cfg["reg_params"])
    a.p_src = a.p_src or cfg.get("p_from", "")
    src_p = json.loads((REP / "results" / f"{a.p_src}.json").read_text()) if a.p_src else cfg
    clf_model = src_p.get("clf_model", "xgb")
    if a.p_src:
        clf_iter = int(np.mean([f["clf_iter"] for f in src_p["folds"].values()]) * a.scale)
    cats = [c for c in feats if str(df[c].dtype) == "category"]
    if clf_model == "cat":
        from catboost import CatBoostClassifier, Pool
        from mix import cat_prep
        cp = dict(task_type=CB_TASK, loss_function="Logloss", learning_rate=0.08, depth=8, border_count=254, verbose=500, **CB_GPU)
        cp.update(src_p.get("clf_params", {}) if isinstance(src_p.get("clf_params"), dict) and src_p.get("clf_model") == "cat" else {})
        cp = {k: v for k, v in cp.items() if k not in ("iterations", "od_type", "od_wait", "tree_method", "device", "objective", "eta", "max_depth",
                                                        "min_child_weight", "subsample", "colsample_bytree", "max_cat_to_onehot", "max_bin", "eval_metric")}
        if BAG:
            cp["random_seed"] = int(cp.get("random_seed", 0)) + BAG
        clf = CatBoostClassifier(iterations=clf_iter, **cp)
        clf.fit(Pool(cat_prep(df.loc[tr, feats], cats), imp[tr].astype(int), cat_features=cats))
        p = clf.predict_proba(Pool(cat_prep(df.loc[te, feats], cats), cat_features=cats))[:, 1]
        del clf
    else:
        d = xgb.QuantileDMatrix(df.loc[tr, feats], imp[tr], enable_categorical=True, max_bin=256)
        clf = xgb.train(dict(src_p["clf_params"], seed=int(src_p["clf_params"].get("seed", 0)) + BAG) if BAG else src_p["clf_params"], d, clf_iter)
        p = clf.predict(xgb.DMatrix(df.loc[te, feats], enable_categorical=True))
        del d, clf
    report(run, 1, 4, f"{clf_model} classifier fit ({clf_iter} iters); mean p_imp on ranking {p.mean():.4f}")
    rtr = tr & ~imp
    if a.served:
        rtr &= (df["apt"] != "LIRF").to_numpy() & (y <= 80000)
    base_tr = np.zeros(rtr.sum()); base_te = np.zeros(te.sum())
    if cfg.get("reg_target") == "resid":
        fill = np.nanmedian(df["d_aobt"])
        base_tr = df.loc[rtr, "d_aobt"].astype(float).clip(0, 3600).fillna(fill).to_numpy()
        base_te = df.loc[te, "d_aobt"].astype(float).clip(0, 3600).fillna(fill).to_numpy()
    elif cfg.get("reg_target") == "resid_lobt":
        from mix import lobt_base
        base_tr, base_te = lobt_base(df, rtr), lobt_base(df, te)
    ytr_r = y[rtr] - base_tr
    clip = a.reg_clip if a.reg_clip is not None else cfg.get("reg_clip", 0)
    if clip:
        ytr_r = np.clip(ytr_r, -clip, clip)
    if cfg.get("reg_model") == "cat":
        from catboost import CatBoostRegressor, Pool
        from mix import cat_prep
        rp = dict(task_type=CB_TASK, loss_function="RMSE", learning_rate=0.08, depth=8, border_count=254, l2_leaf_reg=3, verbose=1000, **CB_GPU)
        if BAG:
            rp["random_seed"] = int(rp.get("random_seed", 0)) + BAG
        # cat runs record the xgb default dict updated with the CatBoost JSON; keep only CatBoost keys
        rp.update({k: v for k, v in reg_p.items() if k in ("learning_rate", "depth", "l2_leaf_reg", "border_count", "random_strength",
                                                            "bagging_temperature", "one_hot_max_size", "max_ctr_complexity", "random_seed")})
        if cfg.get("by_apt"):  # per-airport experts, iterations = mean of that airport's fold best x scale
            Xtr, Xte = cat_prep(df.loc[rtr, feats], cats), cat_prep(df.loc[te, feats], cats)
            a_tr, a_te = df.loc[rtr, "apt"].to_numpy(), df.loc[te, "apt"].to_numpy()
            r = np.zeros(te.sum())
            for k in np.unique(a_te):
                n_it = int(np.mean([cfg["apt_iter"][f][str(k)] for f in cfg["apt_iter"]]) * a.scale)
                m = CatBoostRegressor(iterations=n_it, **rp)
                m.fit(Pool(Xtr[a_tr == k], ytr_r[a_tr == k], cat_features=cats))
                r[a_te == k] = m.predict(Pool(Xte[a_te == k], cat_features=cats))
                report(run, 2, 4, f"{k} expert refit ({n_it} iters)")
                del m
            r = r + base_te
        else:
            m = CatBoostRegressor(iterations=reg_iter, **rp)
            m.fit(Pool(cat_prep(df.loc[rtr, feats], cats), ytr_r, cat_features=cats))
            r = m.predict(Pool(cat_prep(df.loc[te, feats], cats), cat_features=cats)) + base_te
            del m
    else:
        d = xgb.QuantileDMatrix(df.loc[rtr, feats], ytr_r, enable_categorical=True, max_bin=256)
        reg = xgb.train(dict(reg_p, seed=int(reg_p.get("seed", 0)) + BAG) if BAG else reg_p, d, reg_iter)
        r = reg.predict(xgb.DMatrix(df.loc[te, feats], enable_categorical=True)) + base_te
        del d
    if a.served:
        lirf = (df.loc[te, "apt"] == "LIRF").to_numpy()
        orig = pd.read_parquet(OUTD / "test" / f"{a.src}_raw.parquet")[["MVT_ID_mvt", "reg"]].set_index("MVT_ID_mvt").reg
        r[lirf] = orig.loc[df.loc[te, "MVT_ID_mvt"].to_numpy()[lirf]].to_numpy()
    report(run, 3, 4, f"regressor fit ({reg_iter} iters)")
    delta = float(np.mean(y[tr & imp] - df["d_sched"].to_numpy(float)[tr & imp]))
    dsched = df.loc[te, "d_sched"].to_numpy(float)
    pred = p * (dsched + delta) + (1 - p) * r
    out = pd.DataFrame({"MVT_ID_mvt": df.loc[te, "MVT_ID_mvt"].to_numpy(), "pred": pred, "p_imp": p, "reg": r})
    if a.postproc:
        import postproc
        meta = postproc.load_meta()
        rt, taxi = postproc.rates(meta[meta.split == "train"], 8, 100)
        out = out.merge(meta[["MVT_ID_mvt", "cell", "d_sched"]], on="MVT_ID_mvt", how="left")
        out["pred_model"] = out["pred"]
        out["pred"] = postproc.apply(out, rt, taxi, delta, alpha=0.5)
        out = out.drop(columns=["cell", "d_sched"])
        report(run, 3, 4, f"postproc: {int((out.pred != out.pred_model).sum())} rows overridden")
    out["pred"] = out["pred"].clip(lower=0)
    pred = out["pred"].to_numpy()
    out.to_parquet(OUTD / "test" / f"{out_id}_raw.parquet", index=False)
    template = pd.read_parquet(TEMPLATE)
    m = out.set_index("MVT_ID_mvt")["pred"]
    template["TAXITIME_SEC_mvt"] = template["MVT_ID_mvt"].astype("int64").map(m).to_numpy()
    path = OUTD / "test" / f"{out_id}.parquet"
    template.to_parquet(path, index=False)
    from baseline import validate_submission
    validate_submission(path)
    report(run, 4, 4, f"wrote {path.name}: mean {pred.mean():.1f}s, p99 {np.quantile(pred, .99):.0f}s ({(time.time()-t0)/60:.1f} min)", finished=True)


if __name__ == "__main__":
    main()
