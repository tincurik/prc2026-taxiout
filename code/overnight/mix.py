# SPDX-License-Identifier: GPL-3.0-or-later
"""Two-stage mixture for the block-time imputation artifact.

  imp  = |BLOCK - SCHED| <= tol   (train labels only; derived from the masked block time)
  p    = P(imp | x)                GPU XGBoost classifier on all departures
  reg  = E[y | x, not imp]         GPU XGBoost regressor trained on non-imputed rows
  pred = p * (d_sched + delta) + (1 - p) * reg      (delta = mean y - d_sched among imputed rows ~ 0)

The decomposition E[y] = p E[y|imp] + (1-p) E[y|~imp] is exact for any partition, so this is consistent
at every airport; it just lets the regressor ignore the LIRF schedule-imputed rows.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from bench import CATS, NON_FEATURES
from ocommon import FOLDS, OUTD, PROC, report, rmse, save_result, XGB_DEV, CB_TASK, CB_GPU


def load(drop=(), fs3=False, fs4=False, fs3_name="features_dep_fs3", extra=()):
    f = pl.read_parquet(PROC / "features_dep.parquet")
    if fs3:
        f = f.join(pl.read_parquet(PROC / f"{fs3_name or 'features_dep_fs3'}.parquet"), on="MVT_ID_mvt", how="left")
    if fs4:
        f = f.join(pl.read_parquet(PROC / "features_dep_fs4.parquet"), on="MVT_ID_mvt", how="left")
    for name in extra:
        f = f.join(pl.read_parquet(PROC / f"features_dep_{name}.parquet"), on="MVT_ID_mvt", how="left")
    blk = pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "BLOCK_TIME_UTC_mvt"]).filter(
        pl.col("PHASE_mvt") == "DEP").drop("PHASE_mvt")
    df = f.join(blk, on="MVT_ID_mvt", how="left").to_pandas()
    for c in CATS:
        df[c] = df[c].astype("category")
    feats = [c for c in df.columns if c not in NON_FEATURES and c not in drop and c != "BLOCK_TIME_UTC_mvt"]
    return df, feats


def lobt_base(df, m):
    """MVT - LOBT on planned rows, else the clipped AOBT gap, else its median."""
    fill = np.nanmedian(df["d_aobt"])
    return df.loc[m, "d_lobt"].astype(float).fillna(df.loc[m, "d_aobt"].astype(float).clip(0, 3600)).fillna(fill).to_numpy()


def cat_prep(X, cats):
    X = X.copy()
    for c in cats:
        X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str).astype(object)
    return X


def tracked_rows(df):
    return (df["a2_seen"].notna() & (df["a2_to"].abs() <= 300)).to_numpy()


def hide_rates(df=None):
    """Per-airport share of tracked rows the final lost against 2025 in the same months (mirror of overnight/mix.py).
    Read from the final's own feature files (processed_final), whatever phase runs; labels are not read.
    h = max(0, 1 - sum_m n_final,m * trk_final,m / sum_m n_final,m * trk_2025,m) over months 1/2/6/7."""
    P = next(p for p in (PROC.parent / "processed_final",)
             if (p / "features_dep_adsb2fullrel.parquet").exists())
    f = pl.read_parquet(P / "features_dep.parquet", columns=["MVT_ID_mvt", "apt", "month", "split"])
    t = pl.read_parquet(P / "features_dep_adsb2fullrel.parquet", columns=["MVT_ID_mvt", "a2_seen", "a2_to"])
    d = f.join(t, on="MVT_ID_mvt", how="left").to_pandas()
    d["trk"] = tracked_rows(d)
    d = d[d.month.isin([1, 2, 6, 7])]
    g = d.groupby(["apt", "split", "month"]).trk.agg(["mean", "size"]).reset_index()
    r = g[g.split == "rank"].set_index(["apt", "month"])
    t25 = g[g.split == "train"].set_index(["apt", "month"])["mean"]
    r = r.assign(t25=t25.reindex(r.index).to_numpy())
    r = r[r.t25.notna()]
    r["a"] = r["mean"] * r["size"]; r["b"] = r["t25"] * r["size"]
    s = r.groupby("apt")[["a", "b"]].sum()
    return {str(k): float(max(0.0, 1 - v.a / v.b)) if v.b > 0 else 0.0 for k, v in s.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--tol", type=int, default=30)
    ap.add_argument("--drop", default="")
    ap.add_argument("--reg_params", default="{}")
    ap.add_argument("--clf_params", default="{}")
    ap.add_argument("--reg_target", default="raw", choices=["raw", "resid", "resid_lobt"])
    ap.add_argument("--note", default="")
    ap.add_argument("--te", type=int, default=0)
    ap.add_argument("--p_from", default="", help="reuse p_imp OOF from this run id instead of fitting the classifier")
    ap.add_argument("--reg_model", default="xgb", choices=["xgb", "cat"])
    ap.add_argument("--clf_model", default="xgb", choices=["xgb", "cat"])
    ap.add_argument("--save_p_only", type=int, default=0, help="stop after the classifier (writes p_imp OOF for --p_from reuse)")
    ap.add_argument("--reg_clip", type=float, default=0)
    ap.add_argument("--fs3", type=int, default=0)
    ap.add_argument("--fs4", type=int, default=0)
    ap.add_argument("--fs3_name", default="features_dep_fs3")
    ap.add_argument("--es_clip", type=float, default=0, help="clip the validation residual used for early stopping")
    ap.add_argument("--extra", default="", help="comma list of extra feature files data/processed/features_dep_<name>.parquet")
    ap.add_argument("--by_apt", type=int, default=0, help="CatBoost regressor per airport (early stopping per airport)")
    a = ap.parse_args()
    run = f"mix-{a.id}"
    NT = 2 * len(FOLDS) + 1  # one combined unit scale: clf + reg per fold, plus the finish
    t0 = time.time()
    report(run, 0, NT, "loading", purpose="Does an explicit P(block imputed from schedule) mixture beat a plain regressor?")
    df, feats = load(set(filter(None, a.drop.split(","))), fs3=bool(a.fs3), fs4=bool(a.fs4), fs3_name=a.fs3_name,
                     extra=tuple(filter(None, a.extra.split(","))))
    y = df["TAXITIME_SEC_mvt"].to_numpy(float)
    imp = (np.abs(df["BLOCK_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]) <= a.tol).to_numpy()
    dsched = df["d_sched"].to_numpy(float)
    trm = (df["split"] == "train").to_numpy()
    month = df["month"].to_numpy()
    reg_p = dict(tree_method="hist", device=XGB_DEV, objective="reg:squarederror", eta=0.05, max_depth=8, min_child_weight=20,
                 subsample=0.8, colsample_bytree=0.7, max_cat_to_onehot=1, max_bin=256, eval_metric="rmse")
    reg_p.update(json.loads(a.reg_params))
    clf_p = dict(tree_method="hist", device=XGB_DEV, objective="binary:logistic", eta=0.05, max_depth=8, min_child_weight=5,
                 subsample=0.8, colsample_bytree=0.7, max_cat_to_onehot=1, max_bin=256, eval_metric="logloss")
    clf_p.update(json.loads(a.clf_params))
    res = {"id": a.id, "model": f"mix_{a.clf_model}clf_{a.reg_model}reg", "clf_model": a.clf_model, "p_from": a.p_from, "reg_model": a.reg_model, "tol": a.tol, "reg_params": reg_p, "clf_params": clf_p, "note": a.note,
           "reg_target": a.reg_target, "reg_clip": a.reg_clip, "fs3": a.fs3, "fs4": a.fs4, "fs3_name": a.fs3_name, "drop": a.drop, "es_clip": a.es_clip, "extra": a.extra, "by_apt": a.by_apt, "folds": {}}
    oof = []
    step = 0
    for fname, fd in FOLDS.items():
        tr = trm & np.isin(month, fd["train"])
        va = trm & np.isin(month, fd["val"])
        if a.te:
            from te import add_te
            yn = np.where(imp, np.nan, np.clip(y, 0, 3600))
            rs = np.where(imp | df["d_aobt"].isna().to_numpy(), np.nan, np.clip(y - df["d_aobt"].to_numpy(float), -1800, 1800))
            new = add_te(df, tr, va, {"y": yn, "res": rs, "imp": imp.astype(float)})
            for k, v in new.items():
                df[k] = v
            feats_f = feats + list(new)
            report(run, step, NT, f"{fname} target encodings added ({len(new)})")
        else:
            feats_f = feats
        Xva = df.loc[va, feats_f]
        # classifier (or reuse OOF probabilities from an earlier run)
        if a.p_from:
            prev = pd.read_parquet(OUTD / "oof" / f"{a.p_from}.parquet")
            prev = prev[prev.fold == fname].set_index("MVT_ID_mvt")["p_imp"]
            p = df.loc[va, "MVT_ID_mvt"].map(prev).to_numpy()
            assert not np.isnan(p).any()
            clf_score, clf_iter = float("nan"), -1
        elif a.clf_model == "cat":
            from catboost import CatBoostClassifier, Pool
            cats = [c for c in feats_f if str(df[c].dtype) == "category"]
            cp = dict(task_type=CB_TASK, loss_function="Logloss", learning_rate=0.08, depth=8, iterations=5000,
                      od_type="Iter", od_wait=200, border_count=254, verbose=500, **CB_GPU)
            cp.update(json.loads(a.clf_params))
            clf = CatBoostClassifier(**cp)
            Xva_c = cat_prep(Xva, cats)
            clf.fit(Pool(cat_prep(df.loc[tr, feats_f], cats), imp[tr].astype(int), cat_features=cats),
                    eval_set=Pool(Xva_c, imp[va].astype(int), cat_features=cats), use_best_model=True)
            p = clf.predict_proba(Pool(Xva_c, cat_features=cats))[:, 1]
            pc = np.clip(p, 1e-7, 1 - 1e-7)
            clf_score = float(-np.mean(imp[va] * np.log(pc) + (1 - imp[va]) * np.log(1 - pc)))
            clf_iter = clf.get_best_iteration() + 1
            res.setdefault("clf_cat_importance", {})[fname] = sorted(zip(feats_f, clf.get_feature_importance().round(3).tolist()), key=lambda kv: -kv[1])[:40]
            del clf, Xva_c
        else:
            dtr = xgb.QuantileDMatrix(df.loc[tr, feats_f], imp[tr], enable_categorical=True, max_bin=256)
            dva = xgb.DMatrix(Xva, imp[va], enable_categorical=True)
            clf = xgb.train(clf_p, dtr, 3000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=500)
            p = clf.predict(dva, iteration_range=(0, clf.best_iteration + 1))
            clf_score, clf_iter = float(clf.best_score), clf.best_iteration + 1
            del dtr, dva
        step += 1; report(run, step, NT, f"{fname} clf logloss {clf_score:.4f} iter {clf_iter}")
        if a.save_p_only:
            yv = y[va]
            res["folds"][fname] = {"clf_logloss": clf_score, "clf_iter": int(clf_iter), "rmse": float("nan"), "delta": float(np.mean(y[tr & imp] - dsched[tr & imp]))}
            oof.append(pd.DataFrame({"MVT_ID_mvt": df.loc[va, "MVT_ID_mvt"].to_numpy(), "fold": fname, "y": yv, "pred": np.nan, "p_imp": p, "reg": np.nan}))
            step += 1
            continue
        # regressor on non-imputed rows
        rtr = tr & ~imp
        base_tr = np.zeros(rtr.sum()); base_va = np.zeros(va.sum())
        if a.reg_target == "resid":
            fill = np.nanmedian(df["d_aobt"])
            base_tr = df.loc[rtr, "d_aobt"].astype(float).clip(0, 3600).fillna(fill).to_numpy()
            base_va = df.loc[va, "d_aobt"].astype(float).clip(0, 3600).fillna(fill).to_numpy()
        elif a.reg_target == "resid_lobt":  # BLOCK - LOBT is bounded (+-3606) on planned rows: an anchor with no heavy tail
            base_tr, base_va = lobt_base(df, rtr), lobt_base(df, va)
        ytr_r = y[rtr] - base_tr
        if a.reg_clip:
            ytr_r = np.clip(ytr_r, -a.reg_clip, a.reg_clip)
        vr = ~imp[va]
        yva_r = y[va][vr] - base_va[vr]
        if a.es_clip:
            yva_r = np.clip(yva_r, -a.es_clip, a.es_clip)
        if a.reg_model == "xgb":
            dtr = xgb.QuantileDMatrix(df.loc[rtr, feats_f], ytr_r, enable_categorical=True, max_bin=256)
            dvr = xgb.DMatrix(Xva[vr], yva_r, enable_categorical=True)
            reg = xgb.train(reg_p, dtr, 4000, evals=[(dvr, "val")], early_stopping_rounds=200, verbose_eval=500)
            r = reg.predict(xgb.DMatrix(Xva, enable_categorical=True), iteration_range=(0, reg.best_iteration + 1)) + base_va
            reg_score, reg_iter = float(reg.best_score), reg.best_iteration + 1
            del dtr, dvr
        else:
            cat_p = json.loads(a.reg_params)
            # fit on non-imputed rows with early stopping on non-imputed val rows, then predict all val rows
            from catboost import CatBoostRegressor, Pool
            cp = dict(task_type=CB_TASK, loss_function="RMSE", learning_rate=0.08, depth=8, iterations=6000,
                      od_type="Iter", od_wait=300, border_count=254, l2_leaf_reg=3, verbose=1000, **CB_GPU)
            cp.update(cat_p)
            cats = [c for c in feats_f if str(df[c].dtype) == "category"]
            prep = lambda X: cat_prep(X, cats)
            Xva_p = prep(Xva)
            if a.by_apt:  # one regressor per airport, each with its own early stopping (experts for a 50/50 with the global member)
                r = np.zeros(va.sum())
                a_tr, a_va = df.loc[rtr, "apt"].to_numpy(), df.loc[va, "apt"].to_numpy()
                Xtr_p = prep(df.loc[rtr, feats_f])
                it = {}
                for k in np.unique(a_va):
                    mt, mv = a_tr == k, a_va == k
                    m = CatBoostRegressor(**cp)
                    m.fit(Pool(Xtr_p[mt], ytr_r[mt], cat_features=cats),
                          eval_set=Pool(Xva_p[mv & vr], yva_r[(mv & vr)[vr]], cat_features=cats), use_best_model=True)
                    r[mv] = m.predict(Pool(Xva_p[mv], cat_features=cats))
                    it[str(k)] = m.get_best_iteration() + 1
                    report(run, step, NT, f"{fname} {k} expert: iter {it[str(k)]}")
                    del m
                r = r + base_va
                reg_iter = int(np.mean(list(it.values())))
                res.setdefault("apt_iter", {})[fname] = it
                del Xtr_p
            else:
                m = CatBoostRegressor(**cp)
                m.fit(Pool(prep(df.loc[rtr, feats_f]), ytr_r, cat_features=cats),
                      eval_set=Pool(Xva_p[vr], yva_r, cat_features=cats), use_best_model=True)
                r = m.predict(Pool(Xva_p, cat_features=cats)) + base_va
                reg_iter = m.get_best_iteration() + 1
                res.setdefault("cat_importance", {})[fname] = sorted(zip(feats_f, m.get_feature_importance().round(3).tolist()), key=lambda kv: -kv[1])[:40]
                del m
            reg_score = rmse(r[vr], y[va][vr])
        step += 1; report(run, step, NT, f"{fname} reg(non-imp) rmse {reg_score:.2f} iter {reg_iter}")
        delta = float(np.mean(y[tr & imp] - dsched[tr & imp]))
        pred = p * (dsched[va] + delta) + (1 - p) * r
        yv = y[va]
        fr = {"rmse": rmse(pred, yv), "clean_rmse": rmse(pred[yv <= 7200], yv[yv <= 7200]),
              "trim_rmse": rmse(pred[yv <= 20000], yv[yv <= 20000]), "reg_only_rmse": rmse(r, yv),
              "clf_logloss": clf_score, "reg_iter": int(reg_iter), "clf_iter": int(clf_iter),
              "delta": delta}
        apt = df.loc[va, "apt"].to_numpy()
        fr["per_airport"] = {k: rmse(pred[apt == k], yv[apt == k]) for k in np.unique(apt)}
        res["folds"][fname] = fr
        oof.append(pd.DataFrame({"MVT_ID_mvt": df.loc[va, "MVT_ID_mvt"].to_numpy(), "fold": fname, "y": yv, "pred": pred, "p_imp": p, "reg": r}))
    o = pd.concat(oof)
    o.to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    if a.save_p_only:
        res["model"] = f"clf_{a.clf_model}"
        res["cv_logloss"] = {f: v["clf_logloss"] for f, v in res["folds"].items()}
        res["minutes"] = round((time.time() - t0) / 60, 2)
        save_result(res)
        report(run, NT, NT, f"classifier only: logloss {res['cv_logloss']}", finished=True)
        return
    res["cv_rmse"] = rmse(o.pred, o.y)
    res["cv_clean_rmse"] = rmse(o.pred[o.y <= 7200], o.y[o.y <= 7200])
    res["cv_trim_rmse"] = rmse(o.pred[o.y <= 20000], o.y[o.y <= 20000])
    res["minutes"] = round((time.time() - t0) / 60, 2)
    save_result(res)
    fs = ", ".join(f"{k} {v['rmse']:.2f}" for k, v in res["folds"].items())
    report(run, NT, NT, f"CV {res['cv_rmse']:.2f} ({fs}) trim {res['cv_trim_rmse']:.2f} clean {res['cv_clean_rmse']:.2f}", finished=True)
    print(json.dumps({k: res[k] for k in ("id", "cv_rmse", "cv_trim_rmse", "cv_clean_rmse", "minutes")}))


if __name__ == "__main__":
    main()

