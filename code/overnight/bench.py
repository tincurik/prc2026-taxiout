# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark a model config on the season-matched folds (GPU). Saves OOF + result json.

python bench.py --id xgb_raw_v1 --model xgb --target raw [--drop colA,colB] [--params '{"max_depth":10}']
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import polars as pl

from ocommon import FOLDS, OUTD, PROC, report, rmse, save_result, XGB_DEV, CB_TASK

CATS = ["apt", "apt_rwy", "apt_stand", "stand_pre", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
        "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt", "ADES_mvt", "airline"]
NON_FEATURES = {"MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                "TAXITIME_SEC_mvt", "LOBT_flt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt", "IOBT_flt",
                "FLIGHT_RULE_flt", "AIRCRAFT_TYPE_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
                "STAND_mvt", "RUNWAY_mvt", "split", "month", "ob_est"}


def load(drop=()):
    df = pl.read_parquet(PROC / "features_dep.parquet").to_pandas()
    for c in CATS:
        df[c] = df[c].astype("category")
    feats = [c for c in df.columns if c not in NON_FEATURES and c not in drop]
    return df, feats


def make_target(df, kind):
    y = df["TAXITIME_SEC_mvt"].to_numpy(float)
    base = np.zeros(len(df))
    if kind == "resid":
        base = df["d_aobt"].astype(float).clip(0, 3600).fillna(df["d_aobt"].median()).to_numpy() - 0.0
    return y, base


def fit_predict(model, params, Xtr, ytr, Xva, yva, rounds=None):
    if model == "xgb":
        import xgboost as xgb
        p = dict(tree_method="hist", device=XGB_DEV, objective="reg:squarederror", eta=0.05, max_depth=8,
                 min_child_weight=20, subsample=0.8, colsample_bytree=0.7, max_cat_to_onehot=1, max_bin=256,
                 reg_lambda=1.0, eval_metric="rmse")
        p.update(params)
        nr = p.pop("num_boost_round", 4000)
        dtr = xgb.QuantileDMatrix(Xtr, ytr, enable_categorical=True, max_bin=p["max_bin"])
        dva = xgb.DMatrix(Xva, yva, enable_categorical=True)
        es = p.pop("early_stopping", 200)
        bst = xgb.train(p, dtr, num_boost_round=rounds or nr, evals=[(dva, "val")],
                        early_stopping_rounds=None if rounds else es, verbose_eval=500)
        best = rounds or bst.best_iteration + 1
        pred = bst.predict(dva, iteration_range=(0, best))
        gain = bst.get_score(importance_type="total_gain")
        return pred, best, gain
    if model == "cat":
        from catboost import CatBoostRegressor, Pool
        p = dict(task_type=CB_TASK, loss_function="RMSE", learning_rate=0.08, depth=8, iterations=6000,
                 od_type="Iter", od_wait=300, border_count=254, l2_leaf_reg=3, verbose=500)
        p.update(params)
        if rounds:
            p["iterations"] = rounds; p.pop("od_type"); p.pop("od_wait")
        cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
        def prep(X):
            X = X.copy()
            for c in cats:
                X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str).astype(object)
            return X
        tr = Pool(prep(Xtr), ytr, cat_features=cats); va = Pool(prep(Xva), yva, cat_features=cats)
        m = CatBoostRegressor(**p)
        m.fit(tr, eval_set=None if rounds else va, use_best_model=not rounds)
        best = rounds or m.get_best_iteration() + 1
        gain = dict(zip(Xtr.columns, m.get_feature_importance()))
        return m.predict(va), best, gain
    raise ValueError(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--target", default="raw", choices=["raw", "resid"])
    ap.add_argument("--clip", type=float, default=0, help="clip TRAIN target at this value (0=off); val never clipped")
    ap.add_argument("--drop", default="")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    params = json.loads(a.params)
    run = f"bench-{a.id}"
    t0 = time.time()
    report(run, 0, 3, "loading", purpose=f"Internal CV of {a.model}/{a.target} ({a.note or a.id})")
    df, feats = load(set(filter(None, a.drop.split(","))))
    y, base = make_target(df, a.target)
    train_mask = (df["split"] == "train").to_numpy()
    month = df["month"].to_numpy()
    res = {"id": a.id, "model": a.model, "target": a.target, "clip": a.clip, "params": params, "note": a.note,
           "n_features": len(feats), "folds": {}}
    oof = []
    gains = {}
    for i, (fname, fd) in enumerate(FOLDS.items()):
        tr = train_mask & np.isin(month, fd["train"])
        va = train_mask & np.isin(month, fd["val"])
        ytr = y[tr] - base[tr]
        if a.clip:
            ytr = np.clip(ytr, -a.clip if a.target == "resid" else 0, a.clip)
        pred, best, gain = fit_predict(a.model, params, df.loc[tr, feats], ytr, df.loc[va, feats], y[va] - base[va])
        pred = pred + base[va]
        yv = y[va]
        clean = yv <= 7200
        fres = {"rmse": rmse(pred, yv), "clean_rmse": rmse(pred[clean], yv[clean]), "best_iter": int(best),
                "n": int(va.sum()), "mse": float(np.mean((pred - yv) ** 2))}
        fres["per_airport"] = {ap_: rmse(pred[(df.loc[va, "apt"] == ap_).to_numpy()], yv[(df.loc[va, "apt"] == ap_).to_numpy()])
                               for ap_ in sorted(df["apt"].cat.categories)}
        res["folds"][fname] = fres
        for k, v in gain.items():
            gains[k] = gains.get(k, 0) + float(v)
        oof.append(pd.DataFrame({"MVT_ID_mvt": df.loc[va, "MVT_ID_mvt"].to_numpy(), "fold": fname, "y": yv, "pred": pred}))
        report(run, i + 1, 3, f"fold {fname}: rmse {fres['rmse']:.2f} clean {fres['clean_rmse']:.2f} iter {best}")
    o = pd.concat(oof)
    o.to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    res["cv_rmse"] = rmse(o["pred"], o["y"])
    c = o["y"] <= 7200
    res["cv_clean_rmse"] = rmse(o.loc[c, "pred"], o.loc[c, "y"])
    c = o["y"] <= 20000
    res["cv_trim_rmse"] = rmse(o.loc[c, "pred"], o.loc[c, "y"])
    tot = sum(gains.values()) or 1
    res["top_features"] = sorted(((k, round(v / tot, 4)) for k, v in gains.items()), key=lambda kv: -kv[1])[:40]
    res["minutes"] = round((time.time() - t0) / 60, 2)
    save_result(res)
    report(run, 3, 3, f"CV {res['cv_rmse']:.2f} (jan {res['folds']['jan']['rmse']:.2f}, jul {res['folds']['jul']['rmse']:.2f}) clean {res['cv_clean_rmse']:.2f}", finished=True)
    print(json.dumps({k: res[k] for k in ("id", "cv_rmse", "cv_clean_rmse", "minutes")}))


if __name__ == "__main__":
    main()
