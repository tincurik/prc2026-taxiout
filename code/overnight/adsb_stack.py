# SPDX-License-Identifier: GPL-3.0-or-later
"""ADS-B residual stacker on top of a full-chain result.

For departures with a matched ADS-B take-off (features_dep_adsb, |MVT - ADS-B take-off| <= 300 s), a GPU XGBoost
predicts the chain's residual y - pred from the ADS-B timings (MVT minus first seen / first move / push / roll
start), stop time, taxi distance, the chain's own prediction and clock gaps, and the airport. Cross-fitted between
the two CV folds (fit on Jul rows, predict Jan, and vice versa; early stopping on a day-split 15% of the fitting
fold); unmatched rows keep the chain's prediction. Test: one model on both folds' rows (iterations = mean best x 1.1)
added to the chain's test file. Residual target clipped to +-CLIP; the added correction is scaled by --w.

python adsb_stack.py --id ar_01_adsb_stack --base rc_v4_gate2_rules
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from ocommon import OUTD, PROC, ROOT, report, rmse, save_result, TEMPLATE, XGB_DEV, CB_TASK

AD = ["ad_to", "ad_seen", "ad_move", "ad_push", "ad_roll", "ad_stop_sec", "ad_dist_km", "ad_npts", "ad_maxgap",
      "ad_stand_seen", "ad_spells", "ad_match"]
AD2 = ["a2_seen", "a2_gs1", "a2_gs3", "a2_disp15", "a2_disp30", "a2_anchor", "a2_roll", "a2_hold_end", "a2_to", "a2_hold_sec",
       "a2_stops", "a2_stop_sec", "a2_path_m", "a2_maxgap", "a2_npts", "a2_first_gs", "a2_dwell", "a2_match", "a2_d_stand",
       "a2_parked", "a2_stand_vis"]
REL2 = ["a2_seen", "a2_gs1", "a2_disp15", "a2_anchor", "a2_roll", "a2_hold_end"]
FEAT = "adsb"
BAG_SEED = int(os.environ.get("BAG_SEED") or 0)   # seed offset of a deliberate redraw (full-chain seed bag); 0 = the shipped seeds
EXTRA = []  # extra feature files (features_dep_<name>.parquet), columns added to both models
EXTRA_COLS = []
RELCLK = False  # add ADS-B timings relative to AOBT_3 / LOBT / EOBT / SCHED (trees cannot form differences)
BASE = ["d_aobt", "d_eobt", "d_sched", "d_lobt"]
P = dict(tree_method="hist", device=XGB_DEV, objective="reg:squarederror", eta=0.03, max_depth=5, min_child_weight=50,
         subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, max_cat_to_onehot=1, eval_metric="rmse")


def frame(ids):
    f = pd.read_parquet(PROC / f"features_dep_{FEAT}.parquet")
    b = pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "apt", "MVT_TIME_UTC_mvt", *BASE]).to_pandas()
    d = ids.merge(f, on="MVT_ID_mvt", how="left").merge(b, on="MVT_ID_mvt", how="left")
    for name in EXTRA:
        e = pd.read_parquet(PROC / f"features_dep_{name}.parquet")
        e = e[["MVT_ID_mvt"] + [c for c in e.columns if c != "MVT_ID_mvt" and c not in d.columns]]
        for c in e.columns[1:]:
            if c not in EXTRA_COLS:
                EXTRA_COLS.append(c)
        d = d.merge(e, on="MVT_ID_mvt", how="left")
    for c in ("adsb_dir", "adsb_dir2", "adsb_dir3", "adsb_dir4", "adsb_dir5", "adsb_dir6", "adsb_dir7"):  # month-complete ADS-B models (adsb_direct.py): also their gap to the chain
        if c in d.columns:
            d[c + "_rel"] = d[c] - d["pred"]
            if c + "_rel" not in EXTRA_COLS:
                EXTRA_COLS.append(c + "_rel")
    d["apt"] = d["apt"].astype("category")
    d["hour"] = (d.MVT_TIME_UTC_mvt // 3600) % 24
    d["day"] = d.MVT_TIME_UTC_mvt // 86400
    if FEAT == "adsb":
        d["ok"] = (d.ad_match > 0) & (d.ad_to.abs() <= 300)
        for c in ["ad_seen", "ad_move", "ad_push", "ad_roll"]:  # timings relative to the chain's prediction
            d[c + "_rel"] = d[c] - d["pred"]
    else:
        d["ok"] = d.a2_seen.notna() & (d.a2_to.abs() <= 300)
        for c in REL2:
            d[c + "_rel"] = d[c] - d["pred"]
            if RELCLK:
                for b in BASE:
                    d[f"{c}_{b}"] = d[c] - d[b]
    return d


def feats(kind="ok"):
    if kind == "un":  # rows without an ADS-B match: chain prediction + clocks + extras only
        return EXTRA_COLS + BASE + ["pred", "apt", "hour"]
    if FEAT != "adsb":
        rc = [f"{c}_{b}" for c in REL2 for b in BASE] if RELCLK else []
        return AD2 + [c + "_rel" for c in REL2] + rc + EXTRA_COLS + BASE + ["pred", "apt", "hour"]
    return AD + [c + "_rel" for c in ["ad_seen", "ad_move", "ad_push", "ad_roll"]] + BASE + ["pred", "apt", "hour"]


def regbranch(o, apts):
    """pred0 = the chain's prediction; pred = what the stacker corrects (reg at the reg-branch airports); mult = scale of
    the correction; imp = training row is a BLOCK = SCHED copy (excluded from fitting at those airports)."""
    o = o.copy()
    o["pred0"] = o.pred
    o["mult"] = 1.0
    o["imp"] = False
    if apts:
        apt = pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "apt"]).to_pandas()
        o = o.merge(apt, on="MVT_ID_mvt", how="left")
        rb = o.apt.isin(apts).to_numpy() & o.reg.notna().to_numpy()
        o.loc[rb, "pred"] = o.loc[rb, "reg"]
        o.loc[rb, "mult"] = 1 - o.loc[rb, "p_imp"].clip(0, 1)
        if o.y.notna().any():
            mv = pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]).filter(pl.col("PHASE_mvt") == "DEP").to_pandas()
            imp = (mv.BLOCK_TIME_UTC_mvt - mv.SCHED_TIME_UTC_mvt).abs() <= 30
            o["imp"] = rb & o.MVT_ID_mvt.map(pd.Series(imp.to_numpy(), index=mv.MVT_ID_mvt)).fillna(False).to_numpy().astype(bool)
        o = o.drop(columns="apt")
    return o


MODEL = "xgb"
LP = dict(objective="regression", learning_rate=0.03, num_leaves=80, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8)  # LightGBM stacker (--model lgb), overridden by --params


def lgb_x(X):
    X = X.copy()
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
    return X


def lgb_fit(X, r, w, seed, n_it=None, Xva=None, rva=None, wva=None):
    import lightgbm as lgb
    p = dict(LP, seed=seed)
    dtr = lgb.Dataset(lgb_x(X), r, weight=w)
    if Xva is None:
        return lgb.train(p, dtr, n_it)
    dva = lgb.Dataset(lgb_x(Xva), rva, weight=wva, reference=dtr)
    return lgb.train(p, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])


def predict(m, X, it):
    if MODEL == "lgb":
        return m.predict(lgb_x(X), num_iteration=it)
    if MODEL == "cat":
        return m.predict(cat_pool(X), ntree_end=it)
    return m.predict(xgb.DMatrix(X, enable_categorical=True), iteration_range=(0, it))


def cat_pool(X, y=None, w=None):
    from catboost import Pool
    X = X.copy()
    cats = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
    for c in cats:
        X[c] = X[c].astype(str)
    return Pool(X, y, weight=w, cat_features=cats)


def cat_fit(X, r, w, seed, n_it=None, Xva=None, rva=None, wva=None):
    from catboost import CatBoostRegressor
    kw = dict(od_type="Iter", od_wait=200) if Xva is not None else {}
    m = CatBoostRegressor(iterations=n_it or 5000, depth=8, learning_rate=0.08, loss_function="RMSE", random_seed=seed,
                          task_type="CPU" if os.environ.get("STACK_CAT_CPU") else CB_TASK, verbose=0,
                          **({"thread_count": int(os.environ["STACK_CAT_CPU"])} if os.environ.get("STACK_CAT_CPU") else {}), **kw)
    m.fit(cat_pool(X, r, w), eval_set=cat_pool(Xva, rva, wva) if Xva is not None else None, use_best_model=Xva is not None)
    return m


def fit(d, clip, seed=0, kind="ok"):
    """Early-stopped fit on d (day split 85/15). Returns booster, best iteration."""
    P["seed"] = seed
    days = np.sort(d.day.unique())
    rng = np.random.default_rng(seed)
    va_days = set(rng.choice(days, max(1, int(len(days) * 0.15)), replace=False))
    va = d.day.isin(va_days).to_numpy()
    X = d[feats(kind)]
    r = np.clip(d.y - d.pred, -clip, clip)
    cw = d["cw"].to_numpy() if "cw" in d else np.ones(len(d))
    if MODEL == "cat":
        m = cat_fit(X[~va], r[~va], cw[~va], seed, Xva=X[va], rva=r[va], wva=cw[va])
        return m, m.get_best_iteration() + 1
    if MODEL == "lgb":
        m = lgb_fit(X[~va], r[~va], cw[~va], seed, Xva=X[va], rva=r[va], wva=cw[va])
        return m, max(m.best_iteration, 1)
    dtr = xgb.DMatrix(X[~va], r[~va], weight=cw[~va], enable_categorical=True)
    dva = xgb.DMatrix(X[va], r[va], weight=cw[va], enable_categorical=True)
    m = xgb.train(P, dtr, 5000, evals=[(dva, "va")], early_stopping_rounds=200, verbose_eval=False)
    return m, m.best_iteration + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--base", default="rc_v4_gate2_rules")
    ap.add_argument("--clip", type=float, default=1800)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--skip", default="", help="comma list of airports left uncorrected (still used for fitting)")
    ap.add_argument("--train_ymax", type=float, default=7200, help="fit only on rows with y <= this (copies/outliers out)")
    ap.add_argument("--params", default="{}", help="json overrides of the XGB params")
    ap.add_argument("--feat", default="adsb", help="adsb (v1) or adsb2 (features_adsb2.py)")
    ap.add_argument("--relclk", type=int, default=0, help="1 = add ADS-B timings minus each clock gap (d_aobt, d_lobt, ...)")
    ap.add_argument("--cv", default="month", help="month (fit Jul -> Jan and back) or dayK (day-block K-fold over both months)")
    ap.add_argument("--regbranch", default="", help="airports whose correction targets the regressor branch: fit on y - reg (non-imputed rows), add (1 - p_imp) x corr")
    ap.add_argument("--seeds", type=int, default=1, help="seed bag size (different day splits and subsamples)")
    ap.add_argument("--extra", default="", help="comma list of extra feature files features_dep_<name>.parquet")
    ap.add_argument("--covw", type=int, default=0, help="1 = weight training rows to the 2026 ADS-B coverage mix (covw_cv.py)")
    ap.add_argument("--unmatched", type=int, default=0, help="1 = also correct rows without an ADS-B match (extras + clocks + pred)")
    ap.add_argument("--model", default="xgb", choices=["xgb", "cat", "lgb"], help="cat = CatBoost GPU stacker (diversity twin), lgb = LightGBM CPU stacker (--params overrides LP)")
    ap.add_argument("--aux", default="", help="B1: comma list of OOF ids for the other 2025 months (month-fold members); their rows "
                    "join every training split and the test model, never scored")
    ap.add_argument("--aux_w", type=float, default=1.0, help="weight of aux rows")
    a = ap.parse_args()
    global FEAT, RELCLK, MODEL
    MODEL = a.model
    FEAT, RELCLK = a.feat, bool(a.relclk)
    EXTRA.extend(x for x in a.extra.split(",") if x)
    (LP if a.model == "lgb" else P).update(json.loads(a.params))
    run = f"adsb-stack-{a.id}"
    report(run, 0, 9, "loading", purpose="Does a same-flight ADS-B residual stacker on top of the chain cut RMSE on matched departures?")
    oof = pd.read_parquet(OUTD / "oof" / f"{a.base}.parquet")
    rb_apts = [x for x in a.regbranch.split(",") if x]
    oof = regbranch(oof, rb_apts)
    oof["aux"] = False
    if a.aux:  # B1: other-month rows (their base = a month-OOF member), training only
        ax = pd.concat([pd.read_parquet(OUTD / "oof" / f"{x}.parquet")[["MVT_ID_mvt", "y", "pred"]] for x in a.aux.split(",")])
        ax = ax[~ax.MVT_ID_mvt.isin(oof.MVT_ID_mvt)].assign(fold="aux", pred0=lambda t: t.pred, mult=1.0, imp=False, aux=True)
        oof = pd.concat([oof, ax.reindex(columns=oof.columns)], ignore_index=True)
    d = frame(oof[["MVT_ID_mvt", "fold", "y", "pred", "pred0", "mult", "imp", "aux"]])
    isaux = d.aux.to_numpy().astype(bool)
    if a.aux:
        d["cw"] = (d["cw"].to_numpy() if "cw" in d else 1.0) * np.where(isaux, a.aux_w, 1.0)
    if a.covw:  # fit on the 2026 ADS-B coverage mix (covw_cv.py importance weights)
        from covw_cv import cov_weights
        d["cw"] = 1.0
        d.loc[~d.aux, "cw"] = cov_weights(d.MVT_ID_mvt[~d.aux])  # aux (other-month) rows keep weight 1
    skip = [x for x in a.skip.split(",") if x]
    use = ~d.apt.isin(skip).to_numpy()
    new = d.pred0.to_numpy().copy()
    mult = d.mult.to_numpy()
    info, iters, iters_un = {}, [], []
    if a.cv == "month":
        splits = [(d.fold == "jul", d.fold == "jan", "jan"), (d.fold == "jan", d.fold == "jul", "jul")]
    else:  # day-block k-fold across both months (base preds are already month-OOF, so no base leak)
        k = int(a.cv[3:])
        days = np.sort(d.day.unique())
        dk = pd.Series(np.arange(len(days)) % k, index=days)
        fk = d.day.map(dk).to_numpy()
        splits = [(fk != i, fk == i, f"k{i}") for i in range(k)]
    for k, (trm, vam, name) in enumerate(splits, 1):
        trm, vam = np.asarray(trm) | isaux, np.asarray(vam) & ~isaux
        tr = d[trm & d.ok.to_numpy() & (d.y <= a.train_ymax).to_numpy() & ~d.imp.to_numpy()]
        va = vam & d.ok.to_numpy() & use
        corr = np.zeros(int(va.sum()))
        its = []
        for sd in range(a.seeds):
            m, it = fit(tr, a.clip, seed=sd)
            its.append(it)
            corr += predict(m, d.loc[va, feats()], it) / a.seeds
        iters += its
        new[va] += a.w * corr * mult[va]
        info[name] = dict(n_fit=len(tr), n_applied=int(va.sum()), iters=its, corr_rms=float(np.sqrt(np.mean(corr ** 2))))
        if a.unmatched:
            tru = d[trm & ~d.ok.to_numpy() & (d.y <= a.train_ymax).to_numpy() & ~d.imp.to_numpy()]
            vau = vam & ~d.ok.to_numpy() & use
            cu = np.zeros(int(vau.sum()))
            for sd in range(a.seeds):
                m, it = fit(tru, a.clip, seed=sd, kind="un")
                iters_un.append(it)
                cu += predict(m, d.loc[vau, feats("un")], it) / a.seeds
            new[vau] += a.w * cu * mult[vau]
            info[name].update(n_fit_un=len(tru), n_applied_un=int(vau.sum()), corr_rms_un=float(np.sqrt(np.mean(cu ** 2))))
        report(run, k, len(splits) + 2, f"{name}: fit {len(tr):,} rows, iters {its} -> {int(va.sum()):,} rows corrected")
    report(run, len(splits) + 1, len(splits) + 2, f"aux rows {int(isaux.sum()):,} (training only)")
    full_d = d
    d = d[~isaux].reset_index(drop=True)
    new = new[~isaux]
    y = d.y.to_numpy()
    res = {"aux": a.aux, "aux_w": a.aux_w, "n_aux": int(isaux.sum()), "id": a.id, "model": "adsb_stack", "base": a.base, "clip": a.clip, "w": a.w, "train_ymax": a.train_ymax, "skip": skip, "feat": a.feat, "relclk": a.relclk, "cv": a.cv, "seeds": a.seeds, "extra": a.extra, "unmatched": a.unmatched, "params": LP if a.model == "lgb" else P, "stack_model": a.model,
           "cv_rmse": rmse(new, y), "cv_trim_rmse": rmse(new[y <= 20000], y[y <= 20000]), "cv_clean_rmse": rmse(new[y <= 7200], y[y <= 7200]),
           "base_cv_rmse": rmse(d.pred0, y), "regbranch": a.regbranch, "folds": {}, "info": info, "per_airport": {}}
    for f in ("jan", "jul"):
        k = (d.fold == f).to_numpy()
        res["folds"][f] = {"rmse": rmse(new[k], y[k]), "base_rmse": rmse(d.pred0.to_numpy()[k], y[k]), "n": int(k.sum())}
    ok = d.ok.to_numpy()
    res["matched"] = {"n": int(ok.sum()), "base_rmse": rmse(d.pred0.to_numpy()[ok], y[ok]), "rmse": rmse(new[ok], y[ok])}
    for apt, g in d.groupby("apt", observed=True):
        k = g.index.to_numpy()
        res["per_airport"][apt] = {"base": rmse(d.pred0.to_numpy()[k], y[k]), "new": rmse(new[k], y[k]), "matched": float(g.ok.mean())}
    pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt, "fold": d.fold, "y": y, "pred": new}).to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)

    # test: one model on every matched OOF row
    report(run, 8, 9, "test model on both folds")
    tfile = OUTD / "test" / f"{a.base}.parquet"
    if tfile.is_file():
        full = full_d[full_d.ok & (full_d.y <= a.train_ymax) & ~full_d.imp]
        n_it = int(np.mean(iters) * 1.1)
        t = pd.read_parquet(tfile).rename(columns={"TAXITIME_SEC_mvt": "pred"})
        if rb_apts:
            t = t.merge(pd.read_parquet(OUTD / "test" / f"{a.base}_parts.parquet"), on="MVT_ID_mvt", how="left")
        t = regbranch(t.assign(y=np.nan), rb_apts)
        td = frame(t[["MVT_ID_mvt", "y", "pred", "pred0", "mult"]])
        tok = td.ok.to_numpy() & ~td.apt.isin(skip).to_numpy()
        tp = td.pred0.to_numpy().copy()
        tmult = td.mult.to_numpy()
        dfull = xgb.DMatrix(full[feats()], np.clip(full.y - full.pred, -a.clip, a.clip),
                            weight=full["cw"].to_numpy() if "cw" in full else None, enable_categorical=True)
        dt = xgb.DMatrix(td.loc[tok, feats()], enable_categorical=True)
        for sd in range(a.seeds):
            P["seed"] = 100 + BAG_SEED + sd
            if MODEL == "lgb":
                m = lgb_fit(full[feats()], np.clip(full.y - full.pred, -a.clip, a.clip), full["cw"].to_numpy() if "cw" in full else None, 100 + BAG_SEED + sd, n_it)
                tp[tok] += a.w * predict(m, td.loc[tok, feats()], n_it) / a.seeds * tmult[tok]
                continue
            if MODEL == "cat":
                m = cat_fit(full[feats()], np.clip(full.y - full.pred, -a.clip, a.clip), full["cw"].to_numpy() if "cw" in full else None, 100 + BAG_SEED + sd, n_it)
                tp[tok] += a.w * predict(m, td.loc[tok, feats()], n_it) / a.seeds * tmult[tok]
                continue
            m = xgb.train(P, dfull, n_it)
            tp[tok] += a.w * m.predict(dt) / a.seeds * tmult[tok]
        if a.unmatched:
            fu = full_d[~full_d.ok & (full_d.y <= a.train_ymax) & ~full_d.imp]
            tun = ~td.ok.to_numpy() & ~td.apt.isin(skip).to_numpy()
            dfu = xgb.DMatrix(fu[feats("un")], np.clip(fu.y - fu.pred, -a.clip, a.clip), enable_categorical=True)
            dtu = xgb.DMatrix(td.loc[tun, feats("un")], enable_categorical=True)
            for sd in range(a.seeds):
                P["seed"] = 100 + BAG_SEED + sd
                if MODEL == "lgb":
                    m = lgb_fit(fu[feats("un")], np.clip(fu.y - fu.pred, -a.clip, a.clip), fu["cw"].to_numpy() if "cw" in fu else None, 100 + BAG_SEED + sd, int(np.mean(iters_un) * 1.1))
                    tp[tun] += a.w * predict(m, td.loc[tun, feats("un")], int(np.mean(iters_un) * 1.1)) / a.seeds * tmult[tun]
                    continue
                m = xgb.train(P, dfu, int(np.mean(iters_un) * 1.1))
                tp[tun] += a.w * m.predict(dtu) / a.seeds * tmult[tun]
        tp = np.clip(tp, 0, None)
        template = pd.read_parquet(TEMPLATE)
        template["TAXITIME_SEC_mvt"] = template.MVT_ID_mvt.astype("int64").map(pd.Series(tp, index=td.MVT_ID_mvt.to_numpy())).to_numpy()
        path = OUTD / "test" / f"{a.id}.parquet"
        template.to_parquet(path, index=False)
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from baseline import validate_submission
        validate_submission(path)
        res.update(test_matched=float(tok.mean()), test_rms_change=float(np.sqrt(np.mean((tp - td.pred0.to_numpy()) ** 2))), test_iters=n_it)
    save_result(res)
    print(json.dumps({k: v for k, v in res.items() if k != "per_airport"}, indent=1))
    print(pd.DataFrame(res["per_airport"]).T.round(2).to_string())
    report(run, 9, 9, f"CV {res['cv_rmse']:.2f} (base {res['base_cv_rmse']:.2f}); matched {res['matched']['base_rmse']:.1f} -> {res['matched']['rmse']:.1f}", finished=True)


if __name__ == "__main__":
    main()
