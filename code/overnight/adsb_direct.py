# SPDX-License-Identifier: GPL-3.0-or-later
"""Month-complete ADS-B model: a stacker feature learned from all 12 months of 2025, not only Jan/Jul.

The residual stacker (adsb_stack.py) only sees Jan+Jul 2025 matched rows (180 k). About 12 % of the test's matched
rows sit in regimes with almost no Jan/Jul analogue (LEMD Jan, LIRF Jan, EGLL Jul at 98 % coverage), while other 2025
months cover them. This fits an XGB taxi-time model on ADS-B + clock features over ALL matched 2025 rows:
  Jan-2025 rows <- model fitted on the other 11 months, Jul-2025 rows <- likewise, test rows <- model on all of 2025.
Also writes per-(airport, day) receiver-regime descriptors (median npts / maxgap, matched share); serve-time safe.
Output: data/processed/features_dep_adsbdir.parquet (MVT_ID_mvt, adsb_dir, rx_npts, rx_gap, rx_share), used through
  adsb_stack.py --extra adsbdir   (frame() adds adsb_dir_rel = adsb_dir - pred)

  python adsb_direct.py [--rounds 3000]
"""
import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from adsb_stack import AD2, BASE, REL2
from ocommon import PROC, REP, ROOT, report, rmse, XGB_DEV, CB_TASK

P = dict(tree_method="hist", device=XGB_DEV, objective="reg:squarederror", eta=0.05, max_depth=8, min_child_weight=50,
         subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, max_cat_to_onehot=1, eval_metric="rmse", max_bin=256)


CTX_CAT = ["apt_rwy", "apt_stand", "AIRCRAFT_TYPE_mvt", "airline"]
CTX_NUM = ["local_hour", "dow", "rwy_queue_at_mvt", "rwy_dep_during_taxi", "rwy_prev_10", "rwy_gap_prev", "rwy_nb_daobt_pm30"]
CTX = False
GEO_COLS = ["g_x0", "g_y0", "g_dxs", "g_dys", "g_bear", "g_sx", "g_sy"]  # first-heard position (adsb_geo.py): the taxi map
GEO2_COLS = [f"v{w}_{k}" for w in (60, 180) for k in ("dx", "dy", "sp", "gsm", "gsx", "n", "trs", "away")]  # first-heard motion (adsb_geo2.py)
GEO = 0
MODEL = "xgb"
SRC = "adsb2fullrel"


def load():
    f = pd.read_parquet(PROC / f"features_dep_{SRC}.parquet", columns=["MVT_ID_mvt", "a2_ok", *AD2])  # all 2025 months
    b = pd.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "apt", "split", "MVT_TIME_UTC_mvt",
                                                                "TAXITIME_SEC_mvt", *BASE, *((CTX_CAT + CTX_NUM) if CTX else [])])
    d = b.merge(f, on="MVT_ID_mvt", how="left")
    if GEO:
        d = d.merge(pd.read_parquet(PROC / "features_dep_adsbgeo.parquet"), on="MVT_ID_mvt", how="left")
    if GEO >= 2:
        d = d.merge(pd.read_parquet(PROC / "features_dep_adsbgeo2.parquet"), on="MVT_ID_mvt", how="left")
    d["ok"] = d.a2_ok.fillna(0).astype(bool)
    d["mo"] = pd.to_datetime(d.MVT_TIME_UTC_mvt, unit="s").dt.strftime("%y%m")
    d["day"] = d.MVT_TIME_UTC_mvt // 86400
    d["hour"] = (d.MVT_TIME_UTC_mvt // 3600) % 24
    g = d.groupby(["apt", "day"])
    d["rx_share"] = g.ok.transform("mean")
    okd = d[d.ok].groupby(["apt", "day"])
    d = d.merge(okd.a2_npts.median().rename("rx_npts").reset_index(), on=["apt", "day"], how="left")
    d = d.merge(okd.a2_maxgap.median().rename("rx_gap").reset_index(), on=["apt", "day"], how="left")
    for c in REL2:
        for bb in BASE:
            d[f"{c}_{bb}"] = d[c] - d[bb]
    d["apt"] = d["apt"].astype("category")
    if CTX:
        for c in CTX_CAT:
            d[c] = d[c].astype(str).astype("category")
    return d


def capify(d, caps):
    """Density cap (density_cap.py) on the test rows of a loaded frame: a2_npts / a2_maxgap clipped, rx_npts / rx_gap (per airport-day medians) recomputed from them."""
    import density_cap
    d = d.copy()
    t = (d.split != "train").to_numpy()
    n2, g2, nch = density_cap.clip(d.a2_npts, d.a2_maxgap, d.apt.astype(str), d.MVT_TIME_UTC_mvt, caps)
    d.loc[t, "a2_npts"], d.loc[t, "a2_maxgap"] = n2[t], g2[t]
    keys = d.apt.astype(str)
    okd = d[d.ok].assign(k=keys[d.ok]).groupby(["k", "day"])
    rx = pd.DataFrame({"rx_npts": okd.a2_npts.median(), "rx_gap": okd.a2_maxgap.median()}).reset_index()
    d = d.drop(columns=["rx_npts", "rx_gap"]).assign(k=keys).merge(rx, on=["k", "day"], how="left").drop(columns="k")
    return d, nch


def cols():
    base = AD2 + [f"{c}_{b}" for c in REL2 for b in BASE] + BASE + ["apt", "hour", "rx_share", "rx_npts", "rx_gap"]
    return base + (CTX_CAT + CTX_NUM if CTX else []) + (GEO_COLS if GEO else []) + (GEO2_COLS if GEO >= 2 else [])


def fit_predict(tr, te, rounds, seed):
    # inner early stopping on a day-block holdout of the training months (never on the target month)
    hold = (tr.day % 10) == seed % 10
    if MODEL == "cat":
        from catboost import CatBoostRegressor, Pool
        cats = [c for c in cols() if str(tr[c].dtype) == "category"]
        prep = lambda x: x[cols()].assign(**{c: x[c].astype(str) for c in cats})
        m = CatBoostRegressor(iterations=rounds, learning_rate=0.08, depth=8, l2_leaf_reg=5, loss_function="RMSE", task_type=CB_TASK,
                              random_seed=seed, od_type="Iter", od_wait=100, verbose=0, border_count=254)
        m.fit(Pool(prep(tr[~hold]), tr.y[~hold], cat_features=cats), eval_set=Pool(prep(tr[hold]), tr.y[hold], cat_features=cats), use_best_model=True)
        return m.predict(Pool(prep(te), cat_features=cats)), m.get_best_iteration() + 1
    p = dict(P, seed=seed)
    dtr = xgb.QuantileDMatrix(tr.loc[~hold, cols()], tr.y[~hold], enable_categorical=True, max_bin=256)
    dho = xgb.DMatrix(tr.loc[hold, cols()], tr.y[hold], enable_categorical=True)
    m = xgb.train(p, dtr, rounds, evals=[(dho, "ho")], early_stopping_rounds=100, verbose_eval=False)
    it = m.best_iteration + 1
    return m.predict(xgb.DMatrix(te[cols()], enable_categorical=True), iteration_range=(0, it)), it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--ymax", type=float, default=7200)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=0, help="1 = add stand/runway/type/airline/queue context -> features_dep_adsbdir2 (adsb_dir2)")
    ap.add_argument("--model", default="xgb", help="cat = CatBoost GPU (with --geo 2 -> adsbdir5 / adsb_dir5)")
    ap.add_argument("--tag", default="", help="override output: features_dep_adsbdir<tag> / adsb_dir<tag>")
    ap.add_argument("--geo", type=int, default=0, help="1 = + first-heard position (taxi map) -> adsbdir3 (adsb_dir3); 2 = + first-heard motion -> adsbdir4 (adsb_dir4)")
    ap.add_argument("--cap", default="", help="density-cap A/B (overnight/density_cap.py json): fit once on all of 2025, predict the test rows uncapped (features_dep_adsbdir<tag>ctl) and capped "
                    "(features_dep_adsbdir<tag>cap); the Jan/Jul OOF part is skipped (the shipped adsbdir4 rows stay)")
    ap.add_argument("--src", default="adsb2fullrel", help="ADS-B event file features_dep_<src>.parquet (adsb2mrel = extended match)")
    a = ap.parse_args()
    global CTX, GEO, MODEL, SRC
    SRC = a.src
    CTX, GEO, MODEL = bool(a.ctx), a.geo, a.model
    col, fname = ("adsb_dir6", "adsbdir6") if CTX and GEO >= 2 else ("adsb_dir5", "adsbdir5") if MODEL == "cat" else ("adsb_dir4", "adsbdir4") if GEO >= 2 else ("adsb_dir3", "adsbdir3") if GEO else ("adsb_dir2", "adsbdir2") if CTX else ("adsb_dir", "adsbdir")
    if a.tag:
        col, fname = f"adsb_dir{a.tag}", f"adsbdir{a.tag}"
    run = f"adsb-direct{'-ctx' if CTX else ''}{f'-geo{GEO}' if GEO else ''}{'-cat' if MODEL == 'cat' else ''}"
    report(run, 0, 4, "loading", purpose="Does a month-complete ADS-B taxi model (all 12 months of 2025) give the Jan/Jul stacker regimes it never saw?")
    d = load()
    d["y"] = d.TAXITIME_SEC_mvt
    tr_all = d[(d.split == "train") & d.ok & (d.y <= a.ymax)]
    if a.cap:
        import density_cap
        dc, nch = capify(d, density_cap.load(a.cap))
        dc.index = d.index
        mt = ((d.split != "train") & d.ok).to_numpy()
        te, tec = d[mt], dc[mt]
        p0, p1 = np.zeros(len(te)), np.zeros(len(te))
        its = []
        for s in range(a.seeds):
            hold = (tr_all.day % 10) == s % 10   # fit_predict's own split and params; one model per seed predicts both frames
            dtr = xgb.QuantileDMatrix(tr_all.loc[~hold, cols()], tr_all.y[~hold], enable_categorical=True, max_bin=256)
            dho = xgb.DMatrix(tr_all.loc[hold, cols()], tr_all.y[hold], enable_categorical=True)
            m = xgb.train(dict(P, seed=s), dtr, a.rounds, evals=[(dho, "ho")], early_stopping_rounds=100, verbose_eval=False)
            it = m.best_iteration + 1
            its.append(it)
            p0 += m.predict(xgb.DMatrix(te[cols()], enable_categorical=True), iteration_range=(0, it)) / a.seeds
            p1 += m.predict(xgb.DMatrix(tec[cols()], enable_categorical=True), iteration_range=(0, it)) / a.seeds
            report(run, s + 1, a.seeds + 1, f"seed {s}: {it} iters")
        for suf, p in (("ctl", p0), ("cap", p1)):
            pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.to_numpy(), col: p}).to_parquet(PROC / f"features_dep_{fname}{suf}.parquet", index=False)
        old = pd.read_parquet(PROC / f"features_dep_{fname}.parquet", columns=["MVT_ID_mvt", col]).set_index("MVT_ID_mvt")[col]
        o = old.reindex(te.MVT_ID_mvt.to_numpy()).to_numpy()
        info = dict(test_rows=len(te), iters=its, rows_changed=nch, rms_cap_vs_ctl=float(np.sqrt(np.nanmean((p1 - p0) ** 2))),
                    rms_ctl_vs_shipped=float(np.sqrt(np.nanmean((p0 - o) ** 2))), mean_cap_minus_ctl=float(np.mean(p1 - p0)))
        (REP / f"{fname}_cap_ab.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info, indent=1))
        report(run, a.seeds + 1, a.seeds + 1, f"done: cap-vs-ctl rms {info['rms_cap_vs_ctl']:.1f}", finished=True)
        return
    out = pd.Series(np.nan, index=d.index)
    info = {}
    for k, mo in enumerate(["2501", "2507"], 1):
        te = d[(d.mo == mo) & d.ok & (d.split == "train")]
        p = np.zeros(len(te)); its = []
        for s in range(a.seeds):
            q, it = fit_predict(tr_all[tr_all.mo != mo], te, a.rounds, s)
            p += q / a.seeds; its.append(it)
        out.loc[te.index] = p
        y = te.y.to_numpy()
        info[mo] = dict(n=len(te), iters=its, rmse_all=rmse(p, y), rmse_le7200=rmse(p[y <= 7200], y[y <= 7200]))
        report(run, k, 4, f"{mo}: {len(te)} rows, rmse(y<=7200) {info[mo]['rmse_le7200']:.1f}, iters {its}")
    te = d[(d.split != "train") & d.ok]
    p = np.zeros(len(te))
    for s in range(a.seeds):
        q, _ = fit_predict(tr_all, te, a.rounds, s)
        p += q / a.seeds
    out.loc[te.index] = p
    info["test_n"] = len(te)
    d[col] = out
    keep = ["MVT_ID_mvt", col] + ([] if CTX or GEO else ["rx_share", "rx_npts", "rx_gap"])
    d[keep].to_parquet(PROC / f"features_dep_{fname}.parquet", index=False)
    (REP / f"{fname}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    report(run, 4, 4, f"done: jan {info['2501']['rmse_le7200']:.1f} jul {info['2507']['rmse_le7200']:.1f} test {len(te)}", finished=True)


if __name__ == "__main__":
    main()
