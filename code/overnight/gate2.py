# SPDX-License-Identifier: GPL-3.0-or-later
"""Second copy gate for delayed departures at every airport, mixed into the blend's gate before postproc and clock rules.

The global p_imp classifier sees all departures; BLOCK = SCHED copies that matter sit in delayed rows (MVT-SCHED large),
where a copy costs ~(MVT-SCHED)^2. In the 2025 training data, at LIRF the copy share follows the callsign prefix
(handler), stand block, hour and day congestion (copies ~constant per month, genuine delays grow in summer).
q = CatBoost on those label-free inputs, fitted on delayed rows only (MVT-SCHED > MIN_DS), cross-fitted by FOLDS months
and refitted on all train months for the ranking set. Gate g = p_imp + w_group*(q - p_imp) on rows with
MIN_DS < MVT-SCHED <= MAX_DS (group: LIRF / other), then the unchanged chain: mixture -> postproc (alpha 0.5) -> clockfix R1-R3.
Writes OOF + result json per w setting, the q cache, and with --test the test file in the submitting.parquet layout.

python gate2.py --id gate2_v1 --blend blend_v20_cf --test_raw final_blend_v20
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

import clockfix
import postproc
from ocommon import FOLDS, OUTD, PROC, REP, ROOT, report, rmse, save_result, TEMPLATE, CB_TASK

CATS = ["apt", "air", "apt_air", "apt_stand", "stand_blk", "ades", "actype", "len_fl", "apt_rwy"]
NUM = ["sched_h", "mvt_h", "dow", "lds", "noplan", "da", "ml", "gap", "ae", "day_n", "day_aobt_delay_med", "day_ds_med",
       "day_n_del", "day_n_del_noplan", "hr_ds_med", "hr_n_del_noplan", "day_noplan_share", "ds_rel_day", "ds_rel_hr"]


def build(min_ds):
    m = (pl.read_parquet(PROC / "movements.parquet")
         .filter(pl.col("PHASE_mvt") == "DEP")
         .with_columns(ds=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).cast(pl.Float64),
                       imp=((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).abs() <= 30).cast(pl.Int8),
                       noplan=pl.col("AOBT_3_flt").is_null().cast(pl.Int8),
                       da=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).cast(pl.Float64),
                       ml=(pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).cast(pl.Float64),
                       gap=(pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).cast(pl.Float64),
                       ae=(pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).cast(pl.Float64),
                       apt=pl.col("ADEP_mvt"))
         .with_columns(day=pl.col("SCHED_TIME_UTC_mvt") // 86400, hr=pl.col("SCHED_TIME_UTC_mvt") // 3600))
    plan = pl.col("noplan") == 0
    # congestion per airport-day / airport-hour from clocks that are present on the ranking set too
    m = m.with_columns(
        day_n=pl.len().over("apt", "day"),
        day_aobt_delay_med=(pl.col("AOBT_3_flt") - pl.col("SCHED_TIME_UTC_mvt")).cast(pl.Float64).median().over("apt", "day"),
        day_ds_med=pl.col("ds").median().over("apt", "day"),
        day_n_del=(pl.col("ds") > 3600).sum().over("apt", "day"),
        day_n_del_noplan=(~plan & (pl.col("ds") > 1800)).sum().over("apt", "day"),
        hr_ds_med=pl.col("ds").median().over("apt", "hr"),
        hr_n_del_noplan=(~plan & (pl.col("ds") > 1800)).sum().over("apt", "hr"),
        day_noplan_share=pl.col("noplan").mean().over("apt", "day"),
    ).with_columns(ds_rel_day=pl.col("ds") - pl.col("day_aobt_delay_med"), ds_rel_hr=pl.col("ds") - pl.col("hr_ds_med"))
    d = m.filter(pl.col("ds") > min_ds).to_pandas()
    t = pd.to_datetime(d.MVT_TIME_UTC_mvt, unit="s")
    s = pd.to_datetime(d.SCHED_TIME_UTC_mvt, unit="s")
    d["month"] = t.dt.month
    d["air"] = d.FLIGHT_mvt.fillna("NA").str[:3]
    d["apt_air"] = d.apt + "_" + d.air
    d["apt_stand"] = d.apt + "_" + d.STAND_mvt.fillna("NA").astype(str)
    d["stand_blk"] = d.apt + "_" + d.STAND_mvt.fillna("NA").astype(str).str[0]
    d["apt_rwy"] = d.apt + "_" + d.RUNWAY_mvt.fillna("NA").astype(str)
    d["ades"] = d.ADES_mvt.fillna("NA")
    d["actype"] = d.AIRCRAFT_TYPE_mvt.fillna("NA")
    d["len_fl"] = d.FLIGHT_mvt.fillna("").str.len().astype(str)
    d["sched_h"] = s.dt.hour + s.dt.minute / 60
    d["mvt_h"] = t.dt.hour + t.dt.minute / 60
    d["dow"] = s.dt.dayofweek
    d["lds"] = np.log1p(d.ds.clip(0))
    return d


BAG_SEED = int(os.environ.get("BAG_SEED") or 0)   # seed offset of a deliberate redraw (full-chain seed bag); 0 = the shipped seeds


def fit_predict(tr, te, iters, seed=0, seeds=1):
    """seeds > 1: mean of the probabilities of `seeds` fits (random_seed seed..seed+seeds-1); GPU CatBoost is not bit-reproducible
    and its q noise moves the CV by about 0.1 RMSE, so the variant chain averages it away at the source."""
    q = 0
    for s in range(seed + BAG_SEED, seed + BAG_SEED + seeds):
        clf = CatBoostClassifier(iterations=iters, depth=6, learning_rate=0.08, l2_leaf_reg=10, verbose=0, task_type=CB_TASK,
                                 random_seed=s, border_count=254)
        clf.fit(Pool(tr[CATS + NUM], tr.imp, cat_features=CATS))
        q = q + clf.predict_proba(Pool(te[CATS + NUM], cat_features=CATS))[:, 1] / seeds
    return q, clf


def chain(o, g, delta_by_row, rates_by_fold, c, pp_alpha, clip=False):
    """mixture with gate g -> postproc per fold -> clockfix R1-R3. o holds MVT_ID_mvt, fold, reg, d_sched, cell."""
    tmp = o.copy()
    tmp["pred"] = g * (tmp.d_sched + delta_by_row) + (1 - g) * tmp.reg
    out = tmp.pred.to_numpy().copy()
    for f, (rt, taxi, delta) in rates_by_fold.items():
        fm = (tmp.fold == f).to_numpy()
        out[fm] = postproc.apply(tmp[fm], rt, taxi, delta, pp_alpha)
    if clip:  # final_blend clips the test path; the OOF chain (postproc -> clockfix) never did
        out = np.clip(out, 0, None)
    pred, *_ = clockfix.apply(tmp[["MVT_ID_mvt"]], out, c, True, True)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--blend", default="blend_v20_cf", help="OOF of the blend (pred = mixture, columns p_imp, reg)")
    ap.add_argument("--test_raw", default="", help="final_blend id whose _raw.parquet holds test p_imp and reg")
    ap.add_argument("--min_ds", type=float, default=900)
    ap.add_argument("--max_ds", type=float, default=20000)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=1, help="CatBoost seeds averaged in every q fit (default 1 = the shipped chain)")
    ap.add_argument("--ws", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--pp_alpha", type=float, default=0.5)
    ap.add_argument("--q_only", type=int, default=0, help="1 = only (re)build the q cache: the OOF q is reused from an existing "
                    "cache/<id>_q.parquet (2025-only, seeded by scripts/phase_seed.py) and the rank q is refitted on all 2025; no chain, "
                    "no test file (final phase: the rank rows are final_ranking.parquet)")
    a = ap.parse_args()
    run = f"gate2-{a.id}"
    total = 6
    report(run, 0, total, "building delayed-row features",
           purpose="Does a dedicated copy gate for delayed departures (handler/congestion inputs) beat the global p_imp inside the full chain?")
    d = build(a.min_ds)
    trn = d[d.split == "train"]
    q_oof = []
    fold_auc = {}
    qpath = OUTD / "cache" / f"{a.id}_q.parquet"
    if a.q_only and qpath.exists():
        q_oof = [pd.read_parquet(qpath).query("split == 'oof'")[["MVT_ID_mvt", "q"]]]
    for i, (f, fd) in enumerate(FOLDS.items() if not q_oof else []):
        tr, va = trn[trn.month.isin(fd["train"])], trn[trn.month.isin(fd["val"])]
        q, _ = fit_predict(tr, va, a.iters, seeds=a.seeds)
        q_oof.append(pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "q": q}))
        fold_auc[f] = float(roc_auc_score(va.imp, q))
        report(run, 1 + i, total, f"{f}: {len(tr)} train rows, val AUC {fold_auc[f]:.4f}")
    q_oof = pd.concat(q_oof)
    q_test = None
    if a.test_raw or a.q_only:
        te = d[d.split == "rank"]
        q, clf = fit_predict(trn, te, a.iters, seeds=a.seeds)
        q_test = pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.to_numpy(), "q": q})
        imp = pd.Series(clf.get_feature_importance(), index=CATS + NUM).sort_values(ascending=False).round(1)
        print("importance (full fit):", imp.head(15).to_dict())
    report(run, 3, total, "q fitted; evaluating chain")
    (OUTD / "cache").mkdir(parents=True, exist_ok=True)
    pd.concat([q_oof.assign(split="oof"), (q_test if q_test is not None else q_oof.iloc[:0]).assign(split="rank")]).to_parquet(
        qpath, index=False)
    if a.q_only:
        report(run, total, total, f"q cache written: {len(q_oof):,} oof rows, {0 if q_test is None else len(q_test):,} rank rows",
               finished=True)
        return

    # OOF chain inputs
    bres = json.loads((REP / "results" / f"{a.blend}.json").read_text())
    meta = postproc.load_meta()
    o = pd.read_parquet(OUTD / "oof" / f"{a.blend}.parquet").merge(meta[["MVT_ID_mvt", "cell", "d_sched", "apt", "kind"]], on="MVT_ID_mvt")
    o = o.merge(q_oof, on="MVT_ID_mvt", how="left")
    fold_delta = {f: bres["folds"][f]["delta"] for f in FOLDS}
    delta_row = o.fold.map(fold_delta).to_numpy()
    trd = meta[meta.split == "train"]
    rates = {f: (*postproc.rates(trd[trd.month.isin(fd["train"])], 8, 100), fold_delta[f]) for f, fd in FOLDS.items()}
    c = clockfix.clocks()
    y = o.y.to_numpy()
    rng = (o.d_sched > a.min_ds) & (o.d_sched <= a.max_ds) & o.q.notna()
    lirf = (o.apt == "LIRF").to_numpy()
    p = o.p_imp.to_numpy()
    qv = o.q.fillna(0).to_numpy()
    ws = [float(x) for x in a.ws.split(",")]
    base_pred = chain(o, p, delta_row, rates, c, a.pp_alpha)
    ref_oof = pd.read_parquet(OUTD / "oof" / f"clk3_{a.blend.replace('_cf', '')}.parquet").set_index("MVT_ID_mvt").pred
    base_chk = float(np.max(np.abs(ref_oof.loc[o.MVT_ID_mvt].to_numpy() - base_pred)))
    print(f"base chain vs clk3 OOF max abs diff {base_chk:.4f}", flush=True)
    grid = {}
    for wl in ws:
        for wo in ws:
            w = np.where(lirf, wl, wo) * rng.to_numpy()
            pr = chain(o, p + w * (qv - p), delta_row, rates, c, a.pp_alpha)
            grid[f"{wl}_{wo}"] = {"cv": rmse(pr, y), **{f: rmse(pr[(o.fold == f).to_numpy()], y[(o.fold == f).to_numpy()]) for f in FOLDS}}
    best = min(grid, key=lambda k: grid[k]["cv"])
    wl, wo = map(float, best.split("_"))
    report(run, 4, total, f"base chain {rmse(base_pred, y):.2f}; best w LIRF {wl} other {wo}: {grid[best]['cv']:.2f}")
    # per-fold robustness: the other fold's best w applied to this fold
    w = np.where(lirf, wl, wo) * rng.to_numpy()
    pred = chain(o, p + w * (qv - p), delta_row, rates, c, a.pp_alpha)
    res = {"id": a.id, "model": "gate2", "blend": a.blend, "min_ds": a.min_ds, "max_ds": a.max_ds, "iters": a.iters,
           "w_lirf": wl, "w_other": wo, "fold_auc_q": fold_auc,
           "auc_pimp_range": {f: float(roc_auc_score(o.kind[(o.fold == f) & rng] == "imp", o.p_imp[(o.fold == f) & rng])) for f in FOLDS},
           "auc_q_range": {f: float(roc_auc_score(o.kind[(o.fold == f) & rng] == "imp", o.q[(o.fold == f) & rng])) for f in FOLDS},
           "cv_rmse_base_chain": rmse(base_pred, y), "oof_base_chain_max_abs_diff": base_chk, "cv_rmse": rmse(pred, y), "cv_trim_rmse": rmse(pred[y <= 20000], y[y <= 20000]),
           "cv_clean_rmse": rmse(pred[y <= 7200], y[y <= 7200]), "grid": grid, "folds": {},
           "note": f"gate2 q (delayed-row CatBoost) mixed into {a.blend} gate, then postproc a{a.pp_alpha} + clockfix R1-R3"}
    for f in FOLDS:
        k = (o.fold == f).to_numpy()
        res["folds"][f] = {"rmse": rmse(pred[k], y[k]), "clean_rmse": rmse(pred[k & (y <= 7200)], y[k & (y <= 7200)]), "n": int(k.sum())}
    pd.DataFrame({"MVT_ID_mvt": o.MVT_ID_mvt, "fold": o.fold, "y": y, "pred": pred, "p_imp": p + w * (qv - p), "reg": o.reg}).to_parquet(
        OUTD / "oof" / f"{a.id}.parquet", index=False)

    if a.test_raw:
        t = pd.read_parquet(OUTD / "test" / f"{a.test_raw}_raw.parquet")[["MVT_ID_mvt", "p_imp", "reg"]]
        t = t.merge(meta[["MVT_ID_mvt", "cell", "d_sched", "apt"]], on="MVT_ID_mvt", how="left").merge(q_test, on="MVT_ID_mvt", how="left")
        delta = float(np.mean(trd.TAXITIME_SEC_mvt[trd.kind == "imp"] - trd.d_sched[trd.kind == "imp"]))
        rt, taxi = postproc.rates(trd, 8, 100)
        tr_ = ((t.d_sched > a.min_ds) & (t.d_sched <= a.max_ds) & t.q.notna()).to_numpy()
        wt = np.where((t.apt == "LIRF").to_numpy(), wl, wo) * tr_
        tp = t.p_imp.to_numpy()
        g = tp + wt * (t.q.fillna(0).to_numpy() - tp)
        t["fold"] = "all"
        tpred = chain(t, g, np.full(len(t), delta), {"all": (rt, taxi, delta)}, c, a.pp_alpha, clip=True)
        base_t = chain(t, tp, np.full(len(t), delta), {"all": (rt, taxi, delta)}, c, a.pp_alpha, clip=True)
        ref = pd.read_parquet(OUTD / "test" / f"clk3_{a.test_raw}.parquet")
        chk = ref.MVT_ID_mvt.astype("int64").map(pd.Series(base_t, index=t.MVT_ID_mvt.to_numpy()))
        res["test_base_chain_max_abs_diff_vs_ref"] = float(np.nanmax(np.abs(chk.to_numpy() - ref.TAXITIME_SEC_mvt.to_numpy())))
        res["test_rows_regated"] = int(tr_.sum())
        res["test_rms_change"] = float(np.sqrt(np.mean((tpred - base_t) ** 2)))
        template = pd.read_parquet(TEMPLATE)
        template["TAXITIME_SEC_mvt"] = template.MVT_ID_mvt.astype("int64").map(pd.Series(tpred, index=t.MVT_ID_mvt.to_numpy())).to_numpy()
        path = OUTD / "test" / f"{a.id}.parquet"
        template.to_parquet(path, index=False)
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from baseline import validate_submission
        validate_submission(path)
    save_result(res)
    report(run, total, total, f"CV {res['cv_rmse']:.2f} (base chain {res['cv_rmse_base_chain']:.2f}), w LIRF {wl} other {wo}", finished=True)
    print(json.dumps({k: v for k, v in res.items() if k not in ("grid",)}, indent=1))
    print("grid:", {k: round(v["cv"], 2) for k, v in grid.items()})


if __name__ == "__main__":
    main()
