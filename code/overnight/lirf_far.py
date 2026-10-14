# SPDX-License-Identifier: GPL-3.0-or-later
"""Dedicated SCHED-copy classifier for LIRF far no-plan departures, applied as a post-step on a stacked chain output.

Rows: LIRF departures without AOBT_3 and LO < MVT - SCHED <= HI (the gate2 domain). 377 of them in the Jan/Jul OOF
folds carry ~11 k of the CV MSE; the pipeline's effective copy probability there (backed out of the chain) has log
loss 0.408 / AUC 0.877, while a CatBoost trained on only these rows over all 12 months of 2025 reaches 0.35 / 0.91.
Features: carrier, stand, type, destination, market segment, delay, hour, weekday, arrival-imputation rates (fs2),
plus (--arrmo 1) the calendar month's LIRF arrival BLOCK = SCHED rate (arrivals are unmasked in 2026, so it is
serve-time safe; across 2025 months it tracks the departure copy share, r 0.79).

The correction keeps everything the base did elsewhere (incl. the ADS-B stacker):
  new = base + w * (p_new - p_chain) * (d_sched - reg),  p_chain = (chain_pred - reg) / (d_sched - reg)
where chain_pred / reg are the pre-stack rules-chain prediction and its regressor branch.
OOF: the Jan 2025 fold uses a model fitted on the other 11 months (same for Jul). Test: fitted on all 2025.

  python lirf_far.py --id ar_45_lirf_far --base ar_39_adsb_mlp4 --chain rc_v24_mlp4
--plan 1 (2026-09-30) switches to departures WITH an NM plan (AOBT_3 present) and adds the plan clocks as features:
LIRF plan rows 20-60 min late copy SCHED in 12.3 % of cases (other airports 0.4-4 %), outside lirf_far's no-plan domain;
the chain's p separates them at AUC 0.85 and their copies cost 2.2 k CV MSE. --apt picks the airport (LEBL is next).
--inb 1 adds the same airframe's inbound leg (inb_imp = its in-block recorded at schedule): in the top chain-p quintile
of plan rows, copy share 64 % with an imputed inbound vs 36 % without; no-plan far rows 64 % vs 31 %.
  python lirf_far.py --id ar_105_lplan --base ar_103_fin --chain rc_v34_more18 --plan 1 --lo 1200 --hi 20000
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ocommon import OUTD, PROC, ROOT, report, rmse, save_result

CATS = ["airline", "stand_pre", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "MARKET_SEGMENT_flt"]
PLAN_NUM = ["d_aobt", "d_lobt", "d_eobt", "d_iobt", "aobt_m_sched", "eobt_m_sched", "aobt_m_eobt", "eobt_m_iobt", "has_flt"]
NUM = ["d_sched", "local_hour", "dow", "fa_arrimp_apt_airline_24h", "fa_arrimp_apt_airline_72h", "fa_arrimp_apt_3h",
       "fa_arrimp_apt_12h", "fa_prevarr_imp", "fa_prevarr_blk_m_sched", "dep_day_cnt", "callsign_mismatch"]


INB = ["inb_imp", "inb_delay", "inb_taxiin", "inb_gap_sched", "inb_gap_mvt", "inb_sched_turn", "inb_same_stand", "inb_same_airline"]


def rows(lo, hi, plan=0, apt="LIRF", inb=0):
    f = pd.read_parquet(PROC / "features_dep.parquet", columns=list(dict.fromkeys(["MVT_ID_mvt", "apt", "d_aobt", "d_sched", "split",
                                                                  "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", *CATS, *NUM, *PLAN_NUM])))
    has = f.d_aobt.notna() if plan else f.d_aobt.isna()
    f = f[(f.apt == apt) & has & (f.d_sched.abs() > lo)].copy()  # train on every row past lo (> hi too: 75% copies when no plan)
    if inb:
        f = f.merge(pd.read_parquet(PROC / "features_dep_ext_inb.parquet"), on="MVT_ID_mvt", how="left")
    f["apply"] = (f.d_sched > lo) & (f.d_sched <= hi)
    f["mo"] = pd.to_datetime(f.MVT_TIME_UTC_mvt, unit="s").dt.strftime("%y%m")
    f["c"] = ((f.TAXITIME_SEC_mvt - f.d_sched).abs() <= 60).astype(int)
    m = pd.read_parquet(PROC / "movements.parquet", columns=["PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
                                                            "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    a = m[(m.PHASE_mvt == "ARR") & (m.ADES_mvt == apt)].copy()
    a["mo"] = pd.to_datetime(a.MVT_TIME_UTC_mvt, unit="s").dt.strftime("%y%m")
    a["c"] = (a.BLOCK_TIME_UTC_mvt - a.SCHED_TIME_UTC_mvt).abs() <= 30
    f["arr_mo"] = f.mo.map(a.groupby("mo").c.mean())
    for c in CATS:
        f[c] = f[c].fillna("NA").astype(str)
    return f


def fit_predict(tr, te, cols, seeds):
    p = np.zeros(len(te))
    for s in range(seeds):
        m = CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05, verbose=0, random_seed=s,
                               cat_features=[c for c in cols if c in CATS], thread_count=8)
        m.fit(tr[cols], tr.c)
        p += m.predict_proba(te[cols])[:, 1] / seeds
    return p


def correct(base, chain, reg, dsched, p_new, w, pref="chain"):
    den = dsched - reg
    p_chain = np.clip(((base if pref == "base" else chain) - reg) / np.where(np.abs(den) < 1, 1, den), 0, 1)
    return base + w * (p_new - p_chain) * den, p_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--base", required=True, help="stacked chain output id (oof/ and test/ parquet)")
    ap.add_argument("--chain", required=True, help="its pre-stack rules chain (oof with reg; test + _parts with reg)")
    ap.add_argument("--lo", type=float, default=3600)
    ap.add_argument("--hi", type=float, default=20000)
    ap.add_argument("--arrmo", type=int, default=1)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--plan", type=int, default=0, help="1 = rows WITH an NM plan (AOBT_3), plan clocks added as features")
    ap.add_argument("--apt", default="LIRF")
    ap.add_argument("--inb", type=int, default=0, help="1 = add the airframe-linked inbound features (features_dep_ext_inb: inb_imp ...)")
    ap.add_argument("--pref", default="chain", choices=["chain", "base"],
                    help="copy probability to replace: the chain's (first correction of a domain) or the base's own (re-correcting "
                         "a domain the base already corrected, e.g. no-plan far rows inside ar_103)")
    ap.add_argument("--adsb", default="all", choices=["all", "ok", "nok", "okdark"],
                    help="apply only to ADS-B-matched (ok) / unmatched (nok) rows: the chain's p beats p_new on unmatched ones; "
                         "okdark = matched rows + every row of a dark month (< 10 %% of these rows matched: LIRF Jan-25, Jul-26)")
    ap.add_argument("--buffer", type=int, default=0,
                    help="1 = the OOF fit also leaves out the months next to the held-out one (Jan: Feb; Jul: Jun, Aug), the "
                         "two-month buffer of docs/validation.md; 0 (the recorded runs) = every other month is in the fit")
    a = ap.parse_args()
    run = f"lirf-far-{a.id}"
    report(run, 0, 4, "loading", purpose=f"Does a dedicated 12-month copy classifier on {a.apt} {'plan' if a.plan else 'far no-plan'} rows beat the chain's copy probability?")
    cols = CATS + NUM + (PLAN_NUM if a.plan else []) + (["arr_mo"] if a.arrmo else []) + (INB if a.inb else [])
    f = rows(a.lo, a.hi, a.plan, a.apt, a.inb)
    if a.adsb != "all":
        ok = f.MVT_ID_mvt.map(pd.read_parquet(PROC / "features_dep_adsb2fullrel.parquet", columns=["MVT_ID_mvt", "a2_ok"])
                              .set_index("MVT_ID_mvt").a2_ok).fillna(0).astype(bool)
        if a.adsb == "okdark":
            dark = f.mo.map(ok.groupby(f.mo).mean()) < 0.10
            print("month coverage:", ok.groupby(f.mo).mean().round(3).to_dict())
            f["apply"] = f["apply"] & (ok | dark)
        else:
            f["apply"] = f["apply"] & (ok if a.adsb == "ok" else ~ok)
    tr_all = f[f.split == "train"]

    oof = pd.read_parquet(OUTD / "oof" / f"{a.base}.parquet")
    ch = pd.read_parquet(OUTD / "oof" / f"{a.chain}.parquet")[["MVT_ID_mvt", "pred", "reg"]].rename(columns={"pred": "chain"})
    info = {}
    new = oof.pred.to_numpy().copy()
    idx = pd.Series(np.arange(len(oof)), index=oof.MVT_ID_mvt)
    for k, mo in enumerate(["2501", "2507"], 1):
        te = tr_all[(tr_all.mo == mo) & tr_all["apply"]].merge(ch, on="MVT_ID_mvt")
        drop = {"2501": ["2501", "2502"], "2507": ["2506", "2507", "2508"]}[mo] if a.buffer else [mo]
        p = fit_predict(tr_all[~tr_all.mo.isin(drop)], te, cols, a.seeds)
        i = idx.loc[te.MVT_ID_mvt].to_numpy()
        new[i], p_chain = correct(new[i], te.chain.to_numpy(), te.reg.to_numpy(), te.d_sched.to_numpy(float), p, a.w, a.pref)
        info[mo] = dict(n=len(te), copy=float(te.c.mean()), p_new=float(p.mean()), p_chain=float(p_chain.mean()))
        report(run, k, 4, f"{mo}: {len(te)} rows, copy {te.c.mean():.3f}, p_new {p.mean():.3f}, p_chain {p_chain.mean():.3f}")
    y = oof.y.to_numpy()
    res = {"id": a.id, "model": "lirf_far", "base": a.base, "chain": a.chain, "lo": a.lo, "hi": a.hi, "arrmo": a.arrmo, "plan": a.plan, "apt": a.apt, "inb": a.inb, "pref": a.pref,
           "w": a.w, "seeds": a.seeds, "adsb": a.adsb, "buffer": a.buffer, "cv_rmse": rmse(new, y), "cv_trim_rmse": rmse(new[y <= 20000], y[y <= 20000]),
           "cv_clean_rmse": rmse(new[y <= 7200], y[y <= 7200]), "base_cv_rmse": rmse(oof.pred, y), "info": info, "folds": {}}
    for fo in ("jan", "jul"):
        k = (oof.fold == fo).to_numpy()
        res["folds"][fo] = {"rmse": rmse(new[k], y[k]), "base_rmse": rmse(oof.pred.to_numpy()[k], y[k]), "n": int(k.sum())}
    oof.assign(pred=new)[["MVT_ID_mvt", "fold", "y", "pred"]].to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)

    report(run, 3, 4, "test model on all 2025")
    t = pd.read_parquet(OUTD / "test" / f"{a.base}.parquet")
    tc = pd.read_parquet(OUTD / "test" / f"{a.chain}.parquet").rename(columns={"TAXITIME_SEC_mvt": "chain"})
    tc = tc.merge(pd.read_parquet(OUTD / "test" / f"{a.chain}_parts.parquet")[["MVT_ID_mvt", "reg"]], on="MVT_ID_mvt")
    te = f[(f.split != "train") & f["apply"]].merge(tc, on="MVT_ID_mvt")
    p = fit_predict(tr_all, te, cols, a.seeds)
    tp = t.TAXITIME_SEC_mvt.to_numpy().copy()
    ti = pd.Series(np.arange(len(t)), index=t.MVT_ID_mvt.astype("int64")).loc[te.MVT_ID_mvt].to_numpy()
    tp[ti], p_chain = correct(tp[ti], te.chain.to_numpy(), te.reg.to_numpy(), te.d_sched.to_numpy(float), p, a.w, a.pref)
    t["TAXITIME_SEC_mvt"] = np.clip(tp, 0, None)
    path = OUTD / "test" / f"{a.id}.parquet"
    t.to_parquet(path, index=False)
    sys.path.insert(0, str(ROOT / "scripts"))
    from baseline import validate_submission
    validate_submission(path)
    te["mo"] = te.mo.astype(str)
    res["test"] = {mo: dict(n=int((te.mo == mo).sum()), p_new=float(p[(te.mo == mo).to_numpy()].mean()),
                            p_chain=float(p_chain[(te.mo == mo).to_numpy()].mean())) for mo in sorted(te.mo.unique())}   # every ranked month (2601 2607, plus 2602 2606 in the final)
    res["test_rms_change"] = float(np.sqrt(np.mean((t.TAXITIME_SEC_mvt.to_numpy() - pd.read_parquet(OUTD / "test" / f"{a.base}.parquet").TAXITIME_SEC_mvt.to_numpy()) ** 2)))
    save_result(res)
    print(json.dumps(res, indent=1))
    report(run, 4, 4, f"CV {res['cv_rmse']:.2f} (base {res['base_cv_rmse']:.2f}); test {res['test']}", finished=True)


if __name__ == "__main__":
    main()
