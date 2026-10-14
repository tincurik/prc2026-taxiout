# SPDX-License-Identifier: GPL-3.0-or-later
"""Tail post-step: label-free runway-neighbour level corrector.

The final chain's residual shares a level across the departures on one runway within about 15 min (oracle with the neighbours' labels
-2.85 RMSE on the final's months, 87% of it on untracked LTFM / LFPG / EGLL / LIRF rows), and no single row's inputs carry it. This step
estimates the part the 2026 files can see: for each departure, the shrunk mean over the other departures on its runway (+-300 / 900 /
1,800 s) and its airport (+-900 s) of u = clock - pred for the NM-AOBT gap, the ADS-B seen-to-take-off time, the rules-chain non-copy
value and the EOBT gap, plus the neighbours' pred and counts (23 columns). A linear fit of clip(pred - y, +-1,800) on them is the
correction; pred' = max(pred - correction, 0) on every row (no label-defined gate). --model lgb fits LightGBM (L2, 15 leaves, 300 rounds,
2 threads) on the same columns + the airport as a category instead (better than the linear fit in every fold); in a
pseudo phase it skips the phase's own OOF file (never scored; it would need one model per left-out pair).

Coefficients come from the fit base's 12-month honest pool (final Jan/Jul OOF + pseudo2-6 held-out tests vs truth), never from the folds
that hold the row's own months (as r16_remix.py): 2025 rows leave out their month's fold and the current pseudo phase's fold; 2026 rows
(the final's test) use every fold. Neighbours are always the rows of the same file (whole months, so whole airport-days).

  EC_PHASE=final python r17_rwnb.py --id ar_134naclh_rwnb --base ar_134naclh_rmx [--fit_base ar_134naclh_rmx] [--model linear|lgb]   (pipeline/39_post_steps.sh)
"""
import argparse
import json
import os

from pathlib import Path

import numpy as np
import pandas as pd

import ocommon  # noqa: F401
from ocommon import OUTD, PHASE, PROC, ROOT, TEMPLATE, report, rmse, save_result

FOLD_MONTHS = {"final": (1, 7), "pseudo2": (2, 6), "pseudo3": (3, 9), "pseudo4": (4, 10), "pseudo5": (5, 11), "pseudo6": (8, 12)}
MONTH_FOLD = {m: f for f, ms in FOLD_MONTHS.items() for m in ms}
# Pool locations: the final slot (this phase when it is a final* phase, else "final") lives under ROOT; the pseudo2-6 pairs under PAIRS_ROOT
# (default ROOT; PAIRS_ROOT points at the tree that holds them) and carry the fit base's pairs id (--pairs_base).
PAIRS = Path(os.environ.get("PAIRS_ROOT") or ROOT)
SLOT = PHASE if PHASE.startswith("final") else "final"
CL = 1800.0
WINDOWS = (300, 900, 1800)
PRED_C = 900.0                         # fixed centre of the neighbours' pred (the same in every phase)
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
LGB = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 2000, "lambda_l2": 10.0, "feature_fraction": 0.8,
       "bagging_fraction": 0.5, "bagging_freq": 1, "num_threads": 2, "seed": 0, "deterministic": True, "verbose": -1}


def inputs(proc, ids):
    f = pd.read_parquet(proc / "features_dep.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "apt", "RUNWAY_mvt", "d_aobt", "d_eobt"])
    f = f[f.MVT_ID_mvt.isin(ids)].drop_duplicates("MVT_ID_mvt")
    a = pd.read_parquet(proc / f"features_dep_{os.environ.get('RWNB_FEAT') or 'adsb2m'}.parquet", columns=["MVT_ID_mvt", "a2_seen"])   # RWNB_FEAT: another ADS-B feature set's seen time
    f = f.merge(a[a.MVT_ID_mvt.isin(ids)].drop_duplicates("MVT_ID_mvt"), on="MVT_ID_mvt", how="left")
    sec = f.MVT_TIME_UTC_mvt
    sec = sec.astype("int64") // 10**9 if np.issubdtype(sec.dtype, np.datetime64) else sec.astype("int64")
    t = pd.to_datetime(sec, unit="s")
    return f.assign(MVT_ID_mvt=f.MVT_ID_mvt.astype("int64"), t=sec.to_numpy(), year=t.dt.year.to_numpy(), month=t.dt.month.to_numpy())


def nbmean(val, key, t, W, K=3.0):
    """Mean of val over the other rows with the same key within +-W s (NaN skipped), shrunk to 0 with K; and their count."""
    ok = np.isfinite(val)
    v, o = np.where(ok, val, 0.0), ok.astype(float)
    order = np.lexsort((t, key))
    comp = key[order].astype(np.int64) * 10**10 + t[order]
    cs, cn = np.concatenate([[0], np.cumsum(v[order])]), np.concatenate([[0], np.cumsum(o[order])])
    i, j = np.searchsorted(comp, comp - W, "left"), np.searchsorted(comp, comp + W, "right")
    s, c = cs[j] - cs[i] - v[order], cn[j] - cn[i] - o[order]
    m, n = np.zeros(len(val)), np.zeros(len(val))
    m[order], n[order] = s / (c + K), c
    return m, n


def frame(pred, parts, proc):
    """pred: MVT_ID_mvt, pred (+ y); parts: MVT_ID_mvt, reg -> rows with the 23 neighbour columns (FEATS), year, month."""
    d = pred.merge(parts[["MVT_ID_mvt", "reg"]].drop_duplicates("MVT_ID_mvt"), on="MVT_ID_mvt", how="left")
    d = d.merge(inputs(proc, d.MVT_ID_mvt.to_numpy()), on="MVT_ID_mvt", how="left").reset_index(drop=True)
    p, t = d.pred.to_numpy(float), d.t.to_numpy(np.int64)
    k_rw = pd.factorize(d.apt.astype(str) + "|" + d.RUNWAY_mvt.fillna("NA").astype(str))[0]
    k_ap = pd.factorize(d.apt.astype(str))[0]
    q = {"u_aobt": d.d_aobt.to_numpy(float) - p, "u_seen": d.a2_seen.to_numpy(float) - p, "u_reg": d.reg.to_numpy(float) - p,
         "u_eobt": d.d_eobt.to_numpy(float) - p, "pred": p - PRED_C}
    for k, v in q.items():
        v = np.clip(v, -CL, CL)
        for W in WINDOWS:
            d[f"rw{W}_{k}"], n = nbmean(v, k_rw, t, W)
            if k == "pred":
                d[f"rw{W}_n"] = np.log1p(n)
        d[f"ap900_{k}"], _ = nbmean(v, k_ap, t, 900)
    return d


FEATS = [f"rw{W}_{k}" for k in ("u_aobt", "u_seen", "u_reg", "u_eobt", "pred") for W in WINDOWS] + [f"rw{W}_n" for W in WINDOWS] + \
        [f"ap900_{k}" for k in ("u_aobt", "u_seen", "u_reg", "u_eobt", "pred")]


def pool(fit_base, pairs_base=None):
    """The fit base's honest pool: neighbour columns + clipped residual + fold (cached; written atomically for parallel phases)."""
    out = ROOT / f"outputs/r17_rwnb/pool_{fit_base}.parquet"
    if out.exists():
        d = pd.read_parquet(out)
        if "apt" in d:
            return d
    pairs_base = pairs_base or fit_base
    parts = []
    o = pd.read_parquet(ROOT / f"outputs/overnight_{SLOT}/oof/{fit_base}.parquet")[["MVT_ID_mvt", "y", "pred"]]
    g = pd.read_parquet(ROOT / f"outputs/overnight_{SLOT}/oof/rc_v34_more18.parquet")
    parts.append(frame(o, g, ROOT / f"data/processed_{SLOT}").assign(fold="final"))
    for k in range(2, 7):
        pf = PAIRS / f"outputs/overnight_pseudo{k}/test/{pairs_base}.parquet"
        if not pf.exists():
            print(f"pool: no {pf.name} in pseudo{k}, fold left out")
            continue
        t = pd.read_parquet(PAIRS / f"data/pseudo{k}/truth.parquet").rename(columns={"TAXITIME_SEC_mvt": "y"})
        p = pd.read_parquet(pf).rename(columns={"TAXITIME_SEC_mvt": "pred"})
        g = pd.read_parquet(PAIRS / f"outputs/overnight_pseudo{k}/test/rc_v34_more18_parts.parquet")
        parts.append(frame(t.merge(p, on="MVT_ID_mvt"), g, PAIRS / f"data/processed_pseudo{k}").assign(fold=f"pseudo{k}"))
    d = pd.concat(parts, ignore_index=True)
    d["ec"] = np.clip(d.pred - d.y, -CL, CL)
    d = d[["MVT_ID_mvt", "fold", "apt", "ec"] + FEATS].astype({c: "float32" for c in FEATS})
    d["apt"] = d.apt.astype(str)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    d.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return d


def aptcode(apt):
    return pd.Categorical(apt.astype(str), categories=APTS).codes.astype(np.float32)


def apply(d, P, own_fold, model="linear"):
    """Corrected predictions for frame d; one coefficient vector (or LightGBM model) per set of left-out folds."""
    def leave(y, m):
        s = {MONTH_FOLD[int(m)]} if int(y) == 2025 else set()
        return "+".join(sorted(s | ({own_fold} if own_fold else set())))
    lo = np.array([leave(y, m) for y, m in zip(d.year.fillna(0), d.month.fillna(0))])
    X = np.c_[d[FEATS].to_numpy(float), np.ones(len(d))]
    corr, coefs = np.zeros(len(d)), {}
    for k in pd.unique(lo):
        keep = ~P.fold.isin([x for x in k.split("+") if x]).to_numpy()
        m = lo == k
        if model == "lgb":
            import lightgbm as lgb
            XP = np.c_[P[FEATS].to_numpy(np.float32)[keep], aptcode(P.apt[keep])]
            ds = lgb.Dataset(XP, P.ec.to_numpy(float)[keep], categorical_feature=[len(FEATS)], free_raw_data=True)
            booster = lgb.train(LGB, ds, 300)
            del XP, ds
            corr[m] = booster.predict(np.c_[d[FEATS].to_numpy(np.float32)[m], aptcode(d.apt[m])])
            coefs[f"lo{k or 'none'}"] = {"model": "lgb", "rows": int(keep.sum()), "gain_top": dict(sorted(zip(FEATS + ["apt"],
                                         booster.feature_importance("gain").round(0).tolist()), key=lambda x: -x[1])[:5])}
            continue
        beta = np.linalg.lstsq(np.c_[P[FEATS].to_numpy(float)[keep], np.ones(keep.sum())], P.ec.to_numpy(float)[keep], rcond=None)[0]
        coefs[f"lo{k or 'none'}"] = dict(zip(FEATS + ["const"], np.round(beta, 5).tolist()))
        corr[m] = X[m] @ beta
    return np.clip(d.pred.to_numpy(float) - corr, 0, None), coefs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--base", required=True, help="tail id with oof/<base>.parquet and/or test/<base>.parquet in this phase")
    ap.add_argument("--fit_base", default="ar_134naclh_rmx", help="id whose honest pool fits the coefficients")
    ap.add_argument("--model", default="linear", choices=["linear", "lgb"])
    ap.add_argument("--pairs_base", default=None, help="id of the fit base in the pseudo phases (default: --fit_base; the bundle id of the rehearsal phases can differ from the shipped one)")
    a = ap.parse_args()
    run = f"r17rwnb-{PHASE or 'rank'}-{a.id}"
    purpose = "Does a linear runway-neighbour level corrector on label-free clocks (LOFO on the 12-month pool) lower E_final RMSE?"
    report(run, 0, 3, "pool", purpose=purpose)
    P = pool(a.fit_base, a.pairs_base)
    own = PHASE if PHASE.startswith("pseudo") else None
    res = {"id": a.id, "base": a.base, "fit_base": a.fit_base, "phase": PHASE, "step": "r17_rwnb", "model": a.model, "pool_rows": len(P),
           "pool_folds": sorted(P.fold.unique().tolist())}
    report(run, 1, 3, "oof")
    of = OUTD / "oof" / f"{a.base}.parquet"
    if of.exists() and not (a.model == "lgb" and own):
        o = pd.read_parquet(of)[["MVT_ID_mvt", "fold", "y", "pred"]]
        d = frame(o, pd.read_parquet(OUTD / "oof" / "rc_v34_more18.parquet"), PROC)
        new, co = apply(d, P, own, a.model)
        y = d.y.to_numpy(float)
        res.update(base_cv_rmse=rmse(d.pred, y), cv_rmse=rmse(new, y), oof_moved=int((np.abs(new - d.pred) > 1).sum()), oof_coefs=co,
                   folds={str(f): {"base": rmse(d.pred[d.fold == f], y[d.fold == f]), "new": rmse(new[d.fold == f], y[d.fold == f])}
                          for f in d.fold.unique()})
        pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt, "fold": d.fold, "y": y, "pred": new}).to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    report(run, 2, 3, "test")
    tf = OUTD / "test" / f"{a.base}.parquet"
    if tf.exists():
        t = pd.read_parquet(tf).rename(columns={"TAXITIME_SEC_mvt": "pred"})
        g = pd.read_parquet(OUTD / "test" / "rc_v34_more18_parts.parquet")
        d = frame(t[["MVT_ID_mvt", "pred"]], g, PROC)
        if d.reg.isna().mean() > 0.01:
            raise SystemExit(f"rc_v34_more18_parts: reg missing on {d.reg.isna().mean():.1%} of test rows")
        new, co = apply(d, P, own, a.model)
        res.update(test_moved=int((np.abs(new - d.pred) > 1).sum()), test_shift_rms=float(np.sqrt(np.mean((new - d.pred) ** 2))), test_coefs=co)
        template = pd.read_parquet(TEMPLATE)
        template["TAXITIME_SEC_mvt"] = template.MVT_ID_mvt.map(pd.Series(new, index=d.MVT_ID_mvt.to_numpy())).astype(float)
        path = OUTD / "test" / f"{a.id}.parquet"
        template.to_parquet(path, index=False)
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from baseline import validate_submission
        validate_submission(path)
    if not of.exists() and not tf.exists():
        report(run, 3, 3, f"no {a.base} in this phase", finished=True)
        raise SystemExit(f"{a.base}: neither oof nor test in {OUTD}")
    save_result(res)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("coefs")}, indent=1))
    report(run, 3, 3, f"done cv {res.get('cv_rmse', float('nan')):.3f} (base {res.get('base_cv_rmse', float('nan')):.3f}), "
           f"test moved {res.get('test_moved', 0)}", finished=True)
    if ocommon._progress:
        ocommon._progress.report(run if run.startswith("eurocontrol-") else "eurocontrol-" + run, "EUROCONTROL " + run, 3, 3,
                                 detail="done", finished=True, outcome="success")


if __name__ == "__main__":
    main()
