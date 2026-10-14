# SPDX-License-Identifier: GPL-3.0-or-later
"""Honest-gap tail post-step: LIRF far rows beyond the hard-copy range.

LIRF no-plan departures with C = d_sched > 50000 s have two outcomes in 2025: the plan copy (y = C) or the '+1 day' record
(y ~ 86400 + taxi, 87.2-88.4 k); none is a normal taxi. Below ~55 k the chain commits to the copy, above it the stack predicts
~75-90 k whatever the rules-chain gate says, although the gate's copy probability p_imp separates the two (2025: p >= .5 3/3 copies,
p < .5 2/13). This step sets those rows to the gate's mixture: pred' = p*C + (1-p)*87500 on LIRF rows with no AOBT_3, C > 50000 and
|pred - C| > 600 (rows already placed at the copy stay). No fitted parameters; 87500 = 86400 + 1100 (2025's day rows).

--calib bin2 (variant 2): p is replaced by the copy share of its gate bin
(p_imp >= .5 / < .5) among the eligible rows of the fit base's 12-month honest pool, Jeffreys-smoothed (k + .5) / (n + 1), leaving out
the folds that hold the row's own months and the current pseudo phase's fold (2026 rows use all six folds), as in r16_remix.py.

  EC_PHASE=final python hg_far50.py --id ar_134naclh_hgf --base ar_134naclh_rwnb [--calib none|bin2 --fit_base ar_134na_dup]   (pipeline/39_post_steps.sh)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import ocommon  # noqa: F401
from ocommon import OUTD, PHASE, PROC, ROOT, TEMPLATE, report, rmse, save_result

DAY, MIN_C, AT_COPY = 87500.0, 50000.0, 600.0
FOLD_MONTHS = {"final": (1, 7), "pseudo2": (2, 6), "pseudo3": (3, 9), "pseudo4": (4, 10), "pseudo5": (5, 11), "pseudo6": (8, 12)}
MONTH_FOLD = {m: f for f, ms in FOLD_MONTHS.items() for m in ms}
# Pool locations: the final slot (this phase when it is a final* phase, else "final") lives under ROOT; the pseudo2-6 pairs under PAIRS_ROOT
# (default ROOT; PAIRS_ROOT points at the tree that holds them) and carry the fit base's pairs id (--pairs_base).
PAIRS = Path(os.environ.get("PAIRS_ROOT") or ROOT)
SLOT = PHASE if PHASE.startswith("final") else "final"


def _feats(proc, ids):
    f = pd.read_parquet(proc / "features_dep.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "apt", "d_sched", "AOBT_3_flt"])
    f = f[f.MVT_ID_mvt.isin(ids)].drop_duplicates("MVT_ID_mvt").copy()
    sec = f.MVT_TIME_UTC_mvt
    sec = sec.astype("int64") // 10**9 if np.issubdtype(sec.dtype, np.datetime64) else sec.astype("int64")
    t = pd.to_datetime(sec, unit="s")
    f["year"], f["month"] = t.dt.year.to_numpy(), t.dt.month.to_numpy()
    return f.drop(columns=["MVT_TIME_UTC_mvt"])


def pool(fit_base, pairs_base=None):
    """Eligible rows of the fit base's 12-month honest pool (final Jan/Jul OOF + pseudo2-6 held-out tests vs truth)."""
    pairs_base = pairs_base or fit_base
    parts = []
    o = pd.read_parquet(ROOT / f"outputs/overnight_{SLOT}/oof/{fit_base}.parquet")[["MVT_ID_mvt", "y", "pred"]]
    parts.append(frame(o, pd.read_parquet(ROOT / f"outputs/overnight_{SLOT}/oof/rc_v34_more18.parquet"), ROOT / f"data/processed_{SLOT}").assign(fold="final"))
    for k in range(2, 7):
        t = pd.read_parquet(PAIRS / f"data/pseudo{k}/truth.parquet").rename(columns={"TAXITIME_SEC_mvt": "y"})
        p = pd.read_parquet(PAIRS / f"outputs/overnight_pseudo{k}/test/{pairs_base}.parquet").rename(columns={"TAXITIME_SEC_mvt": "pred"})
        g = pd.read_parquet(PAIRS / f"outputs/overnight_pseudo{k}/test/rc_v34_more18_parts.parquet")
        parts.append(frame(t.merge(p, on="MVT_ID_mvt"), g, PAIRS / f"data/processed_pseudo{k}").assign(fold=f"pseudo{k}"))
    d = pd.concat(parts, ignore_index=True)
    d = d[d.elig].copy()
    d["iscopy"] = ((d.y - d.d_sched).abs() < 120).astype(int)
    return d[["MVT_ID_mvt", "fold", "p_imp", "iscopy"]]


def calibrate(d, P, own_fold):
    """Two-bin LOFO copy share for the eligible rows of d (see the docstring); returns p per row."""
    p = d.p_imp.astype(float).clip(0, 1).fillna(0).to_numpy().copy()
    coefs = {}
    for i in np.flatnonzero(d.elig.to_numpy()):
        leave = {MONTH_FOLD[int(d.month.iat[i])]} if int(d.year.iat[i]) == 2025 else set()
        if own_fold:
            leave.add(own_fold)
        hi = p[i] >= 0.5
        q = P[~P.fold.isin(leave) & ((P.p_imp >= 0.5) == hi)]
        p[i] = (q.iscopy.sum() + 0.5) / (len(q) + 1)
        coefs[f"{'hi' if hi else 'lo'}_lo{'+'.join(sorted(leave)) or 'none'}"] = [int(q.iscopy.sum()), len(q), float(p[i])]
    return p, coefs


def frame(pred, gate, proc=None):
    f = _feats(proc or PROC, pred.MVT_ID_mvt)
    d = pred.merge(f, on="MVT_ID_mvt", how="left").merge(gate[["MVT_ID_mvt", "p_imp"]].drop_duplicates("MVT_ID_mvt"), on="MVT_ID_mvt", how="left")
    C = d.d_sched.astype(float)
    d["elig"] = ((d.apt.astype(str) == "LIRF") & d.AOBT_3_flt.isna() & (C > MIN_C) & ((d.pred - C).abs() > AT_COPY) & d.p_imp.notna()).to_numpy()
    p = d.p_imp.astype(float).clip(0, 1).fillna(0)
    d["new"] = np.where(d.elig, p * C + (1 - p) * DAY, d.pred)
    return d


def recal(d, P, own):
    """Variant 2: the mixture with the two-bin LOFO copy share instead of p_imp."""
    if P is None:
        return d, None
    p, co = calibrate(d, P, own)
    C = d.d_sched.astype(float).to_numpy()
    d["new"] = np.where(d.elig, p * C + (1 - p) * DAY, d.pred)
    return d, co


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--calib", default="none", choices=["none", "bin2"])
    ap.add_argument("--fit_base", default="ar_134na_dup")
    ap.add_argument("--pairs_base", default=None, help="id of the fit base in the pseudo phases (default: --fit_base; the bundle id of the rehearsal phases can differ from the shipped one)")
    a = ap.parse_args()
    P = pool(a.fit_base, a.pairs_base) if a.calib == "bin2" else None
    own = PHASE if PHASE.startswith("pseudo") else None
    run = f"hgf50-{PHASE or 'rank'}-{a.id}"
    report(run, 0, 2, "oof", purpose="Does setting LIRF no-plan rows beyond 50 k s to the gate's copy/+1-day mixture lower E_final RMSE?")
    res = {"id": a.id, "base": a.base, "phase": PHASE, "step": "hg_far50", "day": DAY, "min_c": MIN_C, "calib": a.calib,
           "fit_base": a.fit_base if P is not None else None, "pool_rows": None if P is None else len(P)}
    of = OUTD / "oof" / f"{a.base}.parquet"
    if of.exists():
        o = pd.read_parquet(of)[["MVT_ID_mvt", "fold", "y", "pred"]]
        d, co = recal(frame(o, pd.read_parquet(OUTD / "oof" / "rc_v34_more18.parquet")), P, own)
        res["oof_coefs"] = co
        y = d.y.to_numpy(float)
        res.update(base_cv_rmse=rmse(d.pred, y), cv_rmse=rmse(d.new, y), oof_moved=int(d.elig.sum()),
                   folds={f: {"base": rmse(d.pred[d.fold == f], y[d.fold == f]), "new": rmse(d.new[d.fold == f], y[d.fold == f])} for f in d.fold.unique()})
        pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt, "fold": d.fold, "y": y, "pred": d.new}).to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    report(run, 1, 2, "test")
    tf = OUTD / "test" / f"{a.base}.parquet"
    if tf.exists():
        t = pd.read_parquet(tf).rename(columns={"TAXITIME_SEC_mvt": "pred"})[["MVT_ID_mvt", "pred"]]
        g = pd.read_parquet(OUTD / "test" / "rc_v34_more18_parts.parquet")
        d, co = recal(frame(t, g), P, own)
        res["test_coefs"] = co
        if d.p_imp.isna().mean() > 0.01:
            raise SystemExit(f"rc_v34_more18_parts: p_imp missing on {d.p_imp.isna().mean():.1%} of test rows")
        res.update(test_moved=int(d.elig.sum()), test_shift_rms=float(np.sqrt(np.mean((d.new - d.pred) ** 2))))
        template = pd.read_parquet(TEMPLATE)
        template["TAXITIME_SEC_mvt"] = template.MVT_ID_mvt.map(pd.Series(d.new.to_numpy(), index=d.MVT_ID_mvt.to_numpy())).astype(float)
        path = OUTD / "test" / f"{a.id}.parquet"
        template.to_parquet(path, index=False)
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from baseline import validate_submission
        validate_submission(path)
    if not of.exists() and not tf.exists():
        report(run, 2, 2, f"no {a.base} in this phase", finished=True)
        raise SystemExit(f"{a.base}: neither oof nor test in {OUTD}")
    save_result(res)
    print(json.dumps(res, indent=1))
    report(run, 2, 2, f"done cv {res.get('cv_rmse', float('nan')):.3f} (base {res.get('base_cv_rmse', float('nan')):.3f}), "
           f"test moved {res.get('test_moved', 0)}", finished=True)
    if ocommon._progress:
        ocommon._progress.report(run if run.startswith("eurocontrol-") else "eurocontrol-" + run, "EUROCONTROL " + run, 2, 2,
                                 detail="done", finished=True, outcome="success")


if __name__ == "__main__":
    main()
