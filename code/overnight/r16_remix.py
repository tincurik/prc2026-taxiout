# SPDX-License-Identifier: GPL-3.0-or-later
"""Tail post-step: LIRF copy-direction re-mix.

At LIRF the chain predicts a mixture of the plan copy C = d_sched and the regressor's non-copy value R (rc_v34_more18's reg), with an
implied copy weight p_eff = (pred - R) / (C - R). On the 12-month honest pool that weight under-commits to copy: refitting the observed
share s* = (y - R) / (C - R) on p_eff (affine, G^2-weighted least squares, G = C - R, s* clipped to [-0.5, 1.5]) per d_sched band x
plan/no-plan, leave-one-fold-out, gives slopes ~1.05-1.2. This step applies pred' = R + clip(a + b * p_eff, 0, 1) * G to LIRF rows with
|G| > 600, d_sched < 20000 and 0.02 < p_eff < 0.98; every other row keeps the base prediction.

Coefficients come from the fit base's honest pool (final Jan/Jul OOF + pseudo2-6 held-out tests vs truth), never from the folds that
hold the row's own months: 2025 rows leave out their month's fold (final = Jan/Jul, pseudoK = its two months) and the current pseudo
phase's fold; 2026 rows (the final's test) use all six.

  EC_PHASE=final python r16_remix.py --id ar_134naclh_rmx --base ar_134na_dup [--fit_base ar_134na_dup]   (pipeline/39_post_steps.sh run; `pools` runs it in pseudo2..6)
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
EDGES = [-1e9, 1800, 3600, 7200, 20000]
FEAT = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "apt", "d_sched", "AOBT_3_flt"]


def feats(proc, ids):
    f = pd.read_parquet(proc / "features_dep.parquet", columns=FEAT)
    f = f[f.MVT_ID_mvt.isin(ids)].drop_duplicates("MVT_ID_mvt")
    sec = f.MVT_TIME_UTC_mvt
    sec = sec.astype("int64") // 10**9 if np.issubdtype(sec.dtype, np.datetime64) else sec.astype("int64")
    t = pd.to_datetime(sec, unit="s")
    return pd.DataFrame({"MVT_ID_mvt": f.MVT_ID_mvt.astype("int64"), "lirf": (f.apt.astype(str) == "LIRF").to_numpy(),
                         "plan": f.AOBT_3_flt.notna().to_numpy(), "C": f.d_sched.astype(float).to_numpy(),
                         "year": t.dt.year.to_numpy(), "month": t.dt.month.to_numpy()})


def frame(pred, parts, proc):
    """pred: MVT_ID_mvt, pred (+ y); parts: MVT_ID_mvt, reg. -> rows with C, R, G, p_eff, band, plan, lirf, year, month."""
    d = pred.merge(parts[["MVT_ID_mvt", "reg"]].drop_duplicates("MVT_ID_mvt"), on="MVT_ID_mvt", how="left")
    d = d.merge(feats(proc, d.MVT_ID_mvt.to_numpy()), on="MVT_ID_mvt", how="left")
    d["R"] = d.reg.astype(float)
    d["G"] = d.C - d.R
    ok = d.G.abs() > 1
    d["pe"] = np.where(ok, (d.pred - d.R) / d.G.where(ok, 1.0), 0.0)
    d["band"] = np.digitize(d.C.fillna(-1e10), EDGES)
    d["elig"] = (d.lirf.fillna(False).astype(bool) & d.R.notna() & d.C.notna() & (d.band >= 1) & (d.band < len(EDGES))
                 & (d.G.abs() > 600) & (d.pe > 0.02) & (d.pe < 0.98))
    return d


def pool(fit_base, pairs_base=None):
    """The fit base's 12-month honest pool, eligible rows only (cached; written atomically so parallel phases can share it)."""
    out = ROOT / f"outputs/r16_remix/pool_{fit_base}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    pairs_base = pairs_base or fit_base
    parts = []
    o = pd.read_parquet(ROOT / f"outputs/overnight_{SLOT}/oof/{fit_base}.parquet")[["MVT_ID_mvt", "y", "pred"]]
    g = pd.read_parquet(ROOT / f"outputs/overnight_{SLOT}/oof/rc_v34_more18.parquet")
    parts.append(frame(o, g, ROOT / f"data/processed_{SLOT}").assign(fold="final"))
    for k in range(2, 7):
        t = pd.read_parquet(PAIRS / f"data/pseudo{k}/truth.parquet").rename(columns={"TAXITIME_SEC_mvt": "y"})
        p = pd.read_parquet(PAIRS / f"outputs/overnight_pseudo{k}/test/{pairs_base}.parquet").rename(columns={"TAXITIME_SEC_mvt": "pred"})
        g = pd.read_parquet(PAIRS / f"outputs/overnight_pseudo{k}/test/rc_v34_more18_parts.parquet")
        parts.append(frame(t.merge(p, on="MVT_ID_mvt"), g, PAIRS / f"data/processed_pseudo{k}").assign(fold=f"pseudo{k}"))
    d = pd.concat(parts, ignore_index=True)
    d = d[d.elig][["MVT_ID_mvt", "fold", "y", "R", "G", "pe", "band", "plan"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    d.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return d


def fit(P, band, plan, leave_out, min_rows=200):
    m = (P.band == band) & (P.plan == plan) & ~P.fold.isin(leave_out)
    if m.sum() < min_rows:
        return None
    q = P[m]
    G = q.G.to_numpy()
    s = np.clip((q.y.to_numpy() - q.R.to_numpy()) / G, -0.5, 1.5)
    w = np.abs(G)                      # sqrt of the G^2 weight
    X = np.c_[np.ones(len(q)), q.pe.to_numpy()]
    a, b = np.linalg.lstsq(X * w[:, None], s * w, rcond=None)[0]
    return float(a), float(b)


def apply(d, P, own_fold):
    """New predictions for frame d; coefficients per (band, plan, left-out folds)."""
    new = d.pred.to_numpy(float).copy()
    coefs = {}
    e = d.elig.to_numpy()
    lo = ["+".join(sorted({MONTH_FOLD[m] for m in [mm] if y == 2025} | ({own_fold} if own_fold else set())))
          for y, mm in zip(d.year.fillna(0).astype(int), d.month.fillna(0).astype(int))]
    key = (d.band.astype(str) + "|" + d.plan.fillna(False).astype(bool).astype(int).astype(str) + "|" + pd.Series(lo, index=d.index)).to_numpy()
    for k in pd.unique(key[e]):
        band, plan, leave = k.split("|")
        c = fit(P, int(band), plan == "1", [x for x in leave.split("+") if x])
        coefs[f"b{band}_{'plan' if plan == '1' else 'noplan'}_lo{leave or 'none'}"] = c
        if c is None:
            continue
        m = e & (key == k)
        s = np.clip(c[0] + c[1] * d.pe.to_numpy()[m], 0.0, 1.0)
        new[m] = np.clip(d.R.to_numpy()[m] + s * d.G.to_numpy()[m], 0, None)
    return new, coefs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--base", required=True, help="tail id with oof/<base>.parquet and/or test/<base>.parquet in this phase")
    ap.add_argument("--fit_base", default="ar_134na_dup", help="id whose 12-month honest pool fits the coefficients")
    ap.add_argument("--pairs_base", default=None, help="id of the fit base in the pseudo phases (default: --fit_base; the bundle id of the rehearsal phases can differ from the shipped one)")
    a = ap.parse_args()
    run = f"r16rmx-{PHASE or 'rank'}-{a.id}"
    purpose = "Does refitting LIRF's implied copy weight per d_sched band x plan (LOFO on the 12-month pool) lower E_final RMSE?"
    report(run, 0, 3, "pool", purpose=purpose)
    P = pool(a.fit_base, a.pairs_base)
    own = PHASE if PHASE.startswith("pseudo") else None
    res = {"id": a.id, "base": a.base, "fit_base": a.fit_base, "phase": PHASE, "step": "r16_remix", "pool_rows": len(P)}
    report(run, 1, 3, "oof")
    of = OUTD / "oof" / f"{a.base}.parquet"
    if of.exists():
        o = pd.read_parquet(of)[["MVT_ID_mvt", "fold", "y", "pred"]]
        g = pd.read_parquet(OUTD / "oof" / "rc_v34_more18.parquet")
        d = frame(o, g, PROC)
        new, co = apply(d, P, own)
        y = d.y.to_numpy(float)
        res.update(base_cv_rmse=rmse(d.pred, y), cv_rmse=rmse(new, y), oof_moved=int((np.abs(new - d.pred) > 1).sum()), oof_coefs=co,
                   folds={f: {"base": rmse(d.pred[d.fold == f], y[d.fold == f]), "new": rmse(new[d.fold == f], y[d.fold == f])}
                          for f in d.fold.unique()})
        pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt, "fold": d.fold, "y": y, "pred": new}).to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    report(run, 2, 3, "test")
    tf = OUTD / "test" / f"{a.base}.parquet"
    if tf.exists():
        t = pd.read_parquet(tf).rename(columns={"TAXITIME_SEC_mvt": "pred"})
        g = pd.read_parquet(OUTD / "test" / "rc_v34_more18_parts.parquet")
        d = frame(t[["MVT_ID_mvt", "pred"]], g, PROC)
        if d.R.isna().mean() > 0.01:
            raise SystemExit(f"rc_v34_more18_parts: reg missing on {d.R.isna().mean():.1%} of test rows")
        new, co = apply(d, P, own)
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
