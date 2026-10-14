# SPDX-License-Identifier: GPL-3.0-or-later
"""Cross-validation of the witness post-step (scripts/witness_post.py) on 2025 labels.

Takes a base prediction with labels (an out-of-fold file: MVT_ID_mvt, y, pred; or a rehearsal-phase test file plus that phase's
truth.parquet), applies the post-step month by month with margins fitted on the other 2025 months (leave-one-month-out, so no row's
own label sets its margin), and reports the RMSE change with a paired day-block bootstrap interval.

  python scripts/witness_eval.py --base outputs/overnight_final/oof/ar_134naclh_tww.parquet [--ideas zf3,3,123] [--boot 3000]
  python scripts/witness_eval.py --base outputs/overnight_pseudo2/test/ar_134naclh_tww.parquet --truth data/pseudo2/truth.parquet
      -> prints one line per idea set; --out writes the numbers as json.

The plan-window clip (idea 3) uses the 2025 edge, 3606 s (--win), as the shipped file does.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import witness_post as W  # noqa: E402

ROOT = W.ROOT


def load(base, truth):
    d = pd.read_parquet(base if Path(base).is_absolute() else ROOT / base)
    if truth:
        t = pd.read_parquet(truth if Path(truth).is_absolute() else ROOT / truth).rename(columns={"TAXITIME_SEC_mvt": "y"})
        d = d.rename(columns={"TAXITIME_SEC_mvt": "pred"}).merge(t[["MVT_ID_mvt", "y"]], on="MVT_ID_mvt")
    d["MVT_ID_mvt"] = d.MVT_ID_mvt.astype("int64")
    return d[["MVT_ID_mvt", "y", "pred"]].drop_duplicates("MVT_ID_mvt")


def evaluate(base, ctx, ideas, boot, win, rng):
    c = ctx.reindex(base.MVT_ID_mvt)
    mo = c.mo.to_numpy()
    day = pd.to_datetime(c.mvt.to_numpy(), unit="s").floor("D").astype("int64").to_numpy()
    ud, inv = np.unique(day, return_inverse=True)
    y = base.y.to_numpy(float)
    p0 = base.pred.to_numpy(float)
    se0 = (p0 - y) ** 2
    d0 = np.bincount(inv, se0, len(ud))
    cnt = np.bincount(inv, minlength=len(ud))
    months = sorted(set(pd.Series(mo).dropna()))
    pars = {m: W.fit_params(ctx, exclude_month=m) for m in months}
    B = rng.multinomial(len(ud), np.ones(len(ud)) / len(ud), size=boot)
    den = B @ cnt
    res = {}
    for v in ideas:
        p = p0.copy()
        moved = {}
        for m, par in pars.items():
            i = mo == m
            new, mv = W.apply(pd.Series(p[i], index=base.MVT_ID_mvt.to_numpy()[i]), ctx, par, v, win=win)
            p[i] = new.to_numpy()
            for k, x in mv.items():
                moved[k] = moved.get(k, 0) + x
        se = (p - y) ** 2
        d1 = np.bincount(inv, se, len(ud))
        diff = np.sqrt(B @ d1 / den) - np.sqrt(B @ d0 / den)
        res[v] = dict(rmse0=float(np.sqrt(se0.mean())), rmse1=float(np.sqrt(se.mean())), delta=float(np.sqrt(se.mean()) - np.sqrt(se0.mean())),
                      ci=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], moved=moved)
    return dict(n=int(len(y)), months=months, variants=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="parquet with MVT_ID_mvt, y, pred (OOF) or MVT_ID_mvt, TAXITIME_SEC_mvt (with --truth)")
    ap.add_argument("--truth", default="", help="labels of a rehearsal phase (data/pseudoN/truth.parquet) for a test-side base")
    ap.add_argument("--ideas", default="zf3,3,123", help="comma-separated idea sets, as in witness_post.py --ideas")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--win", type=float, default=3606)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ctx = W.context(W.read_movements(sorted(glob.glob(str(W.RAW / "training_*.parquet"))), labels=True))
    R = evaluate(load(a.base, a.truth), ctx, a.ideas.split(","), a.boot, a.win, np.random.default_rng(0))
    for v, r in R["variants"].items():
        print(f"ideas {v:4s} N={R['n']} rmse {r['rmse0']:.3f} -> {r['rmse1']:.3f}  d {r['delta']:+.3f} "
              f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] moved {r['moved']}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
