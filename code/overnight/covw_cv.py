# SPDX-License-Identifier: GPL-3.0-or-later
"""Coverage-weighted CV: re-weight OOF rows so each airport x month cell has the 2026 test's ADS-B match share.

2025 CV coverage differs from 2026 test (LEMD Jan 0% -> 97%, EGLL Jul 27% -> 98%, EDDF 92% -> 74% ...), so plain CV
mis-weights anything that acts through ADS-B. Weight of a matched row = test_share / cv_share of its cell, unmatched
= (1 - test_share) / (1 - cv_share); cells with no CV rows of the needed kind keep weight 1 (cannot be emulated).
Compares runs by weighted RMSE with a paired day-block bootstrap.

  python covw_cv.py ar_22_mlp ar_15_adsb_member ar_17_adsbdec_member
"""
import sys

import numpy as np
import pandas as pd

from ocommon import OUTD, PROC


def cov_table():
    """Per-departure meta with ok (ADS-B matched), mo (jan/jul half) and the train/test match share of its cell."""
    m = pd.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "apt", "month", "split", "MVT_TIME_UTC_mvt"])
    a = pd.read_parquet(PROC / "features_dep_adsb2fullrel.parquet", columns=["MVT_ID_mvt", "a2_ok"])
    m = m.merge(a, on="MVT_ID_mvt", how="left")
    m["ok"] = m.a2_ok.fillna(0).astype(bool)
    m["mo"] = np.where(m.month <= 3, "jan", "jul")
    te = m[m.split == "rank"].groupby(["apt", "mo"]).ok.mean().rename("t")
    cv = m[(m.split == "train") & m.month.isin([1, 7])].groupby(["apt", "mo"]).ok.mean().rename("c")
    return m.drop(columns="a2_ok").join(cv, on=["apt", "mo"]).join(te, on=["apt", "mo"])


def cov_weights(ids, table=None):
    """Importance weights for OOF rows `ids` (MVT_ID_mvt) so each airport x month cell has the test's match share."""
    t = (table if table is not None else cov_table()).set_index("MVT_ID_mvt").loc[np.asarray(ids)]
    w = np.where(t.ok, t.t / t.c, (1 - t.t) / (1 - t.c))
    return np.clip(np.where(np.isfinite(w) & (w > 0), w, 1.0), 0, 20)


if __name__ == "__main__":
    m = cov_table()
    runs = sys.argv[1:]
    d = None
    for r in runs:
        o = pd.read_parquet(OUTD / "oof" / f"{r}.parquet")[["MVT_ID_mvt", "y", "pred"]].rename(columns={"pred": r})
        d = o if d is None else d.merge(o.drop(columns="y"), on="MVT_ID_mvt")
    d = d.merge(m[["MVT_ID_mvt", "apt", "mo", "ok", "c", "t", "MVT_TIME_UTC_mvt"]], on="MVT_ID_mvt")
    d["w"] = cov_weights(d.MVT_ID_mvt, m)
    cv = d.groupby(["apt", "mo"]).c.first(); te = d.groupby(["apt", "mo"]).t.first()
    print(pd.concat([cv, te], axis=1).round(2).T.to_string())
    day = (d.MVT_TIME_UTC_mvt // 86400).to_numpy()
    days = np.unique(day); rng = np.random.default_rng(0)
    trim = d.y <= 20000
    for r in runs:
        e2 = (d[r] - d.y) ** 2
        print(f"{r:28s} plain {np.sqrt(e2.mean()):.2f}  covw {np.sqrt((d.w * e2).sum() / d.w.sum()):.2f}  covw trim {np.sqrt((d.w * e2)[trim].sum() / d.w[trim].sum()):.2f}")
    b = runs[0]
    for r in runs[1:]:
        for nm, msk in [("all", np.ones(len(d), bool)), ("trim", trim.to_numpy())]:
            dd = (d.w * ((d[r] - d.y) ** 2 - (d[b] - d.y) ** 2)).to_numpy()[msk]; ww = d.w.to_numpy()[msk]; dy = day[msk]
            s = pd.DataFrame({"dd": dd, "w": ww, "day": dy}).groupby("day").sum()
            idx = np.searchsorted(days, s.index)
            base_mse = (d.w * (d[b] - d.y) ** 2)[msk].sum() / ww.sum()
            def rm(sel):
                dm = s.dd.to_numpy()[sel].sum() / s.w.to_numpy()[sel].sum()
                bm = base_mse
                return np.sqrt(bm + dm) - np.sqrt(bm)
            pt = rm(np.arange(len(s)))
            bs = [rm(rng.integers(0, len(s), len(s))) for _ in range(500)]
            print(f"  {nm:4s} {r} - {b}: {pt:+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
