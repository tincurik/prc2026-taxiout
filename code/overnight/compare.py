# SPDX-License-Identifier: GPL-3.0-or-later
"""Paired day-block bootstrap of RMSE differences between two OOF files (B - A; negative = B better).

Resamples validation DAYS with replacement (errors are correlated within a day), same days for both models.
python compare.py pp_mix_v2_fs2 pp_mix_v4_fs2_resid [--n 1000]
"""
import argparse

import numpy as np
import pandas as pd
import polars as pl

from ocommon import OUTD, PROC


def boot(a, b, n=1000):
    """Returns {subset: (point, lo, hi, p_better)} for B - A."""
    A = pd.read_parquet(OUTD / "oof" / f"{a}.parquet")[["MVT_ID_mvt", "y", "pred"]]
    B = pd.read_parquet(OUTD / "oof" / f"{b}.parquet")[["MVT_ID_mvt", "pred"]].rename(columns={"pred": "pb"})
    day = pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).to_pandas()
    d = A.merge(B, on="MVT_ID_mvt").merge(day, on="MVT_ID_mvt")
    d["day"] = d.MVT_TIME_UTC_mvt // 86400
    ea, eb = (d.pred - d.y) ** 2, (d.pb - d.y) ** 2
    rng = np.random.default_rng(0)
    out = {}
    for name, m in (("all", np.ones(len(d), bool)), ("trim y<=20k", (d.y <= 20000).to_numpy()), ("clean y<=7.2k", (d.y <= 7200).to_numpy())):
        g = pd.DataFrame({"day": d.day[m], "a": ea[m], "b": eb[m]}).groupby("day").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
        sa, sb, nn = g.a.to_numpy(), g.b.to_numpy(), g.n.to_numpy()
        diffs = []
        for _ in range(n):
            i = rng.integers(0, len(g), len(g))
            diffs.append(np.sqrt(sb[i].sum() / nn[i].sum()) - np.sqrt(sa[i].sum() / nn[i].sum()))
        diffs = np.array(diffs)
        point = np.sqrt(sb.sum() / nn.sum()) - np.sqrt(sa.sum() / nn.sum())
        out[name] = (float(point), float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975)), float(np.mean(diffs < 0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--n", type=int, default=1000)
    x = ap.parse_args()
    for name, (point, lo, hi, pb) in boot(x.a, x.b, x.n).items():
        print(f"{name:14s} {x.b} - {x.a}: {point:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  P(B better) {pb:.2f}")

if __name__ == "__main__":
    main()
