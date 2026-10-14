# SPDX-License-Identifier: GPL-3.0-or-later
"""ADS-B member features: features_dep_<src>.parquet (v2 events over every built day) -> features_dep_<src>rel.parquet.

Keeps the a2_* columns (MVT - event, s) with a2_* set to NaN where the match is not `ok` (|a2_to| > 300 s), and adds
each timing event minus the NM clock gaps d_aobt / d_lobt (trees cannot form differences; the member's target is
y - d_aobt, so a2_anchor - d_aobt is the direct "off-block moved by" signal) plus a2_ok.

  python features_adsb_rel.py --src adsb2full
"""
import argparse

import numpy as np
import pandas as pd
import polars as pl

from ocommon import PROC, report

EV = ["a2_seen", "a2_gs1", "a2_disp15", "a2_anchor", "a2_roll", "a2_hold_end"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="adsb2full")
    a = ap.parse_args()
    run = f"features-{a.src}rel"
    report(run, 0, 2, "loading", purpose="ADS-B event timings relative to the NM clocks, as CatBoost member features")
    f = pd.read_parquet(PROC / f"features_dep_{a.src}.parquet")
    b = pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "d_aobt", "d_lobt"]).to_pandas()
    d = f.merge(b, on="MVT_ID_mvt", how="left")
    ok = d.a2_seen.notna() & (d.a2_to.abs() <= 300)
    cols = [c for c in f.columns if c.startswith("a2_")]
    d.loc[~ok, cols] = np.nan
    d["a2_ok"] = ok.astype(float)
    for c in EV:
        for k in ("d_aobt", "d_lobt"):
            d[f"{c}_m_{k[2:]}"] = d[c] - d[k].astype(float)
    out = d.drop(columns=["d_aobt", "d_lobt"])
    out.to_parquet(PROC / f"features_dep_{a.src}rel.parquet", index=False)
    report(run, 2, 2, f"wrote features_dep_{a.src}rel.parquet: {len(out):,} rows, ok {ok.mean():.3f}", finished=True)


if __name__ == "__main__":
    main()
