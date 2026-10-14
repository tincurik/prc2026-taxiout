# SPDX-License-Identifier: GPL-3.0-or-later
"""Clock override: when the NM off-block clock says the block is over ~5.5 h before take-off, trust it.

Rows with MVT - AOBT_3 > THRESH are date-shifted records (NM AOBT, EOBT and the handler block all a day before MVT;
2025 analogue AEZ146R: MVT-AOBT 87181, y 87002). Tree models cannot extrapolate there and predict a normal taxi
(~4.4k). pred = MVT - AOBT_3 on those rows. The 2026 inputs hold the same pattern; the tree models predict a normal taxi on it.
Writes an OOF + result json (cv) and, with --test, a test file.

python clockfix.py --src pp_blend_v11_cf_a05 --id clk_blend_v11 --test final_blend_v11
"""
import argparse
import json

import numpy as np
import pandas as pd
import polars as pl

from ocommon import OUTD, PROC, REP, rmse, save_result

THRESH = 20000
# Rule 2, gate hold (retuned on the 2025 OOF): NM AOBT far after LOBT while
# LOBT is far before take-off -> the handler block sits near LOBT; move pred halfway to MVT - LOBT.
HOLD_GAP, HOLD_ML, HOLD_W = 4800, 4000, 0.5
# Rule 3, LIRF schedule copy: LIRF departures without a flight plan leaving
# 20k-50k s after schedule carry BLOCK = SCHED in 48/50 rows of 2025 (the other 2 are ordinary taxis); the mixture
# shrinks them toward ~1.5k. pred = MVT - SCHED there. Above 50k s the alternative is a 24 h rollover, left to the model.
LIRF_LO, LIRF_HI = 20000, 50000
# Rule 3b: above 50k the alternative to a copy is the "datestamp" record (BLOCK = SCHED's local date + the real block time of
# day, y ~ 86400 + taxi), which needs MVT on a later Europe/Rome date than SCHED (12/12 in 2025). Without that cross the copy
# is the only long mechanism left (2025 LIRF no-plan, no cross, > 20k: 21 copies, 2 normal, 0 datestamps).


def clocks():
    c = (pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt"])
         .filter(pl.col("PHASE_mvt") == "DEP")
         .select("MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", da=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).cast(pl.Float64),
                 ml=(pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).cast(pl.Float64),
                 gap=(pl.col("AOBT_3_flt") - pl.col("LOBT_flt")).cast(pl.Float64),
                 ds=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).cast(pl.Float64),
                 lirf_np=(pl.col("ADEP_mvt") == "LIRF") & pl.col("AOBT_3_flt").is_null()).to_pandas())
    rome = lambda s: pd.to_datetime(s, unit="s", utc=True).dt.tz_convert("Europe/Rome").dt.tz_localize(None).dt.normalize()
    c["xloc_rome"] = (rome(c.MVT_TIME_UTC_mvt) - rome(c.SCHED_TIME_UTC_mvt)).dt.days
    return c.drop(columns=["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])


def apply(df, pred, c, hold=True, lirf=False, nocross=False):
    """Returns (new pred, date-shift mask, hold mask, lirf mask)."""
    df = df.merge(c, on="MVT_ID_mvt", how="left")
    m1 = (df.da > THRESH).to_numpy()
    m2 = ((df.gap > HOLD_GAP) & (df.ml > HOLD_ML)).to_numpy() & ~m1 & hold
    out = np.where(m2, (1 - HOLD_W) * pred + HOLD_W * df.ml.to_numpy(), pred)
    hi = np.where((df.xloc_rome == 0).to_numpy() & nocross, np.inf, LIRF_HI)
    m3 = (df.lirf_np.fillna(False) & (df.ds > LIRF_LO) & (df.ds <= hi)).to_numpy() & lirf
    out = np.where(m3, df.ds.to_numpy(), out)
    return np.where(m1, df.da.to_numpy(), out), m1, m2, m3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--test", default="")
    ap.add_argument("--hold", type=int, default=1)
    ap.add_argument("--lirf", type=int, default=0)
    ap.add_argument("--nocross", type=int, default=0, help="rule 3b: no Europe/Rome date cross lifts the 50k cap")
    a = ap.parse_args()
    c = clocks()
    o = pd.read_parquet(OUTD / "oof" / f"{a.src}.parquet")
    pred, m, m2, m3 = apply(o[["MVT_ID_mvt"]], o.pred.to_numpy(), c, bool(a.hold), bool(a.lirf), bool(a.nocross))
    y = o.y.to_numpy()
    res = {"id": a.id, "model": "clockfix", "src": a.src, "note": f"{a.src} with pred = MVT-AOBT_3 where it exceeds {THRESH}s"
           + (f"; hold rule AOBT_3-LOBT>{HOLD_GAP} & MVT-LOBT>{HOLD_ML}: pred += {HOLD_W}*(MVT-LOBT - pred)" if a.hold else "")
           + (f"; LIRF no-plan {LIRF_LO}<MVT-SCHED<={LIRF_HI}: pred = MVT-SCHED" if a.lirf else "")
           + ("; rule 3b: no Rome date cross lifts the 50k cap" if a.nocross else ""),
           "n_override_oof": int(m.sum()), "n_hold_oof": int(m2.sum()), "n_lirf_oof": int(m3.sum()), "cv_rmse": rmse(pred, y), "cv_trim_rmse": rmse(pred[y <= 20000], y[y <= 20000]),
           "cv_clean_rmse": rmse(pred[y <= 7200], y[y <= 7200]), "cv_rmse_before": rmse(o.pred, y), "folds": {}}
    for f in ("jan", "jul"):
        k = (o.fold == f).to_numpy()
        res["folds"][f] = {"rmse": rmse(pred[k], y[k]), "clean_rmse": rmse(pred[k & (y <= 7200)], y[k & (y <= 7200)])}
    o2 = o.copy()
    o2["pred"] = pred
    o2.to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    if a.test:
        t = pd.read_parquet(OUTD / "test" / f"{a.test}.parquet")
        tv, tm, tm2, tm3 = apply(t[["MVT_ID_mvt"]].astype({"MVT_ID_mvt": "int64"}), t.TAXITIME_SEC_mvt.to_numpy(), c, bool(a.hold), bool(a.lirf), bool(a.nocross))
        res["n_override_test"] = int(tm.sum())
        res["n_hold_test"] = int(tm2.sum())
        res["n_lirf_test"] = int(tm3.sum())
        res["test_overrides"] = [{"MVT_ID_mvt": int(i), "before": float(b), "after": float(x)}
                                 for i, b, x in zip(t.MVT_ID_mvt[tm], t.TAXITIME_SEC_mvt[tm], tv[tm])]
        t["TAXITIME_SEC_mvt"] = tv
        t.to_parquet(OUTD / "test" / f"{a.id}.parquet", index=False)
    save_result(res)
    print(json.dumps({k: v for k, v in res.items() if k != "folds"}, indent=1))


if __name__ == "__main__":
    main()
