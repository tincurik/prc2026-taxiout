# SPDX-License-Identifier: GPL-3.0-or-later
"""Departure-stream context features for the stackers: what the other departures of the same airport and hour say about
congestion, which the per-flight features miss (docs/model.md; -1.14 CV RMSE at the stacker average).
Label-free window aggregates of each airport's movement stream:
  delay vs SCHED / EOBT / LOBT of the other departures (+-30 / +-60 min, previous / next hour, +-3 h, same runway +-30 min),
  share of NM AOBT values off the airport's synthetic mode, overdue backlog (scheduled minus departed, last 1 h / 3 h / next 1 h),
  throughput vs the airport x hour typical, arrivals' delay at in-block, day level of the delay vs schedule,
and the neighbours' base-chain predictions with their clock-minus-prediction gaps (base = the stackers' --base: its OOF on the
Jan/Jul rows, its test predictions on the test rows, so both sides see the same kind of neighbour prediction). Phase-aware
(EC_PHASE). Writes PROC/features_dep_<out>.parquet for every departure (NaN where a feature has no source rows).
  [EC_PHASE=pseudo2] python features_ctx.py [--base rc_v34_more18] [--out ctx]
"""
import argparse

import numpy as np
import pandas as pd
import polars as pl

from features3 import keyed_window_mean
from ocommon import OUTD, PROC, report

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="rc_v34_more18")
ap.add_argument("--out", default="ctx")
a = ap.parse_args()
RUN = f"features-ctx-{a.out}"
report(RUN, 0, 4, "loading", purpose="Departure-stream context (delay windows, backlog, neighbours' predictions) as stacker features")
o = pd.read_parquet(OUTD / "oof" / f"{a.base}.parquet")[["MVT_ID_mvt", "pred"]]
t_ = pd.read_parquet(OUTD / "test" / f"{a.base}.parquet")[["MVT_ID_mvt", "TAXITIME_SEC_mvt"]].rename(columns={"TAXITIME_SEC_mvt": "pred"})
BP = pd.concat([o, t_[~t_.MVT_ID_mvt.isin(o.MVT_ID_mvt)]], ignore_index=True)
BP["MVT_ID_mvt"] = BP.MVT_ID_mvt.astype("int64")
mv = pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
                     "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "EOBT_1_flt", "LOBT_flt", "AOBT_3_flt"]).to_pandas()
dep = mv[mv.PHASE_mvt == "DEP"].reset_index(drop=True)
arr = mv[mv.PHASE_mvt == "ARR"].reset_index(drop=True)
del mv
dep["apt"] = dep.ADEP_mvt.astype(str); arr["apt"] = arr.ADES_mvt.astype(str)
apts = sorted(set(dep.apt) | set(arr.apt)); acode = {k: i for i, k in enumerate(apts)}
dk = dep.apt.map(acode).to_numpy(np.int64); ak = arr.apt.map(acode).to_numpy(np.int64)
rk = pd.Series(dep.apt + "|" + dep.RUNWAY_mvt.astype(str)).astype("category").cat.codes.to_numpy(np.int64)
t = dep.MVT_TIME_UTC_mvt.to_numpy(np.int64)
f64 = lambda s: s.astype("float64").to_numpy()
dsched = np.clip(t - f64(dep.SCHED_TIME_UTC_mvt), -1800, 10800)
deobt = np.clip(t - f64(dep.EOBT_1_flt), -1800, 7200)
dlobt = np.clip(t - f64(dep.LOBT_flt), -1800, 7200)
daobt = t - f64(dep.AOBT_3_flt)
daobt = np.where((daobt >= 0) & (daobt <= 7200), daobt, np.nan)
mode = pd.Series(np.round(daobt / 30) * 30).groupby(dep.apt).agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
aoff = np.where(np.isnan(daobt), np.nan, (np.abs(daobt - dep.apt.map(mode).to_numpy(float)) > 120).astype(float))
C = {}


def win(name, key, v, lo, hi):
    m, n = keyed_window_mean(key, t, v, key, t, lo, hi, self_v=v)
    C[name] = m.astype(np.float32)


W = {"pm30": (-1800, 1800), "pm60": (-3600, 3600), "prev60": (-3600, 0), "next60": (0, 3600), "pm180": (-10800, 10800)}
for vn, v in (("dsched", dsched), ("deobt", deobt), ("dlobt", dlobt), ("aoff", aoff)):
    for wn, (lo, hi) in W.items():
        if wn == "pm180" and vn not in ("dsched", "aoff"):
            continue
        win(f"cx_{vn}_{wn}", dk, v, lo, hi)
    win(f"cx_{vn}_rwy_pm30", rk, v, -1800, 1800)
    if vn == "dsched":
        med = pd.Series(v).groupby(dep.apt).transform("median").to_numpy()
        for wn in ("pm30", "pm60", "pm180"):
            C[f"cx_dsched_{wn}_dev"] = C[f"cx_dsched_{wn}"] - med
report(RUN, 1, 4, "stream windows")
one = np.ones(len(dep))
hs = ~np.isnan(f64(dep.SCHED_TIME_UTC_mvt))
st = f64(dep.SCHED_TIME_UTC_mvt)[hs].astype(np.int64); sk = dk[hs]
for w in (3600, 10800):
    _, ns = keyed_window_mean(sk, st, one[hs], dk, t, -w, 0)
    _, nm = keyed_window_mean(dk, t, one, dk, t, -w, 0)
    C[f"cx_backlog_{w // 60}"] = ns - nm
_, ns_n = keyed_window_mean(sk, st, one[hs], dk, t, 0, 3600)
_, nm_n = keyed_window_mean(dk, t, one, dk, t, 0, 3600)
C["cx_backlog_next60"] = ns_n - nm_n
_, n60 = keyed_window_mean(dk, t, one, dk, t, -3600, 3600)
hr = (t // 3600) % 24
C["cx_thru_ratio"] = n60 / np.maximum(pd.Series(n60).groupby([dep.apt, hr]).transform("mean").to_numpy(), 1)
ta = arr.MVT_TIME_UTC_mvt.to_numpy(np.int64)
adel = np.clip(f64(arr.BLOCK_TIME_UTC_mvt) - f64(arr.SCHED_TIME_UTC_mvt), -3600, 10800)
for wn, (lo, hi) in (("pm60", (-3600, 3600)), ("prev60", (-3600, 0))):
    m, _ = keyed_window_mean(ak, ta, adel, dk, t, lo, hi)
    C[f"cx_arrdel_{wn}"] = m
day = t // 86400
for vn, v in (("dsched", dsched), ("aoff", aoff)):
    C[f"cx_day_{vn}"] = pd.Series(v).groupby([dep.apt, day]).transform("mean").to_numpy()
C["cx_day_dsched_dev"] = C["cx_day_dsched"] - pd.Series(dsched).groupby(dep.apt).transform("median").to_numpy()
report(RUN, 2, 4, "backlog / arrivals / day level")
pr = dep[["MVT_ID_mvt"]].merge(BP, on="MVT_ID_mvt", how="left").pred.to_numpy(float)
for vn, v in (("pred", pr), ("dsched_m_pred", dsched - pr), ("daobt_m_pred", np.clip(daobt, 0, 3600) - pr),
              ("deobt_m_pred", deobt - pr), ("dlobt_m_pred", dlobt - pr)):
    for wn in ("pm30", "pm60"):
        lo, hi = W[wn]; win(f"cx_nb_{vn}_{wn}", dk, v, lo, hi)
    win(f"cx_nb_{vn}_rwy_pm30", rk, v, -1800, 1800)
C["own_m_nbpred_pm30"] = pr - C["cx_nb_pred_pm30"]
report(RUN, 3, 4, f"{len(C)} features, base preds on {np.isfinite(pr).sum():,} departures")
out = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in C.items()})
out.insert(0, "MVT_ID_mvt", dep.MVT_ID_mvt.to_numpy())
out.to_parquet(PROC / f"features_dep_{a.out}.parquet", index=False)
print(out.shape, "pred coverage", round(float(np.isfinite(pr).mean()), 3), flush=True)
report(RUN, 4, 4, f"wrote features_dep_{a.out}.parquet {out.shape}", finished=True)
