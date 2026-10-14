# SPDX-License-Identifier: GPL-3.0-or-later
"""Twin anti-window tail post-step : LIRF departures without an NM flight, read through the flight's
ARRIVAL twin at the destination challenge airport.

The organisers' join puts every NM-matched departure inside |BLOCK - LOBT| <= 3,606 s (all 2,062,577 plan rows; rc_v2_cliff_window).
A no-NM departure whose arrival twin (same FLIGHT_mvt, ADEP, ADES; arrival MVT 10 min-8 h later) carries an NM flight failed that join,
so its BLOCK sits OUTSIDE the twin's LOBT +- 3,606 s (97.4% of 897 such 2025 rows). With yl = MVT - LOBT(twin), ya = MVT - AOBT_3(twin),
d = MVT - SCHED:
  copy cell    |ya - yl| <= 3606, d > yl + 3606, d <= 30000: the actual off-block lies inside the window, so the recorded BLOCK is not it;
               the schedule copy is the usual artefact (2025 LIRF: 15/17 copies where the chain was not already at d) -> pred' = q*d + (1-q)*pred
  normal cell  |ya - yl| > 3606: the actual off-block itself explains the failed join; 0/75 copies at LIRF in 2025 -> pred' = reg
               (the rules chain's non-copy value, rc_v34_more18), where |pred - reg| > 300.
Label-free: twin fields come from the phase's own test file (arrivals are not masked) or, for the OOF, the 2025 training files.
No fitted parameter at q = 1 (2025 LOFO copy share of the cell .82-.90).

  EC_PHASE=final python twin_window.py --id ar_134naclh_tww --base ar_134naclh_hgf [--q 1.0]   (pipeline/39_post_steps.sh)
  EC_PHASE=pseudo2 python twin_window.py --id ar_134naclh_tww --base ar_134naclh_hgf      (a rehearsal phase; pseudo3-6 likewise)
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

import ocommon  # noqa: F401
from ocommon import OUTD, PHASE, RAW, ROOT, TEMPLATE, TEST_FILE, report, rmse, save_result

W, DMAX, AT_COPY, AT_REG = 3606.0, 30000.0, 600.0, 300.0
COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "LOBT_flt", "AOBT_3_flt"]
TIMES = ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]


def _sec(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return (s - pd.Timestamp("1970-01-01", tz="UTC" if s.dt.tz is not None else None)).dt.total_seconds()
    return s.astype(float)


def _load(path):
    """No-NM departures to a challenge airport and NM-matched arrivals of one file (label-free columns only)."""
    m = pd.read_parquet(path, columns=COLS)
    for c in TIMES:
        m[c] = _sec(m[c])
    dep = m[(m.PHASE_mvt == "DEP")]
    apts = set(dep.ADEP_mvt.dropna().unique())
    dep = dep[dep.FLIGHT_ID_mvt.isna() & dep.ADES_mvt.isin(apts)]
    arr = m[(m.PHASE_mvt == "ARR") & m.LOBT_flt.notna() & m.AOBT_3_flt.notna()]
    return dep, arr


def twins(paths):
    """MVT_ID -> yl, ya, d, apt for no-NM departures whose arrival twin has an NM flight."""
    deps, arrs = zip(*[_load(p) for p in paths])
    dep, arr = pd.concat(deps), pd.concat(arrs)
    a = arr[["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]]
    j = dep[["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]].merge(
        a, on=["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"], suffixes=("", "_a"))
    dt = j.MVT_TIME_UTC_mvt_a - j.MVT_TIME_UTC_mvt
    j = j[(dt > 600) & (dt < 8 * 3600)].sort_values("MVT_TIME_UTC_mvt_a").drop_duplicates("MVT_ID_mvt")
    return pd.DataFrame({"MVT_ID_mvt": j.MVT_ID_mvt.astype("int64").to_numpy(), "apt": j.ADEP_mvt.to_numpy(),
                         "yl": (j.MVT_TIME_UTC_mvt - j.LOBT_flt).to_numpy(), "ya": (j.MVT_TIME_UTC_mvt - j.AOBT_3_flt).to_numpy(),
                         "d": (j.MVT_TIME_UTC_mvt - j.SCHED_TIME_UTC_mvt).to_numpy()})


def step(pred, gate, tw, q):
    """pred: MVT_ID_mvt, pred (+ y, fold); gate: MVT_ID_mvt, reg. Returns pred with 'new', 'cell'."""
    d = pred.merge(tw, on="MVT_ID_mvt", how="left").merge(gate[["MVT_ID_mvt", "reg"]].drop_duplicates("MVT_ID_mvt"), on="MVT_ID_mvt", how="left")
    lirf = (d.apt == "LIRF").to_numpy()
    yl, ya, C, p, reg = (d[c].to_numpy(float) for c in ["yl", "ya", "d", "pred", "reg"])
    inside = np.abs(ya - yl) <= W
    copy = lirf & inside & (C > yl + W) & (C <= DMAX) & (np.abs(p - C) > AT_COPY)
    norm = lirf & ~inside & ~np.isnan(ya) & ~np.isnan(reg) & (np.abs(p - reg) > AT_REG)
    d["new"] = np.where(copy, q * C + (1 - q) * p, np.where(norm, reg, p))
    d["cell"] = np.select([copy, norm], ["copy", "normal"], "")
    return d


def _end(run, detail, outcome):
    """One terminal progress call, so the run's wake carries its real outcome (ocommon.report has no outcome)."""
    if ocommon._progress:
        ocommon._progress.report("eurocontrol-" + run, "EUROCONTROL " + run, 2, 2, detail=detail, finished=True, outcome=outcome)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--q", type=float, default=1.0)
    a = ap.parse_args()
    run = f"tww-{PHASE or 'rank'}-{a.id}"
    report(run, 0, 2, "oof", purpose="Does reading no-NM LIRF departures through their arrival twin's NM window (copy inside, normal outside) lower E_final RMSE?")
    res = {"id": a.id, "base": a.base, "phase": PHASE, "step": "twin_window", "q": a.q, "window": W, "dmax": DMAX}
    of = OUTD / "oof" / f"{a.base}.parquet"
    if of.exists():
        o = pd.read_parquet(of)[["MVT_ID_mvt", "fold", "y", "pred"]]
        tw = twins(sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet"))))
        d = step(o, pd.read_parquet(OUTD / "oof" / "rc_v34_more18.parquet"), tw, a.q)
        y = d.y.to_numpy(float)
        res.update(base_cv_rmse=rmse(d.pred, y), cv_rmse=rmse(d.new, y), oof_moved=d.cell.value_counts().drop("", errors="ignore").to_dict(),
                   folds={f: {"base": rmse(d.pred[d.fold == f], y[d.fold == f]), "new": rmse(d.new[d.fold == f], y[d.fold == f])} for f in d.fold.unique()})
        pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt, "fold": d.fold, "y": y, "pred": d.new}).to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    report(run, 1, 2, "test")
    tf = OUTD / "test" / f"{a.base}.parquet"
    if tf.exists():
        t = pd.read_parquet(tf).rename(columns={"TAXITIME_SEC_mvt": "pred"})[["MVT_ID_mvt", "pred"]]
        t["MVT_ID_mvt"] = t.MVT_ID_mvt.astype("int64")
        g = pd.read_parquet(OUTD / "test" / "rc_v34_more18_parts.parquet")
        g["MVT_ID_mvt"] = g.MVT_ID_mvt.astype("int64")
        d = step(t, g, twins([TEST_FILE]), a.q)
        res.update(test_moved=d.cell.value_counts().drop("", errors="ignore").to_dict(),
                   test_shift_rms=float(np.sqrt(np.mean((d.new - d.pred) ** 2))))
        template = pd.read_parquet(TEMPLATE)
        template["TAXITIME_SEC_mvt"] = template.MVT_ID_mvt.astype("int64").map(pd.Series(d.new.to_numpy(), index=d.MVT_ID_mvt.to_numpy())).astype(float)
        path = OUTD / "test" / f"{a.id}.parquet"
        template.to_parquet(path, index=False)
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from baseline import validate_submission
        validate_submission(path)
    if not of.exists() and not tf.exists():
        _end(run, f"no {a.base} in this phase", "failure")
        raise SystemExit(f"{a.base}: neither oof nor test in {OUTD}")
    save_result(res)
    print(json.dumps(res, indent=1))
    done = (f"done cv {res.get('cv_rmse', float('nan')):.3f} (base {res.get('base_cv_rmse', float('nan')):.3f}), "
            f"test moved {res.get('test_moved', {})}")
    report(run, 2, 2, done)
    _end(run, done, "success")


if __name__ == "__main__":
    main()
