# SPDX-License-Identifier: GPL-3.0-or-later
"""Calibrate a chain's predictions to the published PRC monthly taxi-out per airport x runway.

PRC's (TOT_REF_TIME + TOT_ADD_TIME) / TOT_FLT per (airport, runway, month) is the mean of our label over its valid
sample (taxi-out <= 120 min, de-iced flights dropped, valid P10 reference). Implied label mean of a cell =
PRC mean + offset, where offset = mean(label | 0 < y <= 7200) - PRC mean of the same airport x runway in 2025:
  test (2026 month m): offset from 2025 month m (same season);
  CV (2025 month m):   offset from the neighbouring 2025 months (m-1, m+1; Jan uses Feb and Dec) so the fold's
                        own labels never enter.
Shift c = alpha * (implied - mean pred over rows with pred <= PMAX) is added to regressor rows (raw p_imp < 0.5,
pred <= PMAX) of the cell. January at the de-icing-after-off-block airports (EDDF, EDDM, LFPG, LSZH) is skipped:
PRC drops de-iced flights there, so its January mean is not our label's.

python prc_cal.py --id ar_05_prc_cal --base ar_03_adsb2_stack --alpha 0.5
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import polars as pl

from ocommon import OUTD, PROC, ROOT, report, rmse, save_result, TEMPLATE

DEICE = {"EDDF", "EDDM", "LFPG", "LSZH"}
NB = {1: [2, 12], 7: [6, 8]}


def prc_table():
    d = pd.read_csv(ROOT / "data/external/prc_dashboard/runway_tot_ref_time_tot_add_time_tot_flt_type_avg_txot_time.csv")
    d = d[d.MONTH.notna()].drop_duplicates(["AIRPORT", "YEAR", "MONTH", "RUNWAY"])
    d["prc"] = (d.TOT_REF_TIME + d.TOT_ADD_TIME) / d.TOT_FLT * 60
    return d.rename(columns={"AIRPORT": "apt", "YEAR": "year", "MONTH": "month", "RUNWAY": "rwy", "TOT_FLT": "flt"})[["apt", "year", "month", "rwy", "prc", "flt"]].astype({"month": int})


def shifts(rows, prc, lab, year, off_months, a):
    """rows: MVT_ID_mvt, apt, rwy, month, pred, reg_row. Returns per-row shift."""
    p = prc[prc.year == year]
    out = np.zeros(len(rows))
    cells = []
    for (apt, rwy, m), g in rows.groupby(["apt", "rwy", "month"], observed=True):
        if (m == 1 and apt in DEICE and a.skip_deice) or len(g) < a.min_rows:
            continue
        pc = p[(p.apt == apt) & (p.rwy == rwy) & (p.month == m)]
        if pc.empty or pc.flt.iloc[0] < a.min_flt or not np.isfinite(pc.prc.iloc[0]):
            continue
        om = off_months(m)
        l25 = lab[(lab.apt == apt) & (lab.rwy == rwy) & lab.month.isin(om)]
        p25 = prc[(prc.year == 2025) & (prc.apt == apt) & (prc.rwy == rwy) & prc.month.isin(om)]
        p25 = p25[p25.flt > 0]
        if len(l25) < a.min_rows or p25.empty:
            continue
        off = l25.y.mean() - np.average(p25.prc, weights=p25.flt)  # both over the same months
        implied = pc.prc.iloc[0] + off
        body = g.pred <= a.pmax
        c = a.alpha * (implied - g.pred[body].mean())
        c = float(np.clip(c, -a.cap, a.cap))
        idx = g.index[body & g.reg_row]
        out[idx] = c
        cells.append(dict(apt=apt, rwy=rwy, month=int(m), n=len(g), n_shift=len(idx), c=round(c, 1), implied=round(implied, 1), off=round(off, 1)))
    return out, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--base", required=True, help="chain/stack id with oof/<base>.parquet and test/<base>.parquet")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--pmax", type=float, default=7200)
    ap.add_argument("--min_flt", type=int, default=200)
    ap.add_argument("--min_rows", type=int, default=100)
    ap.add_argument("--cap", type=float, default=150, help="max |shift| per cell, s")
    ap.add_argument("--skip_deice", type=int, default=1)
    ap.add_argument("--apts", default="", help="comma list: calibrate only these airports")
    a = ap.parse_args()
    run = f"prc-cal-{a.id}"
    report(run, 0, 3, "loading", purpose="Does shifting each airport x runway x month cell toward the published PRC mean taxi-out cut RMSE?")
    prc = prc_table()
    f = pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "apt", "RUNWAY_mvt", "month", "split", "TAXITIME_SEC_mvt"]).to_pandas()
    f = f.rename(columns={"RUNWAY_mvt": "rwy", "TAXITIME_SEC_mvt": "y"})
    lab = f[(f.split == "train") & (f.y > 0) & (f.y <= 7200)][["apt", "rwy", "month", "y"]]
    keep = set(a.apts.split(",")) if a.apts else None

    oof = pd.read_parquet(OUTD / "oof" / f"{a.base}.parquet")[["MVT_ID_mvt", "fold", "y", "pred"]]
    pimp = pd.read_parquet(OUTD / "oof" / "rc_v4_gate2_rules.parquet")[["MVT_ID_mvt", "p_imp"]]
    o = oof.merge(f[["MVT_ID_mvt", "apt", "rwy", "month"]], on="MVT_ID_mvt", how="left").merge(pimp, on="MVT_ID_mvt", how="left")
    o["reg_row"] = o.p_imp.fillna(0) < 0.5
    if keep:
        o["reg_row"] &= o.apt.isin(keep)
    s_oof, cells_oof = shifts(o, prc, lab, 2025, lambda m: NB[m], a)
    y = o.y.to_numpy()
    new = np.clip(o.pred.to_numpy() + s_oof, 0, None)
    res = {"id": a.id, "model": "prc_cal", "base": a.base, "alpha": a.alpha, "pmax": a.pmax, "cap": a.cap, "skip_deice": a.skip_deice, "apts": a.apts,
           "cv_rmse": rmse(new, y), "cv_trim_rmse": rmse(new[y <= 20000], y[y <= 20000]), "cv_clean_rmse": rmse(new[y <= 7200], y[y <= 7200]),
           "base_cv_rmse": rmse(o.pred, y), "folds": {}, "per_airport": {}, "oof_shift_rms": float(np.sqrt(np.mean(s_oof ** 2)))}
    for fo in ("jan", "jul"):
        k = (o.fold == fo).to_numpy()
        res["folds"][fo] = {"rmse": rmse(new[k], y[k]), "base_rmse": rmse(o.pred.to_numpy()[k], y[k]), "n": int(k.sum())}
    for apt, g in o.groupby("apt"):
        k = g.index.to_numpy()
        res["per_airport"][apt] = {"base": rmse(o.pred.to_numpy()[k], y[k]), "new": rmse(new[k], y[k]), "shift_rms": float(np.sqrt(np.mean(s_oof[k] ** 2)))}
    pd.DataFrame({"MVT_ID_mvt": o.MVT_ID_mvt, "fold": o.fold, "y": y, "pred": new}).to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    report(run, 1, 3, f"CV {res['cv_rmse']:.3f} (base {res['base_cv_rmse']:.3f})")

    tfile = OUTD / "test" / f"{a.base}.parquet"
    if tfile.is_file():
        t = pd.read_parquet(tfile).rename(columns={"TAXITIME_SEC_mvt": "pred"})
        tp_raw = pd.read_parquet(OUTD / "test" / "mix_v18_fs3_deep_raw.parquet")[["MVT_ID_mvt", "p_imp"]]
        t = t.merge(f[["MVT_ID_mvt", "apt", "rwy", "month"]], on="MVT_ID_mvt", how="left").merge(tp_raw, on="MVT_ID_mvt", how="left")
        t["reg_row"] = t.p_imp.fillna(0) < 0.5
        if keep:
            t["reg_row"] &= t.apt.isin(keep)
        s_t, cells_t = shifts(t, prc, lab, 2026, lambda m: [m], a)
        tp = np.clip(t.pred.to_numpy() + s_t, 0, None)
        template = pd.read_parquet(TEMPLATE)
        template["TAXITIME_SEC_mvt"] = template.MVT_ID_mvt.astype("int64").map(pd.Series(tp, index=t.MVT_ID_mvt.to_numpy())).to_numpy()
        path = OUTD / "test" / f"{a.id}.parquet"
        template.to_parquet(path, index=False)
        sys.path.insert(0, str(ROOT / "scripts"))
        from baseline import validate_submission
        validate_submission(path)
        res.update(test_shift_rms=float(np.sqrt(np.mean(s_t ** 2))), test_cells=cells_t)
    res["oof_cells"] = cells_oof
    save_result(res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_airport", "oof_cells", "test_cells")}, indent=1))
    print(pd.DataFrame(res["per_airport"]).T.round(2).to_string())
    if "test_cells" in res:
        tc = pd.DataFrame(res["test_cells"])
        print(tc.groupby(["apt", "month"]).apply(lambda g: pd.Series({"n": g.n.sum(), "c_mean": np.average(g.c, weights=g.n_shift + 1e-9)})).round(1).to_string())
    report(run, 3, 3, f"CV {res['cv_rmse']:.3f} (base {res['base_cv_rmse']:.3f})", finished=True)


if __name__ == "__main__":
    main()
