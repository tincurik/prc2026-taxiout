# SPDX-License-Identifier: GPL-3.0-or-later
"""prc_cal with offset schemes that transfer to unseen months (Feb/Jun 2026), a winter alpha and a configurable skip list.

prc_cal.py (unchanged) shifts each airport x runway x month cell's regressor rows by alpha * (PRC mean + offset - mean
pred), offset = label mean - PRC mean of 2025. For the 2026 test it takes the offset from the same 2025 month, which
carries that month's weather: LTFM Feb 2025 (9 snow days, PRC kept 79% of flights) has a runway 35L offset of +220 s
against ~5 s in other months, and Feb 2026 at LTFM had 2 snow days. The 12-month leave-one-month-out check
shows offsets do not transfer between winter months, while in Mar-Nov they do
and a larger alpha pays.

Offset schemes (--cv_offset for the OOF folds, --test_offset for the test rows):
  same   the same 2025 month (prc_cal's test scheme)
  nb     2025 months m-1 and m+1 (prc_cal's CV scheme, generalised to every month)
  inv    cell mean offset over the available 2025 months (never the target month itself when the target is 2025)
         + b_apt * (PRC invalid share of the target airport-month - their mean), b_apt >= 0 the airport's pooled
         runway-demeaned slope of offset on invalid share (1 - VALID_FL / TF, PRC airport taxi-out file); slope 0 where
         the airport's 2025 share spans < 0.02 (no de-icing season), target share clipped into the 2025 range;
         --oor 0.02 leaves a cell alone when the target share lies more than 0.02 outside that range (offset unidentified:
         LTFM Feb 2025 has .111 against <= .002 in every other 2025 month, and shifting it lost 150 MSE on pseudo2's Feb)
Cells are keyed by (airport, runway, year, month) of each row, so pseudo phases (2025 test months) and the final
(2026) both work. Defaults reproduce prc_cal.py.

  python prc_cal2.py --id ar_130_prc2 --base ar_129_dup --cv_offset inv --test_offset inv --alpha 0.5 --alpha_winter 0.25
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import polars as pl

import ocommon  # noqa: E402
from ocommon import OUTD, PROC, ROOT, report, rmse, save_result, TEMPLATE
from prc_cal import prc_table

APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
WINTER = {12, 1, 2}
INV_RANGE = 0.02


def invalid_share():
    rows = []
    for y in (2025, 2026):
        p = pd.read_csv(ROOT / f"data/external/prc/taxi_out_additional_time_{y}.csv")
        rows.append(p[p.APT_ICAO.isin(APTS)])
    p = pd.concat(rows).rename(columns={"APT_ICAO": "apt", "YEAR": "year", "MONTH_NUM": "month"})
    p["inv"] = 1 - p.VALID_FL / p.TF
    return p[["apt", "year", "month", "inv"]].astype({"year": int, "month": int})


def parse_skip(s):
    """'1:EDDF,EDDM;2:LTFM' -> {(1, 'EDDF'), (1, 'EDDM'), (2, 'LTFM')}"""
    out = set()
    for part in filter(None, s.split(";")):
        m, apts = part.split(":")
        out |= {(int(m), x) for x in apts.split(",") if x}
    return out


class Offsets:
    def __init__(self, lab, prc, inv, oor=0.0):
        self.oor = oor
        self.L = lab.groupby(["apt", "rwy", "month"]).y.agg(["sum", "size"]).reset_index()
        self.P = prc[(prc.year == 2025) & (prc.flt > 0) & np.isfinite(prc.prc)][["apt", "rwy", "month", "prc", "flt"]]
        self.C = self.L.merge(self.P, on=["apt", "rwy", "month"], how="inner")
        self.C["off"] = self.C["sum"] / self.C["size"] - self.C.prc
        self.C = self.C.merge(inv[inv.year == 2025][["apt", "month", "inv"]], on=["apt", "month"], how="left")
        self.inv = inv
        self.min_rows = 100

    def pooled(self, apt, rwy, months):
        """prc_cal's estimator: label mean over the months - flt-weighted PRC mean over the months"""
        l = self.L[(self.L.apt == apt) & (self.L.rwy == rwy) & self.L.month.isin(months)]
        p = self.P[(self.P.apt == apt) & (self.P.rwy == rwy) & self.P.month.isin(months)]
        if l["size"].sum() < self.min_rows or p.empty:
            return np.nan
        return float(l["sum"].sum() / l["size"].sum() - np.average(p.prc, weights=p.flt))

    def get(self, apt, rwy, year, m, scheme):
        own = year == 2025          # a 2025 target month (CV fold or pseudo-phase test) never uses its own labels
        if scheme == "same":
            return np.nan if own else self.pooled(apt, rwy, [m])
        if scheme == "nb":
            return self.pooled(apt, rwy, [(m - 2) % 12 + 1, m % 12 + 1])
        if scheme == "inv":
            ca = self.C[(self.C.apt == apt) & (self.C["size"] >= self.min_rows)]
            if own:
                ca = ca[ca.month != m]
            o = ca[ca.rwy == rwy]
            if len(o) < 4:
                return np.nan
            it = self.inv[(self.inv.apt == apt) & (self.inv.year == year) & (self.inv.month == m)].inv
            if self.oor and len(it) and np.isfinite(it.iloc[0]) and not ca.inv.min() - self.oor <= it.iloc[0] <= ca.inv.max() + self.oor:
                return np.nan   # target share outside the fitted months' range: no offset estimate, cell left alone
            if ca.inv.max() - ca.inv.min() < INV_RANGE:   # no de-icing season to learn a slope from (EGLL, EHAM, LEBL, LIRF)
                return float(o.off.mean())
            x = ca.inv - ca.groupby("rwy").inv.transform("mean")
            yv = ca.off - ca.groupby("rwy").off.transform("mean")
            b = max(float((x * yv).sum() / max((x * x).sum(), 1e-9)), 0.0)
            if it.empty or not np.isfinite(it.iloc[0]):
                return float(o.off.mean())
            xt = float(np.clip(it.iloc[0], ca.inv.min(), ca.inv.max()))   # no extrapolation beyond the 2025 range
            return float(o.off.mean() + b * (xt - o.inv.mean()))
        raise ValueError(scheme)


def shifts(rows, prc, offs, scheme, a, skip):
    """rows: apt, rwy, year, month, pred, reg_row (index 0..n-1). Returns per-row shift and the cell log."""
    out = np.zeros(len(rows))
    cells = []
    for (apt, rwy, yr, m), g in rows.groupby(["apt", "rwy", "year", "month"], observed=True):
        if (int(m), apt) in skip or len(g) < a.min_rows:
            continue
        pc = prc[(prc.apt == apt) & (prc.rwy == rwy) & (prc.year == yr) & (prc.month == m)]
        if pc.empty or pc.flt.iloc[0] < a.min_flt or not np.isfinite(pc.prc.iloc[0]):
            continue
        off = offs.get(apt, rwy, int(yr), int(m), scheme)
        if not np.isfinite(off):
            continue
        implied = pc.prc.iloc[0] + off
        body = g.pred <= a.pmax
        al = a.alpha_winter if int(m) in WINTER else a.alpha
        c = float(np.clip(al * (implied - g.pred[body].mean()), -a.cap, a.cap))
        idx = g.index[body & g.reg_row]
        out[idx] = c
        cells.append(dict(apt=apt, rwy=rwy, year=int(yr), month=int(m), n=len(g), n_shift=len(idx), c=round(c, 1),
                          implied=round(float(implied), 1), off=round(off, 1), pred_mean=round(float(g.pred[body].mean()), 1)))
    return out, cells


def frame(f, base, pimp):
    d = base.merge(f, on="MVT_ID_mvt", how="left").merge(pimp, on="MVT_ID_mvt", how="left")
    d["reg_row"] = d.p_imp.fillna(0) < 0.5
    return d.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--base", required=True, help="chain/stack id with oof/<base>.parquet and test/<base>.parquet")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--alpha_winter", type=float, default=None, help="alpha for Dec-Feb cells (default: --alpha)")
    ap.add_argument("--cv_offset", default="nb", choices=["nb", "inv"])
    ap.add_argument("--test_offset", default="same", choices=["same", "nb", "inv"])
    ap.add_argument("--skip", default="1:EDDF,EDDM,LFPG,LSZH", help="month:apt,apt;month:apt cells left alone")
    ap.add_argument("--pmax", type=float, default=7200)
    ap.add_argument("--min_flt", type=int, default=200)
    ap.add_argument("--min_rows", type=int, default=100)
    ap.add_argument("--cap", type=float, default=150, help="max |shift| per cell, s")
    ap.add_argument("--oor", type=float, default=0.0, help="inv only: > 0 leaves a cell alone when the target month's invalid share "
                                                         "lies more than this outside the fitted months' range (0.02 recommended)")
    ap.add_argument("--test_pimp", default="mix_v18_fs3_deep_raw", help="test file with p_imp (raw classifier)")
    ap.add_argument("--oof_pimp", default="rc_v4_gate2_rules")
    a = ap.parse_args()
    if a.alpha_winter is None:
        a.alpha_winter = a.alpha
    skip = parse_skip(a.skip)
    run = f"prc-cal2-{a.id}"
    report(run, 0, 3, "loading", purpose="Does a PRC offset that transfers to unseen months (invalid-share fit, winter alpha) beat prc_cal's same-month offsets?")
    prc = prc_table()
    f = pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "apt", "RUNWAY_mvt", "month", "split", "TAXITIME_SEC_mvt",
                                                                 "MVT_TIME_UTC_mvt"]).to_pandas()
    t_ = f.MVT_TIME_UTC_mvt
    f["year"] = (pd.to_datetime(t_, unit="s") if np.issubdtype(t_.dtype, np.number) else pd.to_datetime(t_)).dt.year
    f = f.rename(columns={"RUNWAY_mvt": "rwy", "TAXITIME_SEC_mvt": "y"})
    lab = f[(f.split == "train") & (f.y > 0) & (f.y <= 7200)][["apt", "rwy", "month", "y"]]
    offs = Offsets(lab, prc, invalid_share(), a.oor)
    keep = ["MVT_ID_mvt", "apt", "rwy", "year", "month"]

    oof = pd.read_parquet(OUTD / "oof" / f"{a.base}.parquet")[["MVT_ID_mvt", "fold", "y", "pred"]]
    o = frame(f[keep], oof, pd.read_parquet(OUTD / "oof" / f"{a.oof_pimp}.parquet")[["MVT_ID_mvt", "p_imp"]])
    s_oof, cells_oof = shifts(o, prc, offs, a.cv_offset, a, skip)
    y = o.y.to_numpy()
    new = np.clip(o.pred.to_numpy() + s_oof, 0, None)
    res = {"id": a.id, "model": "prc_cal2", "base": a.base, "alpha": a.alpha, "alpha_winter": a.alpha_winter, "cv_offset": a.cv_offset,
           "test_offset": a.test_offset, "skip": a.skip, "pmax": a.pmax, "cap": a.cap, "oor": a.oor,
           "cv_rmse": rmse(new, y), "cv_trim_rmse": rmse(new[y <= 20000], y[y <= 20000]), "cv_clean_rmse": rmse(new[y <= 7200], y[y <= 7200]),
           "base_cv_rmse": rmse(o.pred, y), "folds": {}, "per_airport": {}, "oof_shift_rms": float(np.sqrt(np.mean(s_oof ** 2)))}
    for fo in sorted(o.fold.unique()):
        k = (o.fold == fo).to_numpy()
        res["folds"][str(fo)] = {"rmse": rmse(new[k], y[k]), "base_rmse": rmse(o.pred.to_numpy()[k], y[k]), "n": int(k.sum())}
    for apt, g in o.groupby("apt"):
        k = g.index.to_numpy()
        res["per_airport"][apt] = {"base": rmse(o.pred.to_numpy()[k], y[k]), "new": rmse(new[k], y[k]), "shift_rms": float(np.sqrt(np.mean(s_oof[k] ** 2)))}
    pd.DataFrame({"MVT_ID_mvt": o.MVT_ID_mvt, "fold": o.fold, "y": y, "pred": new}).to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
    report(run, 1, 3, f"CV {res['cv_rmse']:.3f} (base {res['base_cv_rmse']:.3f})")

    tfile = OUTD / "test" / f"{a.base}.parquet"
    if tfile.is_file():
        t = pd.read_parquet(tfile).rename(columns={"TAXITIME_SEC_mvt": "pred"})
        t["MVT_ID_mvt"] = t.MVT_ID_mvt.astype("int64")
        tp_raw = pd.read_parquet(OUTD / "test" / f"{a.test_pimp}.parquet")[["MVT_ID_mvt", "p_imp"]]
        tp_raw["MVT_ID_mvt"] = tp_raw.MVT_ID_mvt.astype("int64")
        t = frame(f[keep].astype({"MVT_ID_mvt": "int64"}), t[["MVT_ID_mvt", "pred"]], tp_raw)
        miss = float(t.p_imp.isna().mean())
        if miss > 0.01:  # a p_imp file that misses the new months would silently move every row
            raise SystemExit(f"{a.test_pimp}: p_imp missing on {miss:.1%} of test rows")
        s_t, cells_t = shifts(t, prc, offs, a.test_offset, a, skip)
        tp = np.clip(t.pred.to_numpy() + s_t, 0, None)
        template = pd.read_parquet(TEMPLATE)
        template["TAXITIME_SEC_mvt"] = template.MVT_ID_mvt.astype("int64").map(pd.Series(tp, index=t.MVT_ID_mvt.to_numpy())).to_numpy()
        path = OUTD / "test" / f"{a.id}.parquet"
        template.to_parquet(path, index=False)
        sys.path.insert(0, str(ROOT / "scripts"))
        from baseline import validate_submission
        validate_submission(path)
        res.update(test_shift_rms=float(np.sqrt(np.mean(s_t ** 2))), test_pimp_missing=miss, test_cells=cells_t)
    res["oof_cells"] = cells_oof
    save_result(res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_airport", "oof_cells", "test_cells")}, indent=1))
    print(pd.DataFrame(res["per_airport"]).T.round(2).to_string())
    if "test_cells" in res:
        tc = pd.DataFrame(res["test_cells"])
        if len(tc):
            print(tc.groupby(["apt", "year", "month"]).apply(lambda g: pd.Series({"n": g.n.sum(), "c_mean": np.average(g.c, weights=g.n_shift + 1e-9),
                                                                                    "off": np.average(g.off, weights=g.n)}), include_groups=False).round(1).to_string())
    report(run, 3, 3, f"CV {res['cv_rmse']:.3f} (base {res['base_cv_rmse']:.3f})")
    if ocommon._progress:   # ocommon.report has no outcome; finished=True alone files the run as 'unknown'
        ocommon._progress.report("eurocontrol-" + run, "EUROCONTROL " + run, 3, 3, detail=f"CV {res['cv_rmse']:.3f} (base {res['base_cv_rmse']:.3f})", finished=True, outcome="success")


if __name__ == "__main__":
    main()
