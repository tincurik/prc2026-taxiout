# SPDX-License-Identifier: GPL-3.0-or-later
"""Post-step 'witness push' for the final (docs/model.md, the NM plan-attachment window). Inputs only: the movements files and the
predictions; the margins are fitted on 2025 labels (leave-one-month-out for validation, all of 2025 for the final). No label of a test month is read.

A departure row has a flight plan only if |BLOCK - LOBT| <= 3606 s. Three rules use that window (numbered ideas):
  1  witness push with the zero floor. Unmatched DEP rows whose plan is visible through the same flight's ARR row (the "witness",
     w_lobt): the label lies outside [dl-3600, dl+3600], dl = MVT - w_lobt. If the window's lower edge is <= 60 s (zero floor) the label is above
     it: LIRF rows are lifted to at least hi + m (m = median of y - hi on the other months' LIRF zero-floor rows), other airports are left alone.
     Otherwise a prediction inside the window moves to the nearer edge.
  2  LIRF UTC-offset hour-shift. Unmatched, regular (>= 80% of the flight's rows within +-15 days matched), 0 <= MVT-SCHED < 3600, not BEL/AEZ (the same two carriers as the LIRF near-schedule rule, docs/model.md):
     pred += 0.75 * (1 - c_al) * (off + 900 - pred), c_al the airline's copy share in the other months (|y-(MVT-SCHED)| <= 30), shrunk to the pooled share, K=5.
  3  plan-window clip. Matched DEP rows: pred clipped to [max(0, MVT-LOBT-W), MVT-LOBT+W], W = 3606 s, the edge of the plan join measured on 2025, on every file.
Order of application: 2, 1, 3 (3 acts on matched rows, the others on unmatched ones, so only 1 and 2 can touch the same row; 1 then wins).

  python scripts/witness_post.py --source outputs/overnight_final/test/ar_134naclh_tww.parquet --ideas zf3
      -> <source>_wit.parquet (same columns, same row order) + .wit.json (rows moved per idea, margins used)
Ideas by part: z (idea 1 zero-floor LIRF), f (idea 1 far side); "zf3" = ideas 1 and 3 without the hour shift (output _wit_zf3). Default: no change unless --ideas is given. Library use (scripts/witness_eval.py): context(), fit_params(), apply().
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt"]
TZ = {"EGLL": "Europe/London", "EHAM": "Europe/Amsterdam", "LFPG": "Europe/Paris", "EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin",
      "LEMD": "Europe/Madrid", "LEBL": "Europe/Madrid", "LIRF": "Europe/Rome", "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}
EP = pd.Timestamp("1970-01-01", tz="UTC")
WINDOW = 3600          # the plan join window is 3606 on BLOCK (idea 3 and --win use 3606); the witness edges of idea 1 use 3600.  The two constants were fixed when each idea was built and are kept
                       # as they were, since the shipped file was made with them; whether unifying them changes the file was not measured
K = 5                  # shrinkage of the airline copy share toward the pooled share (idea 2)
W0 = 0.75              # idea 2 weight toward off + 900


def sec(s):
    return (pd.to_datetime(s, utc=True) - EP).dt.total_seconds()


def read_movements(paths, labels=False):
    """All DEP and ARR rows of the files. labels=True keeps y (TAXITIME_SEC of DEP rows, training files only)."""
    parts = []
    for p in paths:
        t = pd.read_parquet(p, columns=COLS + (["TAXITIME_SEC_mvt"] if labels else []))
        parts.append(t)
    A = pd.concat(parts, ignore_index=True)
    A["MVT_ID_mvt"] = A.MVT_ID_mvt.astype("int64")
    return A


def context(A):
    """One row per DEP row (index MVT_ID_mvt): apt, al, matched, mvt, sched, ds, lobt (matched), w_lobt (witness), reg, off, mo, y if present."""
    dep = A[A.PHASE_mvt == "DEP"].copy()
    dep["matched"] = dep.FLIGHT_ID_mvt.notna()
    dep["mvt"] = sec(dep.MVT_TIME_UTC_mvt)
    dep["sched"] = sec(dep.SCHED_TIME_UTC_mvt)
    dep["lobt"] = np.where(dep.matched, sec(dep.LOBT_flt), np.nan)
    # witness table: the unmatched DEP row's flight arrives within 6 h at ADES carrying a LOBT
    arr = A[(A.PHASE_mvt == "ARR") & A.LOBT_flt.notna()][["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"]].copy()
    arr["ta"] = sec(arr.MVT_TIME_UTC_mvt)
    arr["w_lobt"] = sec(arr.LOBT_flt)
    u = dep[~dep.matched][["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "mvt"]].rename(columns={"mvt": "td"})
    m = u.merge(arr[["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "ta", "w_lobt"]], on=["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"])
    m = m[(m.ta > m.td) & (m.ta - m.td <= 6 * 3600)].sort_values("ta").drop_duplicates("MVT_ID_mvt")
    dep = dep.merge(m[["MVT_ID_mvt", "w_lobt"]], on="MVT_ID_mvt", how="left")
    # regularity flag: within the file, +-15 days around the row's day, the row's own day excluded
    dep["day"] = dep.SCHED_TIME_UTC_mvt.dt.tz_convert(None).dt.floor("D")
    dep["key"] = dep.ADEP_mvt.astype(str) + "|" + dep.FLIGHT_mvt.astype(str)
    g = dep.groupby(["key", "day"]).agg(n=("matched", "size"), k=("matched", "sum")).reset_index().sort_values(["key", "day"])
    out = []
    for key, gg in g.groupby("key", sort=False):
        days = gg.day.values.astype("datetime64[D]").astype(np.int64)
        cn = np.r_[0, np.cumsum(gg.n.values)]
        ck = np.r_[0, np.cumsum(gg.k.values)]
        lo = np.searchsorted(days, days - 15, "left")
        hi = np.searchsorted(days, days + 15, "right")
        out.append(pd.DataFrame({"key": key, "day": gg.day.values, "wn": cn[hi] - cn[lo] - gg.n.values, "wk": ck[hi] - ck[lo] - gg.k.values}))
    dep = dep.merge(pd.concat(out), on=["key", "day"], how="left")
    rate = dep.wk / dep.wn.replace(0, np.nan)
    dep["reg"] = np.select([(dep.wn >= 5) & (rate >= .8), (dep.wn >= 5) & (rate < .2)], ["regular", "never"], "other")
    # UTC offset of the airport's local time at the take-off (DST aware)
    dep["off"] = np.nan
    for apt, tz in TZ.items():
        i = dep.ADEP_mvt == apt
        if i.any():
            t = dep.loc[i, "MVT_TIME_UTC_mvt"]
            dep.loc[i, "off"] = (t.dt.tz_convert(tz).dt.tz_localize(None) - t.dt.tz_convert(None)).dt.total_seconds()
    dep["apt"] = dep.ADEP_mvt
    dep["al"] = dep.FLIGHT_mvt.astype(str).str[:3]
    dep["ds"] = dep.mvt - dep.sched
    dep["mo"] = dep.MVT_TIME_UTC_mvt.dt.tz_convert(None).dt.strftime("%Y-%m")
    if "TAXITIME_SEC_mvt" in dep:
        dep["y"] = dep.TAXITIME_SEC_mvt.astype(float)
    keep = ["MVT_ID_mvt", "apt", "al", "matched", "mvt", "sched", "ds", "lobt", "w_lobt", "reg", "off", "mo"] + (["y"] if "y" in dep else [])
    return dep[keep].set_index("MVT_ID_mvt")


def fit_params(ctx, exclude_month=None):
    """Margins from labelled 2025 rows (ctx has y), leaving out one month if given: m (idea 1) and the airline copy shares (idea 2)."""
    c = ctx if exclude_month is None else ctx[ctx.mo != exclude_month]
    w = c[c.w_lobt.notna() & (c.apt == "LIRF")]
    lo0 = (w.mvt - w.w_lobt) - WINDOW
    zf = w[lo0 <= 60]
    m = float((zf.y - ((zf.mvt - zf.w_lobt) + WINDOW)).median()) if len(zf) else 0.0
    L = c[(c.apt == "LIRF") & ~c.matched & (c.ds >= 0) & (c.ds < 3600) & ~c.al.isin(["BEL", "AEZ"])]
    cp = (L.y - L.ds).abs() <= 30
    pooled = float(cp.mean())
    s = cp.groupby(L.al).agg(["sum", "size"])
    share = ((s["sum"] + K * pooled) / (s["size"] + K)).to_dict()
    return {"m": m, "n_zero_floor": int(len(zf)), "pooled": pooled, "c_al": share}


def apply(pred, ctx, par, ideas="123", win=3606):
    """pred: Series indexed by MVT_ID_mvt (DEP rows). Returns (new Series, {idea: rows moved}). Rows missing from ctx are left alone."""
    p = pred.astype(float).copy()
    c = ctx.reindex(p.index)
    moved = {}
    if "2" in ideas:
        i = ((c.apt == "LIRF") & (c.matched == False) & (c.reg == "regular") & (c.ds >= 0) & (c.ds < 3600) & ~c.al.isin(["BEL", "AEZ"])).fillna(False).to_numpy()  # noqa: E712
        ca = c.al.map(par["c_al"]).fillna(par["pooled"]).to_numpy(float)
        new = p.to_numpy() + W0 * (1 - ca) * (c.off.to_numpy(float) + 900 - p.to_numpy())
        moved["2"] = int((i & (np.abs(new - p.to_numpy()) > 1e-9)).sum())
        p = pd.Series(np.where(i, new, p.to_numpy()), index=p.index)
    if any(ch in ideas for ch in "1zf"):   # "1" = both parts of idea 1; "z" zero-floor LIRF part only, "f" far-side part only
        v = p.to_numpy().copy()
        do_z, do_f = "1" in ideas or "z" in ideas, any(ch in ideas for ch in "1f")
        has = c.w_lobt.notna().to_numpy()
        dl = (c.mvt - c.w_lobt).to_numpy(float)
        lo, hi = dl - WINDOW, dl + WINDOW
        lirf = (c.apt == "LIRF").to_numpy()
        zero = has & (lo <= 60) & lirf
        far = has & (lo > 60)
        flo, fhi = lo, hi
        inside = far & (v > flo) & (v < fhi)
        inside = inside & do_f
        zero = zero & do_z
        v[zero] = np.maximum(v[zero], hi[zero] + par["m"])
        up = np.where(v - flo <= fhi - v, flo, fhi)
        v[inside] = up[inside]
        moved["1_zero_floor"] = int((zero & (v != p.to_numpy())).sum())
        moved["1_far_side"] = int((inside & (v != p.to_numpy())).sum())
        p = pd.Series(v, index=p.index)
    if "3" in ideas:
        i = (c.matched == True) & c.lobt.notna()  # noqa: E712
        dm = (c.mvt - c.lobt).to_numpy(float)
        v = p.to_numpy().copy()
        new = np.clip(v, np.maximum(0, dm - win), dm + win)
        sel = i.to_numpy()
        moved["3"] = int((sel & (np.abs(new - v) > 1e-9)).sum())
        p = pd.Series(np.where(sel, new, v), index=p.index)
    return p, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="prediction file for final_ranking's DEP rows (MVT_ID_mvt, TAXITIME_SEC_mvt)")
    ap.add_argument("--ideas", default="", help="subset of 123 (default none: the file is copied unchanged)")
    ap.add_argument("--ranking", default=str(RAW / "final_ranking.parquet"))
    ap.add_argument("--win", type=float, default=3606, help="idea 3 half-window (s): the 2025 plan-join edge")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    src = Path(a.source) if Path(a.source).is_absolute() else ROOT / a.source
    out = Path(a.out) if a.out else src.with_name(src.stem + ("_wit.parquet" if a.ideas == "123" else f"_wit_{a.ideas}.parquet"))
    pred = pd.read_parquet(src)
    ctx = context(read_movements([a.ranking]))
    tr = context(read_movements(sorted(glob.glob(str(RAW / "training_*.parquet"))), labels=True))
    par = fit_params(tr)
    new, moved = apply(pred.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt, ctx, par, a.ideas, a.win)
    res = pred.copy()
    res["TAXITIME_SEC_mvt"] = new.reindex(pred.MVT_ID_mvt).to_numpy().astype(pred.TAXITIME_SEC_mvt.dtype)
    res.to_parquet(out, index=False)
    d = np.abs(res.TAXITIME_SEC_mvt.to_numpy(float) - pred.TAXITIME_SEC_mvt.to_numpy(float))
    stats = dict(ideas=a.ideas, rows=len(pred), moved=moved, rows_changed=int((d > 0).sum()), mean_abs_move_changed=float(d[d > 0].mean()) if (d > 0).any() else 0.0,
                 m=par["m"], n_zero_floor_2025=par["n_zero_floor"], pooled_copy_share=par["pooled"], win=a.win,
                 witness_rows=int(ctx.w_lobt.notna().sum()), source=str(a.source))
    out.with_suffix(".wit.json").write_text(json.dumps(stats, indent=1))
    print(out, json.dumps(stats))


if __name__ == "__main__":
    main()
