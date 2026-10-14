# SPDX-License-Identifier: GPL-3.0-or-later
"""Same-flight ADS-B features for departures, from the adsb.lol points (scripts/adsb_fetch.py, zone 0 = within
7 km / 3000 ft of an airport).

Per day and airport, every aircraft's take-offs are detected (a point airborne at >= 90 kt whose previous point, within
5 min, was on the ground or slow). Departures are matched to a take-off by callsign (FLIGHT_mvt, then CALLSIGN_flt;
leading zeros of the number ignored) within [MVT - 20 min, MVT + 20 min], else by the nearest take-off of the same
aircraft type within 120 s of MVT. From the matched aircraft's ground points before its take-off (walking back while
gaps are < 15 min and no earlier airborne point intervenes):
  first_seen   first ground point of that segment
  first_move   end of the last stationary spell (gs < 1 kt) of >= 3 min in the segment (often the post-push
               engine-start pause or a runway hold); else the first point with gs >= 3 kt (flag stand_seen = 0)
  push         end of the FIRST such spell = leaving the stand; NaN when the aircraft was not seen still at the stand
  roll_start   last point with gs < 30 kt before take-off (line-up / start of roll)
Features are relative to MVT (label-free: MVT and the track are inputs): ad_to, ad_seen, ad_move, ad_push, ad_roll (MVT - t), ad_spells,
ad_stop_sec (stationary time after first_move), ad_dist_km, ad_npts, ad_maxgap, ad_stand_seen, ad_match (1 callsign,
2 time+type), ad_zone0 (airport had any zone-0 points that day).

  python features_adsb.py --months 2025-01              # one month
  python features_adsb.py                               # every day on disk -> data/processed/features_dep_adsb.parquet
"""
import argparse
import os
import re
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import polars as pl

from ocommon import OUTD, PROC, REP, ROOT, report

PTS = Path(os.environ["ADSB_PTS"]) if os.environ.get("ADSB_PTS") else ROOT / "data/external/adsblol/points"  # ADSB_PTS: merged feeds
CACHE = OUTD / "cache/adsb"
COLS = ["icao", "type", "t", "ground", "gs", "alt", "callsign", "apt", "zone"]
AIR_GS, SLOW_GS, ROLL_GS, MOVE_GS, STAND_MIN = 90.0, 50.0, 30.0, 3.0, 180.0


def norm_cs(s):
    """THY056 -> THY56; None stays None."""
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip().upper()
    m = re.match(r"^([A-Z]{3})0*(\w*)$", s)
    return (m.group(1) + m.group(2)) if m else s


def takeoffs(p):
    """One row per detected take-off with its ground-segment timings. p: zone-0 points of one day (pandas)."""
    out = []
    for (icao, apt), g in p.groupby(["icao", "apt"], sort=False):
        g = g.sort_values("t")
        t, gs, gr = g.t.to_numpy(), g.gs.to_numpy(float), g.ground.to_numpy(bool)
        gs = np.nan_to_num(gs, nan=0.0)
        air = (~gr) & (gs >= AIR_GS)
        onground = gr | (gs < SLOW_GS)
        cs = g.callsign.to_numpy()
        for i in np.flatnonzero(air[1:] & onground[:-1] & (np.diff(t) < 300)) + 1:
            # walk back over the ground segment
            j = i - 1
            while j > 0 and onground[j - 1] and t[j] - t[j - 1] < 900:
                j -= 1
            seg = slice(j, i)
            ts, v = t[seg], gs[seg]
            dt = np.diff(ts, append=t[i])
            still = v < 1.0
            first_move, stand_seen, push, n_spells = np.nan, 0, np.nan, 0
            # stationary spells: runs of still points; the last one >= STAND_MIN ends at first_move
            k, n = 0, len(ts)
            while k < n:
                if still[k]:
                    e = k
                    while e + 1 < n and still[e + 1]:
                        e += 1
                    end_t = ts[e + 1] if e + 1 < n else t[i]
                    if end_t - ts[k] >= STAND_MIN:
                        first_move, stand_seen = end_t, 1
                        n_spells += 1
                        if np.isnan(push):  # end of the FIRST long spell = leaving the stand (pushback)
                            push = end_t
                    k = e + 1
                else:
                    k += 1
            if np.isnan(first_move):
                mv = np.flatnonzero(v >= MOVE_GS)
                first_move = ts[mv[0]] if len(mv) else ts[0]
            after = ts >= first_move
            slow = np.flatnonzero(v < ROLL_GS)
            c = pd.Series(cs[max(0, j - 5):i + 5]).dropna()
            out.append(dict(icao=icao, apt=apt, type=g.type.iloc[0], cs=norm_cs(c.mode().iloc[0]) if len(c) else None,
                            to=t[i], first_seen=ts[0], first_move=first_move, stand_seen=stand_seen, push=push, n_spells=n_spells,
                            roll_start=ts[slow[-1]] if len(slow) else np.nan,
                            stop_sec=float(dt[after & still].sum()), dist_km=float((v[after] * dt[after]).sum() * 1.852 / 3600),
                            npts=int(n), maxgap=float(dt.max()) if n else np.nan))
    return pd.DataFrame(out)


def day_features(day, deps):
    """deps: departures with MVT on `day` (pandas). Returns per-departure features."""
    f = PTS / f"{day}.parquet"
    prev = PTS / f"{(pd.Timestamp(day) - pd.Timedelta(days=1)).date()}.parquet"
    d0 = pd.Timestamp(day, tz="UTC").timestamp()
    parts = [pl.read_parquet(f, columns=COLS).filter(pl.col("zone") == 0)]
    if prev.is_file():  # taxis that started before midnight
        parts.append(pl.read_parquet(prev, columns=COLS).filter((pl.col("zone") == 0) & (pl.col("t") > d0 - 4 * 3600)))
    p = pl.concat(parts).unique(subset=["icao", "t"]).to_pandas()
    to = takeoffs(p)
    to = to[(to.to >= d0 - 1800) & (to.to < d0 + 86400 + 1800)].reset_index(drop=True)
    seen_apts = set(p.apt.unique())
    res = []
    used = set()
    by_cs = {k: g for k, g in to.groupby(["apt", "cs"])} if len(to) else {}
    by_apt = {k: g for k, g in to.groupby("apt")} if len(to) else {}
    for r in deps.itertuples(index=False):
        row = dict(MVT_ID_mvt=r.MVT_ID_mvt, ad_zone0=int(r.apt in seen_apts), ad_match=0)
        best = None
        for cs in (norm_cs(r.FLIGHT_mvt), norm_cs(r.CALLSIGN_flt)):
            g = by_cs.get((r.apt, cs)) if cs else None
            if g is not None:
                g = g[(g.to - r.MVT).abs() <= 1200]
                if len(g):
                    best, row["ad_match"] = g.loc[(g.to - r.MVT).abs().idxmin()], 1
                    break
        if best is None and r.apt in by_apt:
            g = by_apt[r.apt]
            g = g[((g.to - r.MVT).abs() <= 120) & ((g.type == r.AIRCRAFT_TYPE_mvt) | g.type.isna())]
            if len(g):
                best, row["ad_match"] = g.loc[(g.to - r.MVT).abs().idxmin()], 2
        if best is not None:
            used.add(best.name)
            row.update(ad_to=r.MVT - best.to, ad_seen=r.MVT - best.first_seen, ad_move=r.MVT - best.first_move,
                       ad_roll=r.MVT - best.roll_start, ad_stop_sec=best.stop_sec, ad_dist_km=best.dist_km,
                       ad_npts=best.npts, ad_maxgap=best.maxgap, ad_stand_seen=best.stand_seen,
                       ad_push=r.MVT - best.push, ad_spells=best.n_spells)
        res.append(row)
    return pd.DataFrame(res), len(to)


def deps_frame():
    m = (pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt",
                                                                "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt", "split"])
         .filter(pl.col("PHASE_mvt") == "DEP")
         .select("MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt", "split", apt=pl.col("ADEP_mvt"),
                 MVT=pl.col("MVT_TIME_UTC_mvt").cast(pl.Float64)).to_pandas())
    m["day"] = pd.to_datetime(m.MVT, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    return m


def _work(day, deps):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{day}.parquet"
    if out.is_file():
        return day, pd.read_parquet(out), -1
    f, n_to = day_features(day, deps)
    f.to_parquet(out, index=False)
    return day, f, n_to



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="", help="comma list YYYY-MM (default: every day on disk)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    run = "features-adsb" + (f"-{a.months.replace(',', '_')}" if a.months else "")
    deps = deps_frame()
    days = sorted({p.stem for p in PTS.glob("*.parquet")} | {p.stem for p in CACHE.glob("*.parquet")})  # cached days need no raw points
    if a.months:
        days = [d for d in days if d[:7] in a.months.split(",")]
    have = set(deps.day)
    days = [d for d in days if d in have]  # seeded 2025 caches have no departures in a final/pseudo phase
    purpose = "Same-flight ADS-B timings (take-off, first move, line-up) for departures: how close do they get to BLOCK?"
    report(run, 0, len(days), f"{len(days)} days", purpose=purpose)
    parts, t0 = [], time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_work, d, deps[deps.day == d]) for d in days]
        for k, fu in enumerate(as_completed(futs), 1):
            day, f, n_to = fu.result()
            parts.append(f)
            report(run, k, len(days), f"{day}: {len(f):,} deps, matched {(f.ad_match > 0).mean():.1%}, {n_to} take-offs ({(time.time() - t0) / 60:.1f} min)")
    feat = pd.concat(parts, ignore_index=True)
    allf = deps[["MVT_ID_mvt"]].merge(feat, on="MVT_ID_mvt", how="left")
    allf.to_parquet(PROC / "features_dep_adsb.parquet", index=False)
    report(run, len(days), len(days), f"done: {len(feat):,} deps, matched {(feat.ad_match > 0).mean():.1%}", finished=True)


if __name__ == "__main__":
    main()
