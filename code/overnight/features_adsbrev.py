# SPDX-License-Identifier: GPL-3.0-or-later
"""Pushback from reverse motion (literature digest #2, the traffic library's idea) on the matched ADS-B ground segment.

On the ground, readsb/adsb.lol `track` is the true heading. During pushback the aircraft moves tail-first, so the
bearing between consecutive fixes is ~180 deg off the heading. Per departure matched in the adsb2 stage-1 cache
(icao, first_seen, take-off), over the ground segment first_seen .. take-off:
  reverse step = dt < 60 s, 2 m <= step <= 150 m, |bearing - heading| > 135 deg (both fixes on the ground)
Features (MVT - event, seconds, like a2_*): rv_first, rv_last (end of push), rv_fwd (first forward fix with
gs >= 3 kt after the push, else after first_seen), rv_n, rv_m (reverse steps and metres), rv_frac_hd (share of
ground fixes carrying a heading), rv_seen_push (1 = first fix already inside the push).

  python features_adsbrev.py            # every cached day -> data/processed/features_dep_adsbrev.parquet
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import polars as pl

from features_adsb import PTS
from ocommon import OUTD, PROC, report, write_merged

CACHE = OUTD / "overnight" / "cache" / "adsb2"
if not CACHE.is_dir():
    CACHE = OUTD / "cache" / "adsb2"
COLS = ["icao", "t", "lat", "lon", "ground", "gs", "track", "zone"]


def rev_events(t, lat, lon, gs, gr, hd, mvt):
    out = {}
    hasd = gr & ~np.isnan(hd)
    out["rv_frac_hd"] = float(hasd.mean()) if len(t) else np.nan
    if len(t) < 2:
        return out
    lat0 = np.radians(lat[0])
    x = np.radians(lon) * 6371000 * np.cos(lat0)
    y = np.radians(lat) * 6371000
    dx, dy, dt = np.diff(x), np.diff(y), np.diff(t)
    step = np.hypot(dx, dy)
    brg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    h = hd[1:]
    diff = np.abs((brg - h + 180) % 360 - 180)
    ok = (dt < 60) & (step >= 2) & (step <= 150) & gr[1:] & gr[:-1] & ~np.isnan(h)
    rev = ok & (diff > 135)
    k = np.flatnonzero(rev)
    if len(k):
        out["rv_first"] = mvt - t[k[0]]      # the step ends at fix k+1; its start fix k is the first moving fix
        out["rv_last"] = mvt - t[k[-1] + 1]
        out["rv_n"] = int(len(k))
        out["rv_m"] = float(step[k].sum())
        out["rv_seen_push"] = int(k[0] == 0)
        after = np.flatnonzero((np.arange(len(t)) > k[-1] + 1) & (gs >= 3))
    else:
        out["rv_n"], out["rv_m"] = 0, 0.0
        after = np.flatnonzero(gs >= 3)
    if len(after):
        out["rv_fwd"] = mvt - t[after[0]]
    return out


_MVT = {}


def _init(m):
    _MVT.update(m)


def day_rev(day, mvt_map=None, cache=None):
    mvt_map = mvt_map if mvt_map is not None else _MVT
    c = pd.read_parquet((cache or CACHE) / f"{day}.parquet")
    if not len(c) or "phase" not in c.columns:   # days with no ground coverage cache as empty frames
        return pd.DataFrame()
    c = c[(c.phase == "DEP") & c.first_seen.notna() & c.to.notna()]
    if not len(c):
        return pd.DataFrame()
    d0 = pd.Timestamp(day, tz="UTC").timestamp()
    parts = [pl.read_parquet(PTS / f"{day}.parquet", columns=COLS).filter(pl.col("zone") == 0)]
    prev = PTS / f"{(pd.Timestamp(day) - pd.Timedelta(days=1)).date()}.parquet"
    if prev.is_file():
        parts.append(pl.read_parquet(prev, columns=COLS).filter((pl.col("zone") == 0) & (pl.col("t") > d0 - 4 * 3600)))
    icaos = set(c.icao)
    p = (pl.concat(parts).filter(pl.col("icao").is_in(list(icaos)) & pl.col("lat").is_not_null())
         .unique(subset=["icao", "t"]).with_columns(gs=pl.col("gs").fill_null(-1.0)).sort(["icao", "t"]).to_pandas())
    tr = {k: (g.t.to_numpy(float), g.lat.to_numpy(float), g.lon.to_numpy(float), g.gs.to_numpy(float),
              g.ground.to_numpy(bool), g.track.to_numpy(float)) for k, g in p.groupby("icao", sort=False)}
    rows = []
    for r in c.itertuples(index=False):
        if r.icao not in tr or r.MVT_ID_mvt not in mvt_map:
            continue
        t, la, lo, gs, gr, hd = tr[r.icao]
        m = (t >= r.first_seen) & (t < r.to)
        row = dict(MVT_ID_mvt=r.MVT_ID_mvt)
        row.update(rev_events(t[m], la[m], lo[m], gs[m], gr[m], hd[m], mvt_map[r.MVT_ID_mvt]))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cache", default="adsb2", help="stage-1 cache dir under outputs/overnight/cache (adsb2m = extended match)")
    ap.add_argument("--out", default="adsbrev")
    ap.add_argument("--months", default="", help="comma list YYYY-MM (default: every cached day)")
    a = ap.parse_args()
    cache = CACHE.parent / a.cache
    run = f"{a.out}-build"
    days = sorted(f.stem for f in cache.glob("*.parquet"))
    if a.months:  # only these months (final phase: raw points exist for the new months only; EC_MERGE=1 keeps the seeded 2025 rows)
        days = [d for d in days if d[:7] in a.months.split(",") and (PTS / f"{d}.parquet").exists()]
    if not any((PTS / f"{d}.parquet").exists() for d in days):  # seeded phase without raw points: the seeded rows stand
        print("no raw-point days; nothing to do")
        return
    report(run, 0, len(days) + 1, "loading", purpose="Does pushback detected from reverse motion (heading vs bearing) sharpen the ADS-B off-block anchor?")
    mv = pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).to_pandas()
    mvt_map = dict(zip(mv.MVT_ID_mvt, mv.MVT_TIME_UTC_mvt.astype(float)))   # epoch seconds
    out, t0 = [], time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(mvt_map,)) as ex:
        futs = {ex.submit(day_rev, d, None, cache): d for d in days}
        for i, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if i % 10 == 0 or i == len(days):
                report(run, i, len(days) + 1, f"{i}/{len(days)} days, {time.time() - t0:.0f} s")
    d = pd.concat([o for o in out if len(o)], ignore_index=True).drop_duplicates("MVT_ID_mvt")
    d = write_merged(PROC / f"features_dep_{a.out}.parquet", d)
    # accuracy on labelled rows: y - (rv_first) per airport, against the parked anchor
    lab = pd.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "apt", "TAXITIME_SEC_mvt"])
    a2 = pd.read_parquet(PROC / ("features_dep_adsb2m.parquet" if a.cache == "adsb2m" else "features_dep_adsb2.parquet"), columns=["MVT_ID_mvt", "a2_anchor", "a2_parked"])
    e = d.merge(lab, on="MVT_ID_mvt").merge(a2, on="MVT_ID_mvt", how="left")
    e = e[e.TAXITIME_SEC_mvt.notna() & (e.TAXITIME_SEC_mvt <= 7200)]
    print(f"rows {len(d):,}; with reverse {int((d.rv_n > 0).sum()):,}")
    for apt, g in e.groupby("apt"):
        hr = g[g.rv_n > 0]
        err = hr.TAXITIME_SEC_mvt - hr.rv_first
        np_ = hr[hr.a2_parked != 1]
        e2 = np_.TAXITIME_SEC_mvt - np_.rv_first
        print(f"{apt}: matched {len(g):6d}  reverse {len(hr) / max(len(g), 1):.2f}  |err|<60 {np.mean(np.abs(err) < 60):.2f} "
              f"median {np.median(err) if len(err) else np.nan:+.0f}  | not-parked rows with reverse {len(np_):5d} |err|<60 "
              f"{np.mean(np.abs(e2) < 60) if len(e2) else np.nan:.2f}")
    report(run, len(days) + 1, len(days) + 1, f"wrote {len(d):,} rows, reverse on {int((d.rv_n > 0).sum()):,}", finished=True)


if __name__ == "__main__":
    main()
