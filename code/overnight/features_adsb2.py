# SPDX-License-Identifier: GPL-3.0-or-later
"""Same-flight ADS-B features v2: position-based movement events, learned stand coordinates, arrivals.

v1 (features_adsb.py) used ground speed only and matched ~40%. The design study (adsb_explore.py, Jan/Jul 15 2025,
6 ground-covered airports) showed that most aircraft are first heard AFTER off-block (median +225 s: transponder
off at the stand, or no receiver line of sight), and that when the aircraft is heard parked, the time it leaves its
parked spot is within 60 s of BLOCK for ~70%. So v2 adds positions:

Stage 1 (per day, parallel; cached in outputs/overnight/cache/adsb2/<day>.parquet), per movement at an airport with
ADS-B ground points that day:
  matching   icao by callsign (CALLSIGN_flt, then FLIGHT_mvt; leading zeros dropped) among points within
             MVT +- 90 min, nearest in time; else (departures) the nearest detected take-off of the same type within
             120 s of MVT. m=1 callsign, m=2 type+time.
  DEP        ground segment before take-off (first airborne >= 90 kt point after MVT - 10 min; walk back while gaps
             < 15 min, stop at an airborne point). Events (epoch s): first_seen, gs1 / gs3 (first gs >= 1 / 3 kt),
             disp15 / disp30 (first point > 15 / 30 m from the first point), last_anchor (last point within 30 m of
             the first point), roll (last gs < 30 kt), hold_end (end of the last >= 20 s stop before roll) and its
             length, stops and stopped seconds after disp15, path metres, max gap, first point lat/lon and gs.
  ARR        ground segment after landing (first ground point after MVT - 5 min, forward while gaps < 15 min);
             the parked position = median of the final points with gs < 1 kt (or the last point).
Stage 2: stand coordinates per (airport, STAND_mvt) = median of arrival parked positions and of departure anchors
seen parked (first gs < 1 kt, dwell > 60 s), over every day built (positions are label-free), >= 3 observations.
Stage 3: per departure, relative to MVT: a2_seen, a2_gs1, a2_gs3, a2_disp15, a2_disp30, a2_anchor, a2_roll, a2_hold_end
(= MVT - event), a2_hold_sec, a2_stops, a2_stop_sec, a2_path_m, a2_maxgap, a2_npts, a2_first_gs, a2_dwell, a2_match,
a2_d_stand (metres from the first point to the stand), a2_parked (first point within 60 m of the stand and gs < 1),
a2_stand_vis (share of this stand's departures heard parked, all days), a2_to (MVT - ADS-B take-off).

  python features_adsb2.py --months 2025-01,2025-07     # stage 1 for those months only, then 2-3 over the cache
  python features_adsb2.py                              # every day on disk -> data/processed/features_dep_adsb2.parquet
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import polars as pl

from features_adsb import PTS, norm_cs
from ocommon import OUTD, PROC, report

CACHE = OUTD / "cache/adsb2"
COLS = ["icao", "type", "t", "lat", "lon", "ground", "gs", "callsign", "apt", "zone"]
M_PER_DEG = 111_195.0


def _xy(lat, lon, lat0):
    return lat * M_PER_DEG, lon * M_PER_DEG * np.cos(np.radians(lat0))


def dep_events(t, lat, lon, gs, t_to):
    X, Y = _xy(lat, lon, lat[0])
    d0 = np.hypot(X - X[0], Y - Y[0])
    step = np.r_[0.0, np.hypot(np.diff(X), np.diff(Y))]
    first = lambda mask: t[np.flatnonzero(mask)[0]] if mask.any() else np.nan
    ev = dict(first_seen=t[0], lat0=lat[0], lon0=lon[0], first_gs=gs[0], npts=len(t),
              gs1=first(gs >= 1), gs3=first(gs >= 3), disp15=first(d0 > 15), disp30=first(d0 > 30),
              maxgap=float(np.diff(np.r_[t, t_to]).max()), path_m=float(step[step < 500].sum()), to=t_to)
    k = np.flatnonzero(d0 <= 30)
    ev["last_anchor"] = t[k[-1]]
    ev["dwell"] = t[k[-1]] - t[0]
    slow = np.flatnonzero(gs < 30)
    ev["roll"] = t[slow[-1]] if len(slow) else np.nan
    # stops (gs < 1 for >= 20 s) after the aircraft left its first spot
    dt = np.diff(np.r_[t, t_to])
    still = gs < 1
    start = ev["disp15"] if not np.isnan(ev["disp15"]) else t[0]
    stops, k, n = [], 0, len(t)
    while k < n:
        if still[k]:
            e = k
            while e + 1 < n and still[e + 1]:
                e += 1
            s_t, e_t = t[k], (t[e + 1] if e + 1 < n else t_to)
            if e_t - s_t >= 20 and s_t >= start:
                stops.append((s_t, e_t))
            k = e + 1
        else:
            k += 1
    roll = ev["roll"] if not np.isnan(ev["roll"]) else t_to
    pre = [s for s in stops if s[1] <= roll + 1]
    ev["hold_end"] = pre[-1][1] if pre else np.nan
    ev["hold_sec"] = pre[-1][1] - pre[-1][0] if pre else 0.0
    ev["stops"] = len(stops)
    ev["stop_sec"] = float(sum(e - s for s, e in stops))
    return ev


def day_build(day, mv, pmap=None):
    """mv: movements (DEP and ARR) with MVT on `day` at any airport. Returns per-movement stage-1 rows.
    pmap (IATA -> ICAO airline prefix) turns on the extended match: m=3 IATA flight number mapped to the ICAO
    callsign, m=4 flight number equal to the tail registration (GA rows), tried before the type+time fallback."""
    cols = COLS + (["reg"] if pmap is not None else [])
    f = PTS / f"{day}.parquet"
    d0 = pd.Timestamp(day, tz="UTC").timestamp()
    parts = [pl.read_parquet(f, columns=cols).filter(pl.col("zone") == 0)]
    prev = PTS / f"{(pd.Timestamp(day) - pd.Timedelta(days=1)).date()}.parquet"
    if prev.is_file():
        parts.append(pl.read_parquet(prev, columns=cols).filter((pl.col("zone") == 0) & (pl.col("t") > d0 - 4 * 3600)))
    nxt = PTS / f"{(pd.Timestamp(day) + pd.Timedelta(days=1)).date()}.parquet"
    if nxt.is_file():  # arrivals taxiing in past midnight
        parts.append(pl.read_parquet(nxt, columns=cols).filter((pl.col("zone") == 0) & (pl.col("t") < d0 + 86400 + 3600)))
    p = (pl.concat(parts).unique(subset=["icao", "t"]).filter(pl.col("lat").is_not_null())
         .with_columns(gs=pl.col("gs").fill_null(-1.0)).sort(["apt", "icao", "t"]).to_pandas())
    ground_apts = set(p.loc[p.ground.astype(bool), "apt"].unique())
    mv = mv[mv.apt.isin(ground_apts)]
    if not len(mv):
        return pd.DataFrame()
    p["cs"] = p.callsign.map(norm_cs)
    tracks = {}
    for (apt, icao), g in p.groupby(["apt", "icao"], sort=False):
        tracks[(apt, icao)] = (g.t.to_numpy(float), g.lat.to_numpy(float), g.lon.to_numpy(float), g.gs.to_numpy(float),
                               g.ground.to_numpy(bool), g.type.iloc[0])
    # callsign index: (apt, cs) -> [(icao, t_mid)]
    csi = {}
    for (apt, cs, icao), g in p.dropna(subset=["cs"]).groupby(["apt", "cs", "icao"], sort=False):
        csi.setdefault((apt, cs), []).append((icao, g.t.to_numpy(float)))
    regi = {}
    if pmap is not None and "reg" in p.columns:
        for (apt, rg, icao), g in p.dropna(subset=["reg"]).groupby(["apt", "reg", "icao"], sort=False):
            regi.setdefault((apt, str(rg).replace("-", "").upper()), []).append((icao, g.t.to_numpy(float)))
    # take-offs for the type+time fallback
    tos = {}
    for (apt, icao), (t, lat, lon, gs, gr, typ) in tracks.items():
        air = (~gr) & (gs >= 90)
        k = np.flatnonzero(air[1:] & ~air[:-1]) + 1
        for i in k:
            if gr[i - 1] or gs[i - 1] < 50:
                tos.setdefault(apt, []).append((t[i], icao, typ))
    rows = []
    for r in mv.itertuples(index=False):
        icao, how = None, 0
        for cs in (norm_cs(r.CALLSIGN_flt), norm_cs(r.FLIGHT_mvt)):
            cands = csi.get((r.apt, cs)) if cs else None
            if cands:
                best = min(((np.abs(tt - r.MVT).min(), ic) for ic, tt in cands), key=lambda z: z[0])
                if best[0] <= 5400:
                    icao, how = best[1], 1
                    break
        if icao is None and pmap is not None and isinstance(r.FLIGHT_mvt, str):
            fl = r.FLIGHT_mvt.strip().upper()
            g2 = re.match(r"^([A-Z0-9]{2})0*(\d{1,4}[A-Z]?)$", fl)
            for how2, idx, key in ((3, csi, norm_cs(pmap[g2.group(1)] + g2.group(2)) if g2 and g2.group(1) in pmap else None),
                                   (4, regi, fl.replace("-", ""))):
                cands = idx.get((r.apt, key)) if key else None
                if cands:
                    best = min(((np.abs(tt - r.MVT).min(), ic) for ic, tt in cands), key=lambda z: z[0])
                    if best[0] <= 5400:
                        icao, how = best[1], how2
                        break
        if icao is None and r.PHASE_mvt == "DEP":
            c = [(abs(tt - r.MVT), ic) for tt, ic, typ in tos.get(r.apt, []) if abs(tt - r.MVT) <= 120 and (typ == r.AIRCRAFT_TYPE_mvt or typ is None)]
            if c:
                icao, how = min(c)[1], 2
        if icao is None:
            continue
        t, lat, lon, gs, gr, _ = tracks[(r.apt, icao)]
        row = dict(MVT_ID_mvt=r.MVT_ID_mvt, phase=r.PHASE_mvt, apt=r.apt, stand=r.STAND_mvt, icao=icao, m=how)
        if r.PHASE_mvt == "DEP":
            air = (~gr) & (gs >= 90) & (t > r.MVT - 600) & (t < r.MVT + 900)
            if not air.any():
                continue
            i = np.flatnonzero(air)[0]
            t_to = t[i]
            j = i
            # walk back over the ground segment; points of the take-off roll (< 120 s before lift-off) never stop it
            while j > 0 and t[j] - t[j - 1] < 900 and (gr[j - 1] or gs[j - 1] < 50 or t_to - t[j - 1] < 120):
                j -= 1
            if j >= i or t[i] - t[j] > 4 * 3600:
                continue
            sl = slice(j, i)
            if len(t[sl]) < 2:
                continue
            row.update(dep_events(t[sl], lat[sl], lon[sl], gs[sl], t_to))
        else:
            k = np.flatnonzero((t > r.MVT - 300) & (gr | (gs < 50)) & (t < r.MVT + 3600))
            if not len(k):
                continue
            j = k[0]
            e = j
            while e + 1 < len(t) and (gr[e + 1] or gs[e + 1] < 50) and t[e + 1] - t[e] < 900:
                e += 1
            seg = slice(j, e + 1)
            ts, la, lo, v = t[seg], lat[seg], lon[seg], gs[seg]
            tail = np.flatnonzero((v >= 0) & (v < 1))
            tail = tail[tail >= len(ts) - 20] if len(tail) else tail
            if len(tail):
                row.update(park_lat=float(np.median(la[tail])), park_lon=float(np.median(lo[tail])), park_t=ts[-1], park_n=len(tail))
            else:
                row.update(park_lat=la[-1], park_lon=lo[-1], park_t=ts[-1], park_n=0)
        rows.append(row)
    return pd.DataFrame(rows)


def movements():
    m = (pl.read_parquet(PROC / "movements.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "CALLSIGN_flt",
                                                                "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt", "STAND_mvt"])
         .with_columns(apt=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")),
                       MVT=pl.col("MVT_TIME_UTC_mvt").cast(pl.Float64))
         .drop("ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt").to_pandas())
    m["day"] = pd.to_datetime(m.MVT, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    return m


def prefix_map():
    """IATA -> ICAO airline prefix learned from movements carrying both FLIGHT_mvt and CALLSIGN_flt (label-free)."""
    x = pl.read_parquet(PROC / "movements.parquet", columns=["FLIGHT_mvt", "CALLSIGN_flt"]).drop_nulls().to_pandas()
    fp = x.FLIGHT_mvt.str.upper().str.extract(r"^([A-Z0-9]{2})(\d{1,4}[A-Z]?)$")[0]
    cp = x.CALLSIGN_flt.str.upper().str.extract(r"^([A-Z]{3})(\d{1,4}[A-Z]{0,2})$")[0]
    cnt = pd.DataFrame({"i": fp, "c": cp}).dropna().value_counts().reset_index(name="n")
    cnt = cnt[(cnt.n / cnt.groupby("i").n.transform("sum") >= 0.8) & (cnt.n >= 20)]
    return cnt.drop_duplicates("i").set_index("i").c.to_dict()


def _work(day, mv, cache=None, pmap=None):
    cache = cache or CACHE
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{day}.parquet"
    if out.is_file():
        return day, -1
    f = day_build(day, mv, pmap)
    f.to_parquet(out, index=False)
    return day, len(f)


def assemble(name="adsb2", cache=None):
    s1 = pd.concat([pd.read_parquet(p) for p in sorted((cache or CACHE).glob("*.parquet"))], ignore_index=True)
    dep = s1[s1.phase == "DEP"].copy()
    arr = s1[s1.phase == "ARR"].copy()
    # stage 2: stand coordinates from arrival parked spots and departure anchors seen parked
    obs = [arr[["apt", "stand", "park_lat", "park_lon"]].rename(columns={"park_lat": "lat", "park_lon": "lon"})]
    seen = dep[(dep.first_gs >= 0) & (dep.first_gs < 1) & (dep.dwell > 60)]
    obs.append(seen[["apt", "stand", "lat0", "lon0"]].rename(columns={"lat0": "lat", "lon0": "lon"}))
    obs = pd.concat(obs).dropna()
    st = obs.groupby(["apt", "stand"]).agg(s_lat=("lat", "median"), s_lon=("lon", "median"), s_n=("lat", "size")).reset_index()
    st = st[st.s_n >= 3]
    dep = dep.merge(st, on=["apt", "stand"], how="left")
    X0, Y0 = _xy(dep.lat0.to_numpy(), dep.lon0.to_numpy(), dep.lat0.to_numpy())
    XS, YS = _xy(dep.s_lat.to_numpy(), dep.s_lon.to_numpy(), dep.lat0.to_numpy())
    dep["d_stand"] = np.hypot(X0 - XS, Y0 - YS)
    dep["parked"] = ((dep.d_stand <= 60) & (dep.first_gs >= 0) & (dep.first_gs < 1)).astype(float)
    dep.loc[dep.d_stand.isna(), "parked"] = np.nan
    dep["stand_vis"] = dep.groupby(["apt", "stand"]).parked.transform("mean")
    mv = movements()
    dep = dep.merge(mv[["MVT_ID_mvt", "MVT"]], on="MVT_ID_mvt")
    out = pd.DataFrame({"MVT_ID_mvt": dep.MVT_ID_mvt})
    for c in ["first_seen", "gs1", "gs3", "disp15", "disp30", "last_anchor", "roll", "hold_end", "to"]:
        out["a2_" + {"first_seen": "seen", "last_anchor": "anchor"}.get(c, c)] = dep.MVT - dep[c]
    for c in ["hold_sec", "stops", "stop_sec", "path_m", "maxgap", "npts", "first_gs", "dwell", "d_stand", "parked", "stand_vis"]:
        out["a2_" + c] = dep[c].astype(float)
    out["a2_match"] = dep.m.astype(float)
    if os.environ.get("ADSB_CAP"):  # density cap (overnight/density_cap.py): 2026 rows only
        import density_cap
        n2, g2, nch = density_cap.clip(out.a2_npts, out.a2_maxgap, dep.apt, dep.MVT, density_cap.load(os.environ["ADSB_CAP"]))
        out["a2_npts"], out["a2_maxgap"] = n2, g2
        print(f"density cap {os.environ['ADSB_CAP']}: {nch:,} of {len(out):,} departures changed", flush=True)
    alld = mv.loc[mv.PHASE_mvt == "DEP", ["MVT_ID_mvt"]].merge(out, on="MVT_ID_mvt", how="left")
    alld.to_parquet(PROC / f"features_dep_{name}.parquet", index=False)
    st.to_parquet(PROC / ("adsb_stand_coords.parquet" if name == "adsb2" else f"adsb_stand_coords_{name}.parquet"), index=False)
    return alld, st, len(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="adsb2", help="output features_dep_<out>.parquet (assembled from every cached day)")
    ap.add_argument("--ext", type=int, default=0, help="1 = extended match (IATA->ICAO prefix map, registration); cache/adsb2m")
    ap.add_argument("--cache", default="", help="stage-1 cache dir name under cache/ (default adsb2 / adsb2m)")
    ap.add_argument("--assemble_only", type=int, default=0, help="1 = stage 3 only from the cached days (no stage 1, no prefix map write); used with ADSB_CAP")
    a = ap.parse_args()
    cache, pmap = CACHE, None
    if a.assemble_only:
        cache = OUTD / "cache" / (a.cache or "adsb2")
        alld, st, n_arr = assemble(a.out, cache)
        print(f"assembled {a.out}: {alld.a2_seen.notna().sum():,} departures with a track, {len(st):,} stands", flush=True)
        return
    if a.ext:
        cache, pmap = OUTD / "cache/adsb2m", prefix_map()
        (PROC / "iata_icao_prefix.json").write_text(json.dumps(pmap, indent=0, sort_keys=True))
    if a.cache:
        cache = OUTD / "cache" / a.cache
    run = "features-adsb2"
    mv = movements()
    days = sorted({p.stem for p in PTS.glob("*.parquet")} | {p.stem for p in cache.glob("*.parquet")})  # cached days need no raw points
    if a.months:
        days = [d for d in days if d[:7] in a.months.split(",")]
    purpose = "ADS-B v2 same-flight features (positions, stand coordinates, stops) for the ADS-B CatBoost member"
    report(run, 0, len(days) + 1, f"{len(days)} days", purpose=purpose)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_work, d, mv[mv.day == d], cache, pmap) for d in days]
        for k, fu in enumerate(as_completed(futs), 1):
            day, n = fu.result()
            report(run, k, len(days) + 1, f"{day}: {n} movements matched ({(time.time() - t0) / 60:.1f} min)")
    alld, st, n_arr = assemble(a.out, cache)
    got = alld.a2_seen.notna()
    msg = f"done: {got.sum():,} departures with a track ({got.mean():.1%} of all), {n_arr:,} arrivals, {len(st):,} stands located"
    report(run, len(days) + 1, len(days) + 1, msg, finished=True)


if __name__ == "__main__":
    main()
