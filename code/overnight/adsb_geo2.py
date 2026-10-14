# SPDX-License-Identifier: GPL-3.0-or-later
"""Taxi-map v2: how each matched departure was MOVING when first heard (adds to adsb_geo.py's position).
From the raw zone-0 points of its icao in [first_seen, first_seen + 60 s]: displacement vector (m east/north) and
speed, its projection on the stand -> first-point direction (moving away from the stand = positive), mean/max gs and
track spread (turning). Also the same over [first_seen, first_seen + 180 s]. Label-free.
Output: data/processed/features_dep_adsbgeo2.parquet
  python adsb_geo2.py [--procs 6]
"""
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from adsb_geo import REF, xy
from ocommon import OUTD, PROC, ROOT, report, write_merged

PTS = ROOT / "data/external/adsblol/points"


def one(day):
    c = pd.read_parquet(OUTD / f"cache/adsb2/{day}.parquet")
    if "lat0" not in c.columns:
        return None
    c = c[(c.phase == "DEP") & c.first_seen.notna()][["MVT_ID_mvt", "apt", "icao", "first_seen", "lat0", "lon0"]]
    p = pd.read_parquet(PTS / f"{day}.parquet", columns=["icao", "t", "lat", "lon", "gs", "track", "zone"], filters=[("zone", "==", 0)])
    p = p[p.icao.isin(set(c.icao))]
    g = {k: v.sort_values("t") for k, v in p.groupby("icao")}
    rows = []
    for r in c.itertuples(index=False):
        q = g.get(r.icao)
        row = {"MVT_ID_mvt": r.MVT_ID_mvt}
        if q is not None:
            t = q.t.to_numpy()
            for w in (60, 180):
                m = (t >= r.first_seen - 1) & (t <= r.first_seen + w)
                if m.sum() >= 2:
                    s = q[m]
                    la, lo = s.lat.to_numpy(), s.lon.to_numpy()
                    dx = (lo[-1] - lo[0]) * 111320.0 * np.cos(np.radians(la[0])); dy = (la[-1] - la[0]) * 110540.0
                    dt = max(s.t.iloc[-1] - s.t.iloc[0], 1.0)
                    row.update({f"v{w}_dx": dx, f"v{w}_dy": dy, f"v{w}_sp": np.hypot(dx, dy) / dt, f"v{w}_gsm": s.gs.mean(),
                                f"v{w}_gsx": s.gs.max(), f"v{w}_n": int(m.sum()),
                                f"v{w}_trs": float(np.nanstd(np.unwrap(np.radians(s.track.dropna().to_numpy())))) if s.track.notna().sum() > 1 else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    days = sorted(Path(f).stem for f in glob.glob(str(OUTD / "cache/adsb2/*.parquet")) if (PTS / f"{Path(f).stem}.parquet").exists())
    run = "adsb-geo2"
    if not days:  # no raw points on this host (seeded phase): the seeded rows stand
        print("no raw-point days with a stage-1 cache; nothing to do")
        return
    report(run, 0, len(days), "start", purpose="Taxi map v2: first-heard motion (velocity, away-from-stand speed, turning) for matched departures")
    out = []
    with ProcessPoolExecutor(a.procs) as ex:
        for k, r in enumerate(ex.map(one, days, chunksize=4), 1):
            if r is not None and len(r):
                out.append(r)
            if k % 20 == 0:
                report(run, k, len(days), f"{k} days")
    d = pd.concat(out, ignore_index=True).drop_duplicates("MVT_ID_mvt")
    geo = pd.read_parquet(PROC / "features_dep_adsbgeo.parquet", columns=["MVT_ID_mvt", "g_dxs", "g_dys"])
    d = d.merge(geo, on="MVT_ID_mvt", how="left")
    nrm = np.hypot(d.g_dxs, d.g_dys).replace(0, np.nan)
    for w in (60, 180):
        d[f"v{w}_away"] = (d[f"v{w}_dx"] * d.g_dxs + d[f"v{w}_dy"] * d.g_dys) / nrm  # displacement along stand -> first point
    d = write_merged(PROC / "features_dep_adsbgeo2.parquet", d.drop(columns=["g_dxs", "g_dys"]))  # EC_MERGE=1: keep seeded rows
    report(run, len(days), len(days), f"done: {len(d)} departures", finished=True)
    print(len(d), d.describe().T.round(1).to_string())


if __name__ == "__main__":
    main()
