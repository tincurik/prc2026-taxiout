# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic ADS-B ground points for the smoke test (made-up traces for the made-up flights of make_synthetic.py; same columns as the
files scripts/adsb_fetch.py writes, so the ADS-B feature code runs on them unchanged).

Per departure (about 60 % are heard at all): parked at its stand until a little after off-block, taxi at ~12 kt with a hold at the runway,
roll to 150 kt, airborne points.  Per arrival: approach, touch-down, taxi in, parked.

  python smoke/make_adsb.py --raw data/raw --truth /path/synthetic_truth.parquet --out data/external
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

APTS = {"EDDF": (50.026706, 8.558350), "EDDM": (48.353802, 11.786100), "EGLL": (51.470748, -0.459909), "EHAM": (52.308601, 4.763890),
        "LEBL": (41.297100, 2.078460), "LEMD": (40.493407, -3.572249), "LFPG": (49.008960, 2.554117), "LIRF": (41.804532, 12.251998),
        "LSZH": (47.458056, 8.548056), "LTFM": (41.274874, 28.732136)}
M = 111_195.0
STEP = 10.0


def stand_xy(stand):
    n = int(str(stand)[1:])
    return ((n % 9) - 4) * 90.0 + 300.0, (n // 9) * 90.0 - 300.0     # metres east / north of the reference point


def to_ll(apt, x, y):
    lat0, lon0 = APTS[apt]
    return lat0 + y / M, lon0 + x / (M * np.cos(np.radians(lat0)))


def frame(icao, typ, cs, apt, t, x, y, gs, ground, alt, track):
    lat, lon = to_ll(apt, x, y)
    n = len(t)
    return pd.DataFrame({"icao": icao, "reg": "SY-" + icao[-3:], "type": typ, "t": t, "lat": lat.astype("float32"), "lon": lon.astype("float32"),
                         "alt": alt.astype("float32"), "ground": ground, "gs": gs.astype("float32"), "track": track.astype("float32"),
                         "flags": np.zeros(n, "int8"), "vrate": np.zeros(n, "float32"), "alt_geom": alt.astype("float32"), "ias": gs.astype("float32"),
                         "src": "adsb_icao", "callsign": cs, "squawk": "1000", "apt": apt,
                         "dist_km": (np.hypot(x, y) / 1000).astype("float32"), "zone": np.zeros(n, "int8")})


def dep_trace(r, block, rng):
    x0, y0 = stand_xy(r.STAND_mvt)
    mvt = r.MVT_TIME_UTC_mvt.timestamp()
    b = block.timestamp()
    seen = b - rng.integers(0, 600) if rng.random() < 0.6 else b + rng.integers(60, 400)   # first heard parked, or only after pushback
    taxi_end = mvt - 25.0
    ts = np.arange(seen, taxi_end, STEP)
    ts = ts[ts < mvt - 10]
    if len(ts) < 3:
        return None
    push = b + rng.integers(0, 120)
    dx, dy = 1800.0 + rng.integers(0, 400), -700.0 + rng.integers(0, 300)     # the runway end, metres from the reference point
    u = np.array([dx - x0, dy - y0]) / np.hypot(dx - x0, dy - y0)
    heading = np.degrees(np.arctan2(u[0], u[1])) % 360
    hold = rng.integers(20, 90)
    pb = np.clip(ts - push, 0, 60.0)                                              # pushback: 60 s tail-first at 3 m/s, nose along the taxi direction
    moving = (ts > push + 60) & (ts < taxi_end - hold)
    frac = np.clip((ts - push - 60) / max(taxi_end - hold - push - 60, 1.0), 0, 1)
    px, py = x0 - u[0] * pb * 3.0, y0 - u[1] * pb * 3.0
    x, y = px + (dx - px) * frac, py + (dy - py) * frac
    gs = np.where(moving, 12.0 + rng.normal(0, 1.5, len(ts)), np.where((ts > push) & (ts <= push + 60), 6.0, 0.0))
    roll = np.arange(mvt - 25, mvt + 40, 5.0)
    rx, ry = dx + (roll - (mvt - 25)) * 40.0, np.full(len(roll), dy)
    rgs = np.clip(20 + (roll - (mvt - 25)) * 3.4, 0, 170)
    air = roll >= mvt
    t = np.r_[ts, roll]
    X, Y = np.r_[x, rx], np.r_[y, ry]
    G = np.r_[gs, rgs]
    ground = np.r_[np.ones(len(ts), bool), ~air]
    alt = np.r_[np.zeros(len(ts)), np.where(air, 400.0 + (roll - mvt) * 10, 0.0)]
    return t, X, Y, G, ground, alt, np.full(len(t), heading)


def arr_trace(r, rng):
    x0, y0 = stand_xy(r.STAND_mvt)
    lt = r.MVT_TIME_UTC_mvt.timestamp()
    end = r.BLOCK_TIME_UTC_mvt.timestamp()
    app = np.arange(lt - 60, lt, 5.0)
    ts = np.arange(lt, end, STEP)
    if len(ts) < 3:
        return None
    frac = np.clip((ts - lt - 60) / max(end - lt - 60, 1.0), 0, 1)
    x, y = 2200.0 + (x0 - 2200.0) * frac, -600.0 + (y0 + 600.0) * frac
    gs = np.where(ts - lt < 60, 60.0 - (ts - lt), np.where(frac < 1, 12.0, 0.0))
    gs = np.clip(gs, 0, None)
    t = np.r_[app, ts]
    X, Y = np.r_[np.full(len(app), 2200.0) - (lt - app) * 40.0, x], np.r_[np.full(len(app), -600.0), y]
    G = np.r_[np.full(len(app), 140.0), gs]
    ground = np.r_[np.zeros(len(app), bool), np.ones(len(ts), bool)]
    alt = np.r_[np.full(len(app), 300.0), np.zeros(len(ts))]
    return t, X, Y, G, ground, alt, np.full(len(t), 270.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/external"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    truth = pd.read_parquet(a.truth).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    files = sorted(a.raw.glob("training_*.parquet")) + [a.raw / "final_ranking.parquet"]
    seen_ids = set()
    frames = []
    for f in files:
        d = pq.read_table(f).to_pandas()
        d = d[~d.MVT_ID_mvt.isin(seen_ids)]
        seen_ids |= set(d.MVT_ID_mvt)
        for r in d.itertuples(index=False):
            if rng.random() > 0.6 or (r.ADEP_mvt if r.PHASE_mvt == "DEP" else r.ADES_mvt) not in APTS:
                continue
            icao = f"{int(r.FLIGHT_ID_mvt if pd.notna(r.FLIGHT_ID_mvt) else r.MVT_ID_mvt) % 0xFFFFFF:06x}"
            if r.PHASE_mvt == "DEP":
                taxi = r.TAXITIME_SEC_mvt if pd.notna(r.TAXITIME_SEC_mvt) else truth.get(r.MVT_ID_mvt, np.nan)
                if pd.isna(taxi):
                    continue
                tr = dep_trace(r, r.MVT_TIME_UTC_mvt - pd.Timedelta(seconds=float(taxi)), rng)
            else:
                tr = arr_trace(r, rng)
            if tr is None:
                continue
            t, X, Y, G, gr, alt, trk = tr
            frames.append(frame(icao, r.AIRCRAFT_TYPE_mvt, r.CALLSIGN_flt if pd.notna(r.CALLSIGN_flt) else "", r.ADEP_mvt if r.PHASE_mvt == "DEP" else r.ADES_mvt, t, X, Y, G, gr, alt, trk))
    pts = pd.concat(frames, ignore_index=True)
    pts["day"] = pd.to_datetime(pts.t, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    out = a.out / "adsblol" / "points"
    out.mkdir(parents=True, exist_ok=True)
    for day, g in pts.groupby("day"):
        g.drop(columns="day").sort_values(["icao", "t"]).to_parquet(out / f"{day}.parquet", index=False)
    print(f"synthetic ADS-B points: {len(pts):,} rows over {pts.day.nunique()} days -> {a.out}")


if __name__ == "__main__":
    main()
