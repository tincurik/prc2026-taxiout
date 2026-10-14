# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic stand-ins for the open external data of data/external/ (METAR, EUROCONTROL day tables, holidays, OSM taxi network, OPDI
flights and events) for the smoke test.  Same file names and columns the loaders in overnight/features_ext.py, opdi_extract.py read; made-up values.

  python smoke/make_external.py --raw data/raw --truth /path/synthetic_truth.parquet --out data/external
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from make_adsb import APTS, stand_xy, to_ll

MONTHS = [f"2025-{m:02d}" for m in range(1, 13)] + ["2026-01", "2026-02", "2026-06", "2026-07"]
RUNWAYS = [f"{r:02d}{s}" for r in (7, 9, 16, 25, 27) for s in ("L", "R")][:6]


def metar(out, rng):
    d = out / "metar"
    d.mkdir(parents=True, exist_ok=True)
    for apt in APTS:
        rows = []
        for mo in MONTHS:
            for t in pd.date_range(f"{mo}-01", periods=10 * 48, freq="30min"):
                tmp = 10 + 12 * np.sin((t.dayofyear - 100) / 365 * 2 * np.pi) + rng.normal(0, 3)
                snow = tmp < 1 and rng.random() < 0.3
                ts = rng.random() < 0.02
                vis = 0.5 if rng.random() < 0.03 else 6.2
                gust = rng.integers(20, 35) if rng.random() < 0.1 else None
                wx = ("SN " if snow else "") + ("TS " if ts else "")
                kt = int(rng.integers(2, 25))
                rows.append(dict(station=apt, valid=t.strftime("%Y-%m-%d %H:%M"), tmpf=round(tmp * 1.8 + 32, 1), dwpf=round(tmp * 1.8 + 28, 1),
                                 relh=round(float(rng.uniform(40, 100)), 1), drct=int(rng.integers(0, 36)) * 10, sknt=kt, gust=gust, vsby=vis,
                                 skyc1="BKN" if rng.random() < 0.3 else "FEW", skyc2=None, skyc3=None, skyc4=None,
                                 skyl1=int(rng.integers(5, 40)) * 100, skyl2=None, skyl3=None, skyl4=None,
                                 metar=f"{apt} {t.strftime('%d%H%M')}Z {int(rng.integers(0, 36)) * 10:03d}{kt:02d}KT 9999 {wx}FEW040 12/08 Q1015"))
        pd.DataFrame(rows).to_csv(d / f"{apt}_synthetic.csv", index=False)


def prc(out, rng):
    d = out / "prc"
    d.mkdir(parents=True, exist_ok=True)
    for year in (2025, 2026):
        days = [x for mo in MONTHS if mo.startswith(str(year)) for x in pd.date_range(f"{mo}-01", periods=10)]
        base = [dict(APT_ICAO=a, FLT_DATE=f"{x:%Y-%m-%d}") for a in APTS for x in days]
        n = len(base)
        tr = pd.DataFrame(base).assign(FLT_DEP_IFR_2=rng.integers(300, 800, n), FLT_DEP_1=rng.integers(300, 800, n),
                                       FLT_ARR_IFR_2=rng.integers(300, 800, n), FLT_ARR_1=rng.integers(300, 800, n))
        tr.to_csv(d / f"airport_traffic_{year}.csv", index=False)
        dl = pd.DataFrame(base).assign(DLY_APT_ARR_1=rng.integers(0, 3000, n), FLT_ARR_1=rng.integers(300, 800, n), FLT_ARR_1_DLY_15=rng.integers(0, 60, n))
        for c in ("W", "C", "G", "S", "I", "N", "D", "T", "E", "A", "M", "O", "P", "R", "V", "NA"):
            dl[f"DLY_APT_ARR_{c}_1"] = rng.integers(0, 200, n)
        dl.to_csv(d / f"apt_dly_{year}.csv", index=False)
        sa = pd.DataFrame(base).assign(FLT_DEP_REG_1=rng.integers(0, 100, n), FLT_DEP_1=rng.integers(300, 800, n), FLT_DEP_OUT_LATE_1=rng.integers(0, 30, n))
        sa.to_csv(d / f"atfm_slot_adherence_{year}.csv", index=False)
        tf = rng.integers(8000, 12000, len(APTS) * 12)
        pd.DataFrame([dict(APT_ICAO=a, YEAR=year, MONTH_NUM=m) for a in APTS for m in range(1, 13)]).assign(
            TF=tf, VALID_FL=(tf * rng.uniform(0.97, 1.0, len(tf))).astype(int)).to_csv(d / f"taxi_out_additional_time_{year}.csv", index=False)
        pd.DataFrame(base).assign(DLY_ALL_PRE_2=rng.integers(0, 40, n)).to_csv(d / f"all_pre_departure_delays_{year}.csv", index=False)
    c = out / "calendars"
    c.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(airport=a, date=f"{y}-{md}") for a in APTS for y in (2025, 2026) for md in ("01-01", "01-06", "05-01", "07-14", "12-25")]).to_csv(
        c / "public_holidays.csv", index=False)


def osm(out):
    d = out / "osm"
    d.mkdir(parents=True, exist_ok=True)
    for apt in APTS:
        nid, ids, els = 1, {}, []

        def node(x, y):
            if (x, y) not in ids:
                nonlocal_nid[0] += 1
                ids[(x, y)] = nonlocal_nid[0]
            return ids[(x, y)]

        nonlocal_nid = [1000]
        lines = []
        for y in (-400, 0, 400, 800):
            lines.append([(x, y) for x in range(-200, 801, 100)])
        for x in (-100, 300, 700):
            lines.append([(x, y) for y in range(-400, 801, 100)])
        lines.append([(x, -400) for x in range(800, 1801, 100)])
        way = 1
        for ln in lines:
            way += 1
            els.append({"type": "way", "id": way, "tags": {"aeroway": "taxiway"}, "nodes": [node(*p) for p in ln],
                        "geometry": [dict(zip(("lat", "lon"), to_ll(apt, *p))) for p in ln]})
        for des in RUNWAYS:
            h = np.radians(int(des[:2]) * 10)
            end = (1800 + 2500 * np.sin(h), -400 + 2500 * np.cos(h))
            way += 1
            els.append({"type": "way", "id": way, "tags": {"aeroway": "runway", "ref": des}, "nodes": [node(1800, -400), node(round(end[0]), round(end[1]))],
                        "geometry": [dict(zip(("lat", "lon"), to_ll(apt, *p))) for p in ((1800, -400), end)]})
        for n in range(1, 81):
            x, y = stand_xy(f"S{n}")
            lat, lon = to_ll(apt, x, y)
            els.append({"type": "node", "id": 9000 + n, "lat": float(lat), "lon": float(lon), "tags": {"aeroway": "parking_position", "ref": f"S{n}"}})
        lats, lons = zip(*(to_ll(apt, x, y) for x, y in ((-300, -900), (3500, 900))))
        els.insert(0, {"type": "way", "id": 1, "tags": {"aeroway": "aerodrome"}, "nodes": [], "geometry": [],
                       "bounds": {"minlat": float(min(lats)), "maxlat": float(max(lats)), "minlon": float(min(lons)), "maxlon": float(max(lons))}})
        (d / f"{apt}.json").write_text(json.dumps({"elements": els}), encoding="utf-8")


def opdi(out, raw, truth, rng):
    d = out / "opdi"
    d.mkdir(parents=True, exist_ok=True)
    tx = pd.read_parquet(truth).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    fl, ev, seen = [], [], set()
    for f in sorted(raw.glob("training_*.parquet")) + [raw / "final_ranking.parquet"]:
        df = pq.read_table(f).to_pandas()
        df = df[~df.MVT_ID_mvt.isin(seen)]
        seen |= set(df.MVT_ID_mvt)
        df = df[rng.random(len(df)) < 0.7]
        for r in df.itertuples(index=False):
            mvt = r.MVT_TIME_UTC_mvt.tz_convert("UTC").tz_localize(None)
            key = int(r.FLIGHT_ID_mvt if pd.notna(r.FLIGHT_ID_mvt) else r.MVT_ID_mvt)
            fid = key + 10**12
            icao = f"{key % 0xFFFFFF:06x}"
            if r.PHASE_mvt == "DEP":
                taxi = r.TAXITIME_SEC_mvt if pd.notna(r.TAXITIME_SEC_mvt) else tx.get(r.MVT_ID_mvt, np.nan)
                if pd.isna(taxi):
                    continue
                fl.append((fid, icao, r.CALLSIGN_flt, r.ADEP_mvt, r.ADES_mvt, mvt - pd.Timedelta(seconds=float(taxi) + 120), mvt + pd.Timedelta(hours=2)))
                if rng.random() < 0.7:
                    for typ, dt_ in (("exit-parking_position", -taxi + 60), ("entry-taxiway", -taxi + 90 + rng.normal(0, 20)), ("entry-runway", -30), ("take-off", 0)):
                        ev.append((fid, typ, mvt + pd.Timedelta(seconds=float(dt_)), f'{{"osm_airport":"{r.ADEP_mvt}","osm_ref":"{r.STAND_mvt if typ == "exit-parking_position" else RUNWAYS[0]}","opdi_time_in_use_s":30}}'))
            else:
                fl.append((fid, icao, r.CALLSIGN_flt, r.ADEP_mvt, r.ADES_mvt, mvt - pd.Timedelta(hours=2), mvt))
    flights = pd.DataFrame(fl, columns=["id", "icao24", "flt_id", "adep", "ades", "first_seen", "last_seen"])
    flights["registration"] = "SY-" + flights.icao24.str[-3:]
    flights["typecode"] = "A320"
    flights["dof"] = flights.first_seen.dt.floor("D")
    flights["id"] = flights["id"].astype("uint64")
    flights.to_parquet(d / "flight_list_synthetic.parquet", index=False)
    events = pd.DataFrame(ev, columns=["flight_id", "type", "event_time", "info"])
    events["flight_id"] = events["flight_id"].astype("uint64")
    events.to_parquet(d / "flight_events_synthetic.parquet", index=False)
    print(f"opdi: {len(flights):,} flights, {len(events):,} events")


def prc_dashboard(out, rng):
    """The runway-level monthly table of fetch_prc_dashboard.py (reference + additional taxi-out time per runway and month)."""
    d = out / "prc_dashboard"
    d.mkdir(parents=True, exist_ok=True)
    rows = [dict(AIRPORT=a, YEAR=int(mo[:4]), RUNWAY=r, TOT_REF_TIME=float(rng.uniform(1e5, 2e5)), TOT_ADD_TIME=float(rng.uniform(2e4, 6e4)),
                 TOT_FLT=int(rng.integers(300, 900)), TYPE="Avg. Reference time", AVG_TXOT_TIME=9.5, MONTH=float(int(mo[5:])))
            for a in APTS for mo in MONTHS for r in RUNWAYS]
    pd.DataFrame(rows).to_csv(d / "runway_tot_ref_time_tot_add_time_tot_flt_type_avg_txot_time.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/external"))
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    metar(a.out, rng)
    prc(a.out, rng)
    prc_dashboard(a.out, rng)
    osm(a.out)
    opdi(a.out, a.raw, a.truth, rng)
    print(f"synthetic external data in {a.out}")


if __name__ == "__main__":
    main()
