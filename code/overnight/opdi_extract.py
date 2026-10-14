# SPDX-License-Identifier: GPL-3.0-or-later
"""Compact OPDI (OpenSky-derived) extraction for the ten challenge airports, matched to departures.

Outputs (data/processed/):
  opdi_flights_10apt.parquet   flight list rows with ADEP or ADES in the ten airports (Dec 2024 - Dec 2025, Jan + Jul 2026)
  opdi_events_10apt.parquet    those flights' take-off/landing, FL50 and airport-surface events (OSM ref parsed from info)
  opdi_dep_match.parquet       MVT_ID_mvt -> OPDI flight (callsign + ADEP + first_seen nearest MVT within 30 min) with
                               surface/runway event times: diagnostic table, NOT a model feature file
  features_dep_ext_opdi.parquet  label-free airframe-rotation features only:
      opdi_matched, opdi_turn_min (SCHED minus the same icao24's previous OPDI last_seen landing at this airport, 3 days max),
      opdi_legs_today (flights of the same icao24 earlier the same UTC day, any airport)
Ground coverage is sparse (exit-parking_position on <0.1% of departures), so no block-time reconstruction.
"""
import re
from pathlib import Path

import polars as pl

from features import TZ
from ocommon import PROC, ROOT, report

OPDI = ROOT / "data/external/opdi"
RUN = "opdi-extract"
PURPOSE = "Extract OPDI flights/events for the ten airports, match them to departures and build airframe-rotation features"
EVENT_TYPES = ["first_seen", "last_seen", "take-off", "landing", "first-xing-fl50", "last-xing-fl50",
               "entry-runway", "exit-runway", "entry-taxiway", "exit-taxiway", "entry-parking_position",
               "exit-parking_position", "entry-apron", "exit-apron", "entry-deicing_pad", "exit-deicing_pad",
               "entry-threshold", "exit-threshold", "entry-hangar", "exit-hangar"]
APTS = list(TZ)
KEY = "MVT_ID_mvt"


def _norm(lf, id_cols, time_cols):
    """OPDI files change types from Sep-Dec 2025 on: ids int64 (same 64 bits) and timestamps as strings."""
    schema = lf.collect_schema()
    exprs = []
    for c in id_cols:
        if schema[c] == pl.Int64:
            exprs.append(pl.col(c).reinterpret(signed=False))
    for c in time_cols:
        if schema[c] == pl.String:
            exprs.append(pl.col(c).str.to_datetime(time_unit="us", strict=False))
    return lf.with_columns(exprs) if exprs else lf


def main():
    ev_files = sorted(OPDI.glob("flight_events_*.parquet"))
    total = 1 + len(ev_files) + 2
    step = 0
    report(RUN, step, total, "flight lists", purpose=PURPOSE)
    # Each stage is skipped when its output exists, so a crash costs one stage (delete a file to rebuild it).
    fl_path, ev_path, match_path = (PROC / f"opdi_{n}.parquet" for n in ("flights_10apt", "events_10apt", "dep_match"))
    if fl_path.exists():
        fl = pl.read_parquet(fl_path)
    else:
        fl = pl.concat([_norm(pl.scan_parquet(f), ["id"], ["dof", "first_seen", "last_seen"]).drop("unix_time", strict=False)
                        .collect() for f in sorted(OPDI.glob("flight_list_*.parquet"))],
                       how="vertical_relaxed")  # unix_time is absent from Sep 2025 onwards
        fl = fl.filter(pl.col("adep").is_in(APTS) | pl.col("ades").is_in(APTS)).unique("id").with_columns(
            pl.col("flt_id").str.strip_chars())
        fl.write_parquet(fl_path)
    step += 1
    report(RUN, step, total, f"{fl.height} flights touching the ten airports")

    dep = pl.read_parquet(PROC / "features_dep.parquet", columns=[KEY, "apt", "CALLSIGN_flt", "FLIGHT_mvt",
                                                                  "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt"])
    dep = dep.with_columns(cs=pl.coalesce(pl.col("CALLSIGN_flt"), pl.col("FLIGHT_mvt")).str.strip_chars())
    if match_path.exists():
        match = pl.read_parquet(match_path)
        step += len(ev_files) + 1
        report(RUN, step, total, f"reused match table: {match['opdi_id'].is_not_null().mean():.1%} of departures matched")
    else:
        match = _match(fl, dep, ev_files, ev_path, step, total)
        step += len(ev_files) + 1
    _rotation(fl, dep, match, step, total)


def _match(fl, dep, ev_files, ev_path, step, total):
    if ev_path.exists():
        ev = pl.read_parquet(ev_path)
        step += len(ev_files)
    else:
        ids = fl.select(pl.col("id").alias("flight_id"))
        parts = []
        for f in ev_files:
            e = (_norm(pl.scan_parquet(f).select("flight_id", "type", "event_time", "info"), ["flight_id"], ["event_time"])
                 .filter(pl.col("type").is_in(EVENT_TYPES)).join(ids.lazy(), on="flight_id", how="semi").collect())
            e = e.with_columns(osm_airport=pl.col("info").str.extract(r'"osm_airport":"([^"]*)"'),
                               osm_ref=pl.col("info").str.extract(r'"osm_ref":"([^"]*)"'),
                               time_in_use_s=pl.col("info").str.extract(r'"opdi_time_in_use_s":(\d+)').cast(pl.Int32)
                               ).drop("info")
            parts.append(e)
            step += 1
            report(RUN, step, total, f"{f.name}: {e.height} events")
        ev = pl.concat(parts).unique(["flight_id", "type", "event_time", "osm_ref"])
        ev.write_parquet(ev_path)

    # ---- match departures
    od = fl.filter(pl.col("adep").is_in(APTS)).select(
        opdi_id="id", apt="adep", cs="flt_id", icao24="icao24", registration="registration", typecode="typecode",
        fs_t=pl.col("first_seen").dt.epoch("s")).sort("fs_t")
    m = dep.sort("MVT_TIME_UTC_mvt").join_asof(od, left_on="MVT_TIME_UTC_mvt", right_on="fs_t", by=["apt", "cs"],
                                               strategy="nearest", tolerance=1800)
    # events per matched flight at the departure airport
    evs = ev.filter(pl.col("flight_id").is_in(m["opdi_id"].drop_nulls().implode())).with_columns(
        t=pl.col("event_time").dt.epoch("s"))

    def first_event(typ, at_airport=True, last=False, ref=False):
        name = typ.replace("-", "_")
        x = evs.filter(pl.col("type") == typ).rename({"flight_id": "opdi_id"}).sort("t")
        if at_airport:
            x = x.join(m.select("opdi_id", "apt").drop_nulls().unique(), on="opdi_id").filter(
                pl.col("osm_airport") == pl.col("apt"))
        pick = (lambda c: pl.col(c).last()) if last else (lambda c: pl.col(c).first())
        agg = [pick("t").alias(f"{name}_t")] + ([pick("osm_ref").alias(f"{name}_ref")] if ref else [])
        return x.group_by("opdi_id").agg(agg)

    match = m.select(KEY, "apt", "opdi_id", "icao24", "registration", "typecode", "fs_t", "MVT_TIME_UTC_mvt", "RUNWAY_mvt")
    for kw in (dict(typ="exit-parking_position", last=True, ref=True), dict(typ="entry-taxiway"),
               dict(typ="entry-runway", last=True, ref=True), dict(typ="take-off", at_airport=False),
               dict(typ="first-xing-fl50", at_airport=False)):
        match = match.join(first_event(**kw), on="opdi_id", how="left")
    match.write_parquet(PROC / "opdi_dep_match.parquet")
    step += 1
    report(RUN, step, total, f"matched {match['opdi_id'].is_not_null().mean():.1%} of departures")
    return match


def _rotation(fl, dep, match, step, total):
    # ---- airframe rotation features (label-free, available before off-block)
    arr = fl.filter(pl.col("ades").is_in(APTS) & (pl.col("icao24") != "")).select(
        apt="ades", icao24="icao24", land_t=pl.col("last_seen").dt.epoch("s")).sort("land_t")
    mm = match.join(dep.select(KEY, "SCHED_TIME_UTC_mvt"), on=KEY).filter(pl.col("icao24").is_not_null())
    tr = mm.sort("SCHED_TIME_UTC_mvt").join_asof(arr, left_on="SCHED_TIME_UTC_mvt", right_on="land_t",
                                                 by=["apt", "icao24"], strategy="backward", tolerance=3 * 86400)
    # Earlier legs the same UTC day: running count per (icao24, day), looked up with an as-of join.
    # (A plain icao24 join is many-to-many, ~10^8+ rows; it exhausted RAM and hard-froze the PC on 2026-09-15.)
    legs = (fl.filter(pl.col("icao24") != "").select("icao24", t=pl.col("first_seen").dt.epoch("s"))
            .with_columns(day=pl.col("t") // 86400).sort("t")
            .with_columns(opdi_legs_today=pl.int_range(1, pl.len() + 1).over("icao24", "day")))
    q = (mm.select("icao24", "fs_t").unique()
         .with_columns(day=pl.col("fs_t") // 86400, q_t=pl.col("fs_t") - 301).sort("q_t"))
    q = q.join_asof(legs, left_on="q_t", right_on="t", by=["icao24", "day"], strategy="backward")
    tr = tr.join(q.select("icao24", "fs_t", "opdi_legs_today"), on=["icao24", "fs_t"], how="left")
    feat = dep.select(KEY).join(tr.select(
        KEY, opdi_turn_min=((pl.col("SCHED_TIME_UTC_mvt") - pl.col("land_t")) / 60).cast(pl.Float32),
        opdi_legs_today=pl.col("opdi_legs_today").fill_null(0).cast(pl.Int16)), on=KEY, how="left")
    feat = feat.join(match.select(KEY, opdi_matched=pl.col("opdi_id").is_not_null().cast(pl.Int8)), on=KEY, how="left")
    assert feat.height == feat[KEY].n_unique() == dep.height
    feat.write_parquet(PROC / "features_dep_ext_opdi.parquet")
    step += 1
    cov = match.select([pl.col(c).is_not_null().mean().alias(c) for c in match.columns if c.endswith("_t")])
    report(RUN, step, total, "done; event coverage " + ", ".join(f"{k}={v:.1%}" for k, v in cov.row(0, named=True).items()),
           finished=True)


if __name__ == "__main__":
    main()
