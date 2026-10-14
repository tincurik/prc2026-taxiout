# SPDX-License-Identifier: GPL-3.0-or-later
"""Inbound-arrival features: link each departure to the SAME airframe's previous arrival movement at this airport.

departure --(opdi_dep_match: icao24)--> the icao24's last OPDI landing at this airport before SCHED (3 days max)
          --(apt + callsign, landing time nearest the arrival MVT within 30 min)--> ARR movement row.
Arrival block times are not masked in ranking, so these are label-free. fs2's fa_prevarr_* guessed the inbound
from the stand; this uses the airframe. Output data/processed/features_dep_ext_inb.parquet (mix.py --extra ext_inb).

  inb_imp          inbound in-block recorded at schedule (|BLOCK - SCHED| <= 30 s): handler imputation on the inbound leg
  inb_delay        inbound BLOCK - SCHED (s)
  inb_taxiin       inbound taxi-in (s)
  inb_gap_sched    departure SCHED - inbound in-block (s): scheduled ground time left
  inb_gap_mvt      departure MVT - inbound in-block (s): actual ground time incl. taxi-out
  inb_sched_turn   departure SCHED - inbound SCHED (s)
  inb_same_stand   inbound stand == departure stand
  inb_same_airline inbound flight prefix == departure prefix
"""
import polars as pl

from ocommon import PROC, report

RUN = "features-inbound"
KEY = "MVT_ID_mvt"


def main():
    report(RUN, 0, 4, "loading", purpose="Does the airframe's own inbound arrival (imputed? late?) explain departure taxi/imputation?")
    fl = pl.read_parquet(PROC / "opdi_flights_10apt.parquet", columns=["id", "icao24", "flt_id", "ades", "last_seen"])
    mv = pl.read_parquet(PROC / "movements.parquet", columns=[KEY, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "CALLSIGN_flt",
                                                             "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                                                             "STAND_mvt", "TAXITIME_SEC_mvt"])
    cs = pl.coalesce(pl.col("CALLSIGN_flt"), pl.col("FLIGHT_mvt")).str.strip_chars()
    arr = mv.filter(pl.col("PHASE_mvt") == "ARR").select(
        arr_id=KEY, apt="ADES_mvt", cs=cs, a_mvt="MVT_TIME_UTC_mvt", a_blk="BLOCK_TIME_UTC_mvt", a_sched="SCHED_TIME_UTC_mvt",
        a_stand="STAND_mvt", a_taxiin="TAXITIME_SEC_mvt", a_pre=pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{3})", 1))
    dep = mv.filter(pl.col("PHASE_mvt") == "DEP").select(
        KEY, apt="ADEP_mvt", d_mvt="MVT_TIME_UTC_mvt", d_sched="SCHED_TIME_UTC_mvt", d_stand="STAND_mvt",
        d_pre=pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{3})", 1))

    # OPDI arrivals -> ARR movement rows (callsign + nearest landing time)
    oa = fl.filter(pl.col("ades").is_in(arr["apt"].unique().implode()) & (pl.col("icao24") != "")).select(
        opdi_arr="id", apt="ades", cs=pl.col("flt_id"), icao24="icao24", land_t=pl.col("last_seen").dt.epoch("s")).sort("land_t")
    oa = oa.join_asof(arr.sort("a_mvt"), left_on="land_t", right_on="a_mvt", by=["apt", "cs"], strategy="nearest",
                      tolerance=1800)
    report(RUN, 1, 4, f"OPDI arrivals linked to ARR movements: {oa['arr_id'].is_not_null().mean():.1%}")

    # departure -> icao24 -> previous OPDI landing at this airport before SCHED
    match = pl.read_parquet(PROC / "opdi_dep_match.parquet", columns=[KEY, "icao24"]).filter(pl.col("icao24").is_not_null())
    q = dep.join(match, on=KEY).sort("d_sched")
    q = q.join_asof(oa.drop("cs"), left_on="d_sched", right_on="land_t", by=["apt", "icao24"], strategy="backward",
                    tolerance=3 * 86400)
    report(RUN, 2, 4, f"departures with an inbound OPDI landing: {q['land_t'].is_not_null().mean():.1%}, "
                      f"with an ARR movement: {q['arr_id'].is_not_null().mean():.1%}")
    ok = pl.col("arr_id").is_not_null()
    blk = pl.coalesce(pl.col("a_blk"), pl.col("a_mvt") + 420)
    f = q.select(
        KEY,
        inb_imp=pl.when(ok & pl.col("a_blk").is_not_null()).then(((pl.col("a_blk") - pl.col("a_sched")).abs() <= 30).cast(pl.Float32)),
        inb_delay=pl.when(ok).then((pl.col("a_blk") - pl.col("a_sched")).cast(pl.Float32)),
        inb_taxiin=pl.when(ok).then(pl.col("a_taxiin").cast(pl.Float32)),
        inb_gap_sched=pl.when(ok).then((pl.col("d_sched") - blk).cast(pl.Float32)),
        inb_gap_mvt=pl.when(ok).then((pl.col("d_mvt") - blk).cast(pl.Float32)),
        inb_sched_turn=pl.when(ok).then((pl.col("d_sched") - pl.col("a_sched")).cast(pl.Float32)),
        inb_same_stand=pl.when(ok).then((pl.col("a_stand") == pl.col("d_stand")).cast(pl.Float32)),
        inb_same_airline=pl.when(ok).then((pl.col("a_pre") == pl.col("d_pre")).cast(pl.Float32)),
    )
    feat = dep.select(KEY).join(f, on=KEY, how="left")
    assert feat.height == feat[KEY].n_unique() == dep.height
    report(RUN, 3, 4, "writing")
    feat.write_parquet(PROC / "features_dep_ext_inb.parquet")
    report(RUN, 4, 4, f"done: inbound arrival for {feat['inb_delay'].is_not_null().mean():.1%} of departures", finished=True)


if __name__ == "__main__":
    main()
