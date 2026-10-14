# SPDX-License-Identifier: GPL-3.0-or-later
"""Arrival-witness table of the rules chain: for a departure that has no matched flight record, the LOBT carried by its own
arrival record at the destination (same flight number, same ADEP/ADES, landing within 6 h after the departure's movement time).

  python scripts/build_witness_cache.py [--phase ""|final|pseudoN]

Writes outputs/overnight<SFX>/cache/arr_witness.parquet (MVT_ID_mvt, w_lobt in epoch seconds) for every departure without a flight record (training and ranking rows).
It writes only when the file is missing: the pseudo/final phases receive the base phase's table through scripts/phase_seed.py.
"""
import argparse
import os
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def build(mv):
    dep = mv.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("FLIGHT_ID_mvt").is_null())
    dep = dep.select("MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", td="MVT_TIME_UTC_mvt")
    arr = (mv.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("LOBT_flt").is_not_null())
           .select("FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", ta="MVT_TIME_UTC_mvt", w_lobt="LOBT_flt"))
    m = dep.join(arr, on=["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"])
    m = m.filter((pl.col("ta") > pl.col("td")) & (pl.col("ta") - pl.col("td") <= 6 * 3600))
    return m.sort("ta").unique("MVT_ID_mvt", keep="first", maintain_order=True).select("MVT_ID_mvt", pl.col("w_lobt").cast(pl.Float64))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default=os.environ.get("EC_PHASE", ""))
    a = ap.parse_args()
    sfx = f"_{a.phase}" if a.phase else ""
    out = ROOT / f"outputs/overnight{sfx}/cache/arr_witness.parquet"
    if out.exists():
        print(f"{out} exists; kept")
        return
    mv = pl.read_parquet(ROOT / f"data/processed{sfx}/movements.parquet")
    w = build(mv)
    out.parent.mkdir(parents=True, exist_ok=True)
    w.write_parquet(out)
    print(f"arr_witness: {w.height} departures -> {out}")


if __name__ == "__main__":
    main()
