# SPDX-License-Identifier: GPL-3.0-or-later
"""Combine all 2025 training and the test-phase movements into one processed parquet.

Times become int64 UTC epoch seconds; the split column marks train/rank. 'rank' is ranking.parquet (Jan/Jul 2026) in the
ranking phase and final_ranking.parquet (Jan/Jul plus two further months) when EC_PHASE=final (or pseudo): see ocommon.py.
"""
import sys

from ocommon import PHASE, PROC as OUT, RAW, TEST_FILE  # first: sets the polars extension-type env var

import polars as pl

TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt",
             "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]


def load(path, split):
    df = pl.read_parquet(path)
    exprs = [pl.col(c).dt.epoch("s").alias(c) for c in TIME_COLS]
    df = df.with_columns(exprs).with_columns(
        pl.lit(split).alias("split"),
        pl.col("MVT_ID_mvt").cast(pl.Int64),
        pl.col("FLIGHT_ID_mvt").cast(pl.Int64),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
        pl.col("AIRCRAFT_OPERATOR_flt").cast(pl.Utf8),
    )
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [load(p, "train") for p in sorted(RAW.glob("training_*.parquet"))]
    frames.append(load(TEST_FILE, "rank"))
    df = pl.concat(frames, how="vertical_relaxed").sort("MVT_TIME_UTC_mvt")
    df.write_parquet(OUT / "movements.parquet")
    print(df.shape, df.group_by("split", "PHASE_mvt").len())
    mo = df.filter(pl.col("split") == "rank").select(pl.from_epoch("MVT_TIME_UTC_mvt", time_unit="s").dt.strftime("%Y-%m").alias("m")).get_column("m").value_counts().sort("m")
    print(f"phase {PHASE or 'ranking'}: test file {TEST_FILE.name}, months", dict(zip(mo["m"], mo["count"])))


if __name__ == "__main__":
    sys.exit(main())
