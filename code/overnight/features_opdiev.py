# SPDX-License-Identifier: GPL-3.0-or-later
"""Own-flight OPDI surface events as off-block proxies (label-free: ADS-B, open data).

From opdi_dep_match (departure <-> OPDI flight): seconds from each surface event to the movement's take-off time.
  ev_tax   MVT - first entry-taxiway at the airport   (LSZH: |err| median 31 s vs y, coverage ~70% train and 2026)
  ev_park  MVT - last exit-parking_position
  ev_rwy   MVT - last entry-runway
  ev_tax_m_aobt  ev_tax - (MVT - AOBT_3)
Output data/processed/features_dep_ext_opdiev.parquet (mix.py --extra ext_opdiev).
"""
import polars as pl

from ocommon import PROC

KEY = "MVT_ID_mvt"
m = pl.read_parquet(PROC / "opdi_dep_match.parquet")
dep = pl.read_parquet(PROC / "movements.parquet", columns=[KEY, "PHASE_mvt", "AOBT_3_flt"]).filter(pl.col("PHASE_mvt") == "DEP")
f = m.join(dep, on=KEY, how="right").select(
    KEY,
    ev_tax=(pl.col("MVT_TIME_UTC_mvt") - pl.col("entry_taxiway_t")).cast(pl.Float32),
    ev_park=(pl.col("MVT_TIME_UTC_mvt") - pl.col("exit_parking_position_t")).cast(pl.Float32),
    ev_rwy=(pl.col("MVT_TIME_UTC_mvt") - pl.col("entry_runway_t")).cast(pl.Float32),
    ev_tax_m_aobt=(pl.col("AOBT_3_flt") - pl.col("entry_taxiway_t")).cast(pl.Float32),
)
f = f.with_columns([pl.when(pl.col(c).is_between(-600, 7200)).then(pl.col(c)).alias(c) for c in ("ev_tax", "ev_park", "ev_rwy")])
assert f.height == f[KEY].n_unique() == dep.height
f.write_parquet(PROC / "features_dep_ext_opdiev.parquet")
print(f.select([pl.col(c).is_not_null().mean() for c in f.columns if c != KEY]))
