#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Stage 1: label-free feature files in data/processed/ (every feature uses only columns present in the ranking inputs).
set -e
PY=${PY:-python}
cd "$(dirname "$0")/../overnight"
$PY -u features.py                      # features_dep.parquet (fs1)
$PY -u features2.py                     # fs2: arrival imputation rates, previous arrival at the stand (in place)
$PY -u features3.py                     # features_dep_fs3.parquet: stand / stand x runway rolling proxies
$PY -u features_ext.py metar prc cal osm  # features_dep_ext_{metar,prc,cal,osm}.parquet
$PY -u opdi_extract.py                  # OPDI extraction + opdi_dep_match.parquet (needs data/external/opdi, see docs/data-sources.md)
$PY -u features_opdiev.py               # features_dep_ext_opdiev.parquet
$PY -u features_inbound.py              # features_dep_ext_inb.parquet (read by lirf_far.py)
$PY -u features_ord.py                  # features_dep_ext_ord.parquet (record ordering, gate backlog, route-centred durations)

# ADS-B features. adsb2full = v2 events over every cached day; adsb2fullrel = member features (events minus NM clocks).
$PY -u features_adsb.py --workers 8                         # features_dep_adsb.parquet (take-off match, used by the stacker)
$PY -u features_adsb2.py --out adsb2full --workers 8        # features_dep_adsb2full.parquet
$PY -u features_adsb_rel.py --src adsb2full                 # features_dep_adsb2fullrel.parquet
$PY -u features_adsb2.py --ext 1 --out adsb2m --workers 8   # extended match (IATA->ICAO prefix map, registration)
$PY -u features_adsb2.py --workers 8                        # features_dep_adsb2.parquet (default match; stacker base feature set)
$PY -u adsb_geo.py                                          # features_dep_adsbgeo.parquet (first-heard position, the taxi map)
$PY -u adsb_geo2.py                                         # features_dep_adsbgeo2.parquet (first-heard motion)
$PY -u adsb_direct.py --geo 2                               # features_dep_adsbdir4.parquet (month-complete ADS-B model)
$PY -u features_adsbrev.py                                  # features_dep_adsbrev.parquet (pushback from reverse motion)

