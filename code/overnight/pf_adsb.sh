#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Pseudo/final phase, ADS-B part of feature stage 2 (split out of pf_feats2.sh so it can be rerun alone as raw days arrive).
# EC_PHASE=pseudo|final [MONTHS=2026-01,2026-02] bash overnight/pf_adsb.sh [workers]; MONTHS limits the day-wise stage 1 to months whose
# raw days (and the boundary days around them) are all on disk. Stage-1 results are cached per day (cache/<name>), so a rerun only
# computes days that have raw points and no cache yet; days whose neighbouring day is not on disk yet must not be run
# (each day reads the previous 4 h and the next 1 h). 2025 days come from the seeded caches; EC_MERGE=1 keeps the seeded rows.
cd "$(dirname "$0")"
PY=${PY:-python}
W=${1:-6}
MO=${MONTHS:+--months $MONTHS}
export EC_PHASE EC_MERGE=1
set -e
$PY -u features_adsb.py --workers $W $MO
$PY -u features_adsb2.py --out adsb2full --workers $W $MO
$PY -u features_adsb_rel.py --src adsb2full
$PY -u features_adsb2.py --ext 1 --out adsb2m --workers $W $MO
$PY -u features_adsb2.py --workers $W $MO
$PY -u adsb_geo.py
$PY -u adsb_geo2.py
$PY -u features_adsbrev.py --workers $W $MO
