#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Pseudo/final phase, label-free feature stage 1 (no ADS-B): EC_PHASE=pseudo|final bash overnight/pf_feats1.sh
cd "$(dirname "$0")"
PY=${PY:-python}
export EC_PHASE
set -e
$PY -u features.py
$PY -u features2.py
$PY -u features3.py
$PY -u features_ext.py metar prc cal osm
