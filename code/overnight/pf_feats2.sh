#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Pseudo/final phase, label-free feature stage 2 (OPDI, inbound, ordering, ADS-B from the seeded + new stage-1 caches).
# EC_PHASE=pseudo|final bash overnight/pf_feats2.sh [workers]; run pf_feats1.sh first; the ADS-B steps are pf_adsb.sh. Raw ADS-B points are only needed for
# days without a seeded cache (the final phase's new days); EC_MERGE=1 keeps the seeded 2025 rows of adsbrev / adsbgeo2.
cd "$(dirname "$0")"
PY=${PY:-python}
W=${1:-6}
export EC_PHASE EC_MERGE=1
set -e
$PY -u opdi_extract.py
$PY -u features_opdiev.py
$PY -u features_inbound.py
$PY -u features_ord.py
bash pf_adsb.sh $W
