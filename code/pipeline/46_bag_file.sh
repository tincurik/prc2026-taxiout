#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Stage 5b: the shipped prediction file = the average of K draws of the final chain, then the plan-window step.
#   bash pipeline/46_bag_file.sh [<chain id> [K]]      defaults: ar_134naclh_tww, K = 3
# Needs outputs/overnight_<phase>/test/<chain id>.parquet in the phases final (stage 3d: `make final`) and finalb2 ... finalb<K> (pipeline/45_bag_draw.sh, one per draw).
#   1. scripts/bag_build.py       the row-wise average of the K draws                              -> outputs/overnight_final/test/<id>_bag<K>.parquet (+ .bag.json)
#   2. scripts/witness_post.py    plan-window step zf3 (window 3606 s, the 2025 plan-join edge)  -> <id>_bag<K>_wit_zf3.parquet
#   3. scripts/check_final.py     the template's ids and order, finite non-negative seconds, the recipe chain of the file (recipes/<id>_bag<K>_wit_zf3.json), no hash (the file is new)
# The result is the file that ships; final/ holds the recorded one with its sha256.
set -e
cd "$(dirname "$0")/.."
export EC_PHASE=final
PY=${PY:-python}
ID=${1:-ar_134naclh_tww}
K=${2:-3}
SRC=outputs/overnight_final/test/${ID}_bag${K}.parquet
OUT=outputs/overnight_final/test/${ID}_bag${K}_wit_zf3.parquet
$PY scripts/bag_build.py --id "$ID" --k "$K"
$PY scripts/witness_post.py --source "$SRC" --ideas zf3 --win 3606
$PY scripts/check_final.py "$OUT" --id "${ID}_bag${K}_wit_zf3"
echo "prediction file: $OUT"
