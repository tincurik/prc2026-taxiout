#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Stage 4: the prediction file.  Usage: bash pipeline/40_final_file.sh [chain output parquet]
# The chain output: the shipped chain's is stage 3d's outputs/overnight_final/test/ar_134naclh_tww.parquet (BND_NB=1 39_bnd1.sh, then 39_post_steps.sh), which gives the prediction file
# outputs/overnight_final/test/ar_134naclh_tww_wit_zf3.parquet; the default below is stage 3c's outputs/overnight_final/test/ar_133_prc2.parquet (the chain before the neighbour-label head
# and the four post-steps), and the stage-3b file outputs/overnight_final/test/ar_132_prc2.parquet builds that chain's file instead.
# The plan-window post-step (scripts/witness_post.py, ideas zf3 = the zero-floor push for LIRF and the plan-window clip) is applied to the chain output; the file is then checked
# against final_submitting.parquet (two columns in order, the template's ids in its order, finite non-negative seconds) and, for a recorded chain, its recipe chain (scripts/check_final.py).
# The post-step reads data/raw and the 2025 labels; it never reads a label of a test month.
# A replayed chain file differs from the recorded one in the last bits: CatBoost on a GPU is not bit-reproducible (README.md, "Reproducibility").
set -e
cd "$(dirname "$0")/.."
export EC_PHASE=final
PY=${PY:-python}
SRC=${1:-outputs/overnight_final/test/ar_133_prc2.parquet}
$PY scripts/witness_post.py --source "$SRC" --ideas zf3
OUT=${SRC%.parquet}_wit_zf3.parquet
ID=$(basename "$OUT" .parquet)
if [ -f recipes/$ID.json ] || [ -f recipes/other_chains/$ID.json ]; then
  $PY scripts/check_final.py "$OUT" --id "$ID"   # template and recipe chain
else   # an intermediate chain of this tree (stage 3b / 3c) has no recorded recipe in recipes/: the template check only
  echo "40_final_file: no recipe for $ID in recipes/ (not a recorded chain): template check only"
  $PY scripts/check_final.py "$OUT"
fi
echo "prediction file: $OUT"
