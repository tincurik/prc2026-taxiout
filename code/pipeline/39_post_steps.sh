#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Stage 3d: the four tail post-steps of the final chain, applied to the bundle's tail output <TAIL_ID>_dup (39_bnd1.sh with BND_NB=1; TAIL_ID default ar_134na;
# the final chain's tail stops at _dup, without the monthly runway calibration).
# Each is a switch that can be turned off (the next step then reads the last kept output); every one writes oof (Jan/Jul 2025) and test files in
# outputs/overnight_<phase>/ and a results json.
#   POST_RMX=1   r16_remix.py    LIRF copy-direction re-mix: the chain's implied copy weight p_eff = (pred - R) / (C - R) between the rules-chain value R and the plan copy C = d_sched
#                                is refitted per d_sched band x plan/no-plan (affine, G^2-weighted, clipped to [0, 1]) on LIRF rows with |C - R| > 600, d_sched < 20000, 0.02 < p_eff < 0.98.
#                                -> <TAIL_ID>clh_rmx
#   POST_RWNB=1  r17_rwnb.py     runway-neighbour level corrector (--model lgb): the shrunk mean over the other departures of the runway (+-300 / 900 / 1800 s) and the airport
#                                (+-900 s) of label-free clock gaps (NM-AOBT, ADS-B seen-to-take-off, rules-chain value, EOBT) and of the neighbours' predictions, 23 columns +
#                                the airport -> a LightGBM fit of clip(pred - y, +-1800); applied to every row.  Refit on the pool of <TAIL_ID>clh_rmx.  -> <TAIL_ID>clh_rwnb
#   POST_HGF=1   hg_far50.py     LIRF no-plan rows with C > 50000: the gate's copy probability, binned in two (p_imp >= .5 / < .5, leave-folds-out copy share, Jeffreys smoothing),
#                                mixes the copy and the +1-day record 87500: pred' = p*C + (1-p)*87500 where |pred - C| > 600 (--calib bin2).  Pool: the bundle <TAIL_ID>_dup.  -> <TAIL_ID>clh_hgf
#   POST_TWW=1   twin_window.py  LIRF departures without an NM flight read through the arrival twin at the destination challenge airport: the copy cell goes to the plan copy, the
#                                normal cell to the rules-chain value.  No fitted parameter (q = 1).  -> <TAIL_ID>clh_tww
#
# Coefficient pools: the final phase's Jan/Jul OOF (this tree) + the held-out pairs of pseudo2..6 (PAIRS_ROOT, default this tree, as 35_m12_stackers.sh); each
# row's own month folds are left out and the 2026 rows use all six folds.  The pools need two ids in every rehearsal phase, both built by the commands below:
#   <TAIL_ID>_dup     the same bundle chain on the rehearsal phase:  BND_NB=1 EC_PHASE=pseudoN bash pipeline/39_bnd1.sh chain   (N = 2..6)
#   <TAIL_ID>clh_rmx  its copy-direction re-mix:                     bash pipeline/39_post_steps.sh pools                      (runs r16_remix.py in pseudo2..6)
# (The recorded run built the same two files under the names bndna_dup and r16rmx2na; BUNDLE_PAIRS / RMX_PAIRS point at other names if needed.
#  The step ids carry "clh" because the chain is the ar_134na chain with the monthly runway calibration cut out.)
# No 2026 label is read; the steps' results jsons carry the id chain (base links).  CPU only, a few minutes in all
# (the pool is built once and cached in outputs/r16_remix/ and outputs/r17_rwnb/); memory: peak about 6 GB (r17_rwnb), so run it with a cap (POST_SCOPE=1 wraps each step in
# systemd-run --scope -p MemoryMax=${POST_MEM:-8G}).
# Order, after stage 3c with BND_NB=1 (members in every phase; `chain` in pseudo2..pseudo6 and final):
#   bash pipeline/39_post_steps.sh plan       stage list + input check (touches nothing)
#   bash pipeline/39_post_steps.sh pools      <TAIL_ID>clh_rmx in pseudo2..6 (the re-mix's own pool reads <TAIL_ID>_dup of the final phase and of the other rehearsals)
#   bash pipeline/39_post_steps.sh run        the enabled steps in order, in the final phase
#   REF_ROOT=<tree> bash pipeline/39_post_steps.sh parity   each step's output against the same id in another tree's final phase (rms, max, rows moved)
# Then: bash pipeline/40_final_file.sh outputs/overnight_final/test/<TAIL_ID>clh_tww.parquet <TAIL_ID>clh_tww
set -e
cd "$(dirname "$0")/.."
export EC_PHASE=${EC_PHASE:-final}
PH=$EC_PHASE
PY=${PY:-python}
export PAIRS_ROOT=${PAIRS_ROOT:-$(pwd)}
T=${TAIL_ID:-ar_134na}
POST_RMX=${POST_RMX:-1}; POST_RWNB=${POST_RWNB:-1}; POST_HGF=${POST_HGF:-1}; POST_TWW=${POST_TWW:-1}
RMX=${T}clh_rmx; RWNB=${T}clh_rwnb; HGF=${T}clh_hgf; TWW=${T}clh_tww
BUNDLE_PAIRS=${BUNDLE_PAIRS:-${T}_dup}; RMX_PAIRS=${RMX_PAIRS:-$RMX}
BASE=${T}_dup
MODE=${1:-plan}
# the steps, as "<id>|<args>" lines in order; CUR = the last kept id, CURP = its id in the pseudo phases (the pool of rwnb is built on the previous step's outputs)
steps() {
  CUR=$BASE; CURP=$BUNDLE_PAIRS; S=()
  if [ "$POST_RMX" = 1 ]; then S+=("r16_remix.py|--id $RMX --base $CUR --fit_base $BASE --pairs_base $BUNDLE_PAIRS"); CUR=$RMX; CURP=$RMX_PAIRS; fi
  if [ "$POST_RWNB" = 1 ]; then S+=("r17_rwnb.py|--id $RWNB --base $CUR --fit_base $CUR --pairs_base $CURP --model lgb"); CUR=$RWNB; fi
  if [ "$POST_HGF" = 1 ]; then S+=("hg_far50.py|--id $HGF --base $CUR --calib bin2 --fit_base $BASE --pairs_base $BUNDLE_PAIRS"); CUR=$HGF; fi
  if [ "$POST_TWW" = 1 ]; then S+=("twin_window.py|--id $TWW --base $CUR"); CUR=$TWW; fi
}
steps
case $MODE in
  plan)
    echo "post-steps plan: phase $PH, tail $BASE -> $CUR, pairs from $PAIRS_ROOT (bundle $BUNDLE_PAIRS, re-mix $RMX_PAIRS)"
    n=1; for s in "${S[@]}"; do echo "  $n  python ${s%%|*} ${s#*|}"; n=$((n+1)); done
    miss=0
    chk() { if [ -f "$1" ]; then echo "  ok       $1"; else echo "  MISSING  $1  ($2)"; miss=$((miss+1)); fi; }
    echo "inputs of the final phase ($PH):"
    for f in outputs/overnight_$PH/test/$BASE.parquet outputs/overnight_$PH/oof/$BASE.parquet outputs/overnight_$PH/test/rc_v34_more18_parts.parquet \
             outputs/overnight_$PH/oof/rc_v34_more18.parquet data/processed_$PH/features_dep.parquet data/raw/final_ranking.parquet data/raw/final_submitting.parquet; do
      case $f in
        */test/$BASE.parquet|*/oof/$BASE.parquet) w="stage 3c: BND_NB=1 39_bnd1.sh chain";;
        *rc_v34_more18*) w="the rules chain of 30_chain.sh (stage 3 / 4)";;
        */features_dep.parquet) w="feature stage";;
        *) w="stage 0 data";;
      esac; chk $f "$w"
    done
    [ "$POST_RWNB" = 1 ] && chk data/processed_$PH/features_dep_${RWNB_FEAT:-adsb2m}.parquet "feature stage"
    [ "$POST_TWW" = 1 ] && chk "$(ls data/raw/training_2025-*.parquet 2>/dev/null | head -1)" "challenge data"
    echo "inputs of the pools (pseudo phases, $PAIRS_ROOT):"
    for ph in pseudo2 pseudo3 pseudo4 pseudo5 pseudo6; do
      ids="$BUNDLE_PAIRS"; [ "$POST_RMX" = 1 ] && [ "$POST_RWNB" = 1 ] && ids="$ids $RMX_PAIRS"
      for f in data/$ph/truth.parquet outputs/overnight_$ph/test/rc_v34_more18_parts.parquet data/processed_$ph/features_dep.parquet data/processed_$ph/features_dep_${RWNB_FEAT:-adsb2m}.parquet \
               $(for i in $ids; do echo outputs/overnight_$ph/test/$i.parquet; done); do
        [ -f $PAIRS_ROOT/$f ] || { echo "  MISSING  $PAIRS_ROOT/$f"; miss=$((miss+1)); }
      done
    done
    echo "plan: $miss input(s) missing ($BASE: BND_NB=1 EC_PHASE=<phase> bash pipeline/39_bnd1.sh chain in pseudo2..6 and final; $RMX_PAIRS: bash pipeline/39_post_steps.sh pools)"
    [ $miss = 0 ] ;;
  pools)
    # the re-mix in each rehearsal phase: what r17_rwnb.py's pool reads there (the bundle pool $BUNDLE_PAIRS is the rehearsal's own 39_bnd1.sh chain)
    [ "$PAIRS_ROOT" = "$(pwd)" ] || { echo "39_post_steps pools: builds the rehearsal ids in this tree; unset PAIRS_ROOT"; exit 2; }
    cd overnight
    mkdir -p ../logs
    for ph in pseudo2 pseudo3 pseudo4 pseudo5 pseudo6; do
      echo "+ EC_PHASE=$ph $PY -u r16_remix.py --id $RMX_PAIRS --base $BUNDLE_PAIRS --fit_base $BUNDLE_PAIRS"
      EC_PHASE=$ph ${POST_SCOPE:+systemd-run --user --scope -q -p MemoryMax=${POST_MEM:-8G} -p MemorySwapMax=0 nice -n 10} $PY -u r16_remix.py --id $RMX_PAIRS --base $BUNDLE_PAIRS --fit_base $BUNDLE_PAIRS \
        > ../logs/post_${ph}_${RMX_PAIRS}.log 2>&1 || { echo "39_post_steps: r16_remix.py failed in $ph (log ../logs/post_${ph}_${RMX_PAIRS}.log)"; exit 1; }
    done
    echo "39_post_steps pools done: outputs/overnight_pseudo{2..6}/test/$RMX_PAIRS.parquet" ;;
  run)
    cd overnight
    mkdir -p ../logs
    for s in "${S[@]}"; do
      script=${s%%|*}; args=${s#*|}
      echo "+ $PY -u $script $args"
      ${POST_SCOPE:+systemd-run --user --scope -q -p MemoryMax=${POST_MEM:-8G} -p MemorySwapMax=0 nice -n 10} $PY -u $script $args > ../logs/post_${PH}_$(echo $args | sed 's/--id \([^ ]*\).*/\1/').log 2>&1 \
        || { echo "39_post_steps: $script failed (log ../logs/post_${PH}_*.log)"; exit 1; }
    done
    echo "39_post_steps done: outputs/overnight_$PH/test/$CUR.parquet" ;;
  parity)
    : "${REF_ROOT:?set REF_ROOT to another tree that ran the same steps (its outputs/overnight_final/test/<id>.parquet are compared)}"
    ids=()
    [ "$POST_RMX" = 1 ] && ids+=($RMX)
    [ "$POST_RWNB" = 1 ] && ids+=($RWNB)
    [ "$POST_HGF" = 1 ] && ids+=($HGF)
    [ "$POST_TWW" = 1 ] && ids+=($TWW)
    $PY - "$PH" "$REF_ROOT" "${ids[@]}" <<'PYEOF'
import sys
import numpy as np
import pandas as pd
ph, ref_root = sys.argv[1:3]
ids = sys.argv[3:]
c = "TAXITIME_SEC_mvt"
bad = 0
for i in ids:
    got = pd.read_parquet(f"outputs/overnight_{ph}/test/{i}.parquet")
    ref = pd.read_parquet(f"{ref_root}/outputs/overnight_final/test/{i}.parquet")
    m = ref.merge(got, on="MVT_ID_mvt", suffixes=("_r", "_g"))
    d = (m[c + "_r"] - m[c + "_g"]).to_numpy(float)
    rms, mx = float(np.sqrt(np.mean(d * d))), float(np.abs(d).max())
    print(f"parity {i} vs {ref_root}: {len(m)} of {len(ref)} rows, rms diff {rms:.3f} s, max {mx:.1f} s, rows differing by > 1 s: {int((np.abs(d) > 1).sum())}")
    bad += int(len(m) != len(ref))
sys.exit(1 if bad else 0)
PYEOF
    ;;
  *) echo "usage: 39_post_steps.sh plan|pools|run|parity"; exit 2 ;;
esac
