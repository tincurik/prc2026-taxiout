#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Stage 3b: the month-complete (m12) stacker family of the final chain ar_131 (overnight/stack12.py; on the twelve held-out months:
# -0.712 RMSE [-0.863, -0.560], 11/12 months better, -0.717 without the 20 largest rows).  The four stackers of 30_chain.sh (xgb, cat, lgb2, covw;
# 3 seeds, extras adsbdir4,adsbrev,memb2,ctx, matched + unmatched branches, LIRF uncorrected, clip 1800, train_ymax 20000) are trained on this
# phase's Jan/Jul OOF rows PLUS every honest out-of-month 2025 pair: the held-out pair of each pseudo phase (pseudo2 Feb+Jun, pseudo3 Mar+Sep,
# pseudo4 Apr+Oct, pseudo5 May+Nov, pseudo6 Aug+Dec; 1.74 M rows in all), each with its own phase's features (memb2 gaps, ctx neighbours' base
# predictions) and its own base prediction (that phase's rc_v34_more18 test file).  Output <id>_<x|c|l|w>_m12 in outputs/overnight_<EC_PHASE>/{oof,test}, then
# <TAG>_avg4 = their mean.
# WHERE THE PAIRS COME FROM: PAIRS_ROOT = the tree holding the pseudo phases (data/pseudoN/truth.parquet, data/processed_pseudoN/, outputs/overnight_pseudoN/
# {oof,test}/rc_v34_more18.parquet, features memb2 + ctx).  They are the leave-two-months-out rehearsals: `EC_PHASE=pseudoN bash pipeline/50_final_phase.sh data seed features members chain`
# (months of each pair: pipeline/50_final_phase.sh), all inside this tree, so PAIRS_ROOT defaults to the repository root.  Only the pairs'
# held-out rows and features are read; no label of a test month exists anywhere.
#   EC_PHASE=final bash pipeline/35_m12_stackers.sh x|c|l|w|avg      one stacker, or the average (waits for the four)
# M12_ID (default ar_131) names the stackers <M12_ID>_<x|c|l|w>_m12 and their average; M12_EXTRA swaps the extras (every named features_dep_<name>.parquet must exist in the final slot
# and in all five pseudo phases: stack12 reads each pair's own phase).  The bundle chain (stx_ctx member + ts_kdu3 columns): pipeline/39_bnd1.sh.
# Device (12-10-2026): STACK_GPU=1 (the default when a card is visible) = xgb/covw on the card (CUDA) and CatBoost on the card, LightGBM's OpenCL learner
# (STACK_LGB_GPU; 6.1 min a fold on 5 threads against 10-14 on 10 CPU threads): the recipe the stackers were validated with; a device offset of
# about 0.005 RMSE, inside the stackers' seed noise.  STACK_GPU=0 (or no card) = CPU only: xgb/lgb/covw with 10 threads, CatBoost on the CPU (STACK_CAT_CPU,
# as 30_chain.sh), the recipe ar_131 was first built with; two stackers can run side by side on a 20-thread box.
set -e
cd "$(dirname "$0")/../overnight"
mkdir -p ../logs
PY=${PY:-python}
N=${THREADS:-10}
ID=${M12_ID:-ar_131}
export PAIRS_ROOT=${PAIRS_ROOT:-$(cd .. && pwd)}
[ -d "$PAIRS_ROOT/data/pseudo3" ] || { echo "35_m12_stackers: no pseudo phases under PAIRS_ROOT=$PAIRS_ROOT"; exit 1; }
E="--extra ${M12_EXTRA:-adsbdir4,adsbrev,memb2,ctx}"   # M12_EXTRA: the stackers' feature files (default = ar_131's); pipeline/39_bnd1.sh sets it for the bundle chain ar_133
A="--arms m12 --phases ${EC_PHASE:-final} --seeds 3 --feat ${M12_FEAT:-adsb2m}"   # final, or a rehearsal phase pseudo2..6 (then also its Jan/Jul OOF, which its tail needs)
case ${EC_PHASE:-final} in pseudo*) A="$A --phase_oof 1";; esac
GPU=${STACK_GPU:-auto}
[ "$GPU" = auto ] && { [ "${CUDA_VISIBLE_DEVICES-x}" != "" ] && nvidia-smi -L >/dev/null 2>&1 && GPU=1 || GPU=0; }
if [ "$GPU" = 1 ]; then DEV=cuda; CATCPU=""; export STACK_LGB_GPU=1; else DEV=cpu; CATCPU=$N; export STACK_LGB_GPU=0; fi
echo "35_m12_stackers: STACK_GPU=$GPU"
P="{\"max_depth\":7,\"device\":\"$DEV\",\"nthread\":$N}"
for w in "$@"; do case $w in
  x) $PY -u stack12.py --id ${ID}_x $A $E --params "$P" ;;
  c) STACK_CAT_CPU=$CATCPU $PY -u stack12.py --id ${ID}_c $A $E --model cat ;;
  l) $PY -u stack12.py --id ${ID}_l $A $E --model lgb --params "{\"num_threads\":$N,\"extra_trees\":true,\"num_leaves\":127,\"min_data_in_leaf\":30}" ;;
  w) $PY -u stack12.py --id ${ID}_w $A $E --covw 1 --params "$P" ;;
  avg) O=../outputs/overnight_${EC_PHASE:-final}
       for i in $(seq 1 480); do   # up to 8 h for the other stackers of the family
         ok=1; for s in x c l w; do [ -f $O/test/${ID}_${s}_m12.parquet ] && [ -f $O/oof/${ID}_${s}_m12.parquet ] || ok=0; done
         [ $ok = 1 ] && break; sleep 60
       done
       [ $ok = 1 ] || { echo "35_m12_stackers: the four stackers are not all there"; exit 1; }
       $PY -u avg_multi.py --id ${ID}_avg4 --ids ${ID}_x_m12,${ID}_c_m12,${ID}_l_m12,${ID}_w_m12 ;;
  *) echo "usage: 35_m12_stackers.sh x|c|l|w|avg"; exit 2 ;;
esac; done
