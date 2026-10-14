#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Stage 3c: the bundle chain ar_133 / ar_134na = the month-complete stacker family of stage 3b with more inputs, each a switch that can be turned off:
#   BND_CTX=1  stx_ctx member in the member-disagreement features: features_dep_ef_10111040_stxctx_memb replaces memb2   (38_stx_members.sh ctx)
#   BND_KDU=1  ts_kdu3 columns stxf_reg / stxf_reg_gap / stxf_sig / stxf_p1800 / stxf_p3600: features_dep_ts_kdu_f        (38_stx_members.sh kdu)
#   BND_NB=1   the final chain (ids ar_134na; default off = ar_133): stx_nb's columns (the same five + nbm30 / nbr30, the neighbour-label head) features_dep_stxnb_f replace ts_kdu_f
#              (BND_KDU then defaults to 0)                                                                                                  (38_stx_members.sh nb)
# The stackers' extras are adsbdir4,adsbrev,<memb2 | ef_10111040_stxctx_memb>,ctx[,ts_kdu_f | ,stxnb_f]; the switches BND_CTX and BND_KDU at 0 = the extras of ar_132.
# Every extra must exist in the final phase and in all five rehearsal phases: the stackers train on each rehearsal's own features.  The members' columns are
# built once with fixed seeds (stx.py --seeds 3 --seed0 0, the three runs averaged), so every stacker seed of the bag reads the same columns.  The airport-state features are not part of it.
# Order (after stages 0-3 and the rehearsal phases' own chains of stage 3b / 50_final_phase.sh, and the final phase's chain, which builds its ctx features):
#   bash pipeline/39_bnd1.sh base                                    once, EC_PHASE unset: the recorded 2025 out-of-fold files of the members
#                                                                    (38_stx_members.sh oof_kdu, oof_ctx, oof_nb; GPU, about 16 to 25 min each)
#   EC_PHASE=pseudo2 bash pipeline/39_bnd1.sh members                38_stx_members.sh kdu and ctx on the phase (GPU, about 15-17 min each); then pseudo3 ... pseudo6, final
#   EC_PHASE=final bash pipeline/39_bnd1.sh chain                    the four stackers, their average and the tail -> outputs/overnight_final/test/ar_133_prc2.parquet
# The final chain (stage 3d) sets BND_NB=1 for all of the above and also runs `chain` in each rehearsal phase, because its post-steps fit on those copies:
#   BND_NB=1 EC_PHASE=pseudo2 bash pipeline/39_bnd1.sh chain        then pseudo3 ... pseudo6 (-> outputs/overnight_pseudoN/test/ar_134na_dup.parquet)
#   BND_NB=1 EC_PHASE=final bash pipeline/39_bnd1.sh chain          -> outputs/overnight_final/test/ar_134na_dup.parquet
#     (with BND_NB=1 the tail stops at <id>_dup: TAIL_SKIP defaults to lplan,prc25,prc2, so the monthly runway calibration is not run)
#   bash pipeline/39_post_steps.sh pools                             the copy-direction re-mix in pseudo2..6 (ar_134naclh_rmx)
#   bash pipeline/39_post_steps.sh run                               the four post-steps (-> outputs/overnight_final/test/ar_134naclh_tww.parquet)
#   bash pipeline/40_final_file.sh outputs/overnight_final/test/ar_134naclh_tww.parquet                    the plan-window post-step and the file check
# `chain` runs the stackers one after the other on the CPU (THREADS threads each, default 10); the 35_m12_stackers.sh letters (x c l w) can be run in parallel by hand
# with the same M12_ID / M12_EXTRA, then `35_m12_stackers.sh avg` and `bash overnight/tail_chain.sh ar_133 ar_133_avg4`.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
BND_CTX=${BND_CTX:-1}; BND_NB=${BND_NB:-0}; BND_KDU=${BND_KDU:-$([ "$BND_NB" = 1 ] && echo 0 || echo 1)}
DEF_ID=ar_133; [ "$BND_NB" = 1 ] && DEF_ID=ar_134na
export M12_ID=${M12_ID:-$DEF_ID} TAIL_ID=${TAIL_ID:-$DEF_ID}
MEMB=memb2; [ "$BND_CTX" = 1 ] && MEMB=ef_10111040_stxctx_memb
export M12_EXTRA=adsbdir4,adsbrev,$MEMB,ctx; [ "$BND_KDU" = 1 ] && M12_EXTRA=$M12_EXTRA,ts_kdu_f
[ "$BND_NB" = 1 ] && M12_EXTRA=$M12_EXTRA,stxnb_f
case ${1:-} in
  base)
    [ -z "${EC_PHASE:-}" ] || { echo "39_bnd1 base: phase independent, run with EC_PHASE unset"; exit 2; }
    [ "$BND_KDU" = 1 ] && bash pipeline/38_stx_members.sh oof_kdu
    [ "$BND_NB" = 1 ] && bash pipeline/38_stx_members.sh oof_nb
    [ "$BND_CTX" = 1 ] && (cd overnight && $PY -u features_ctx.py --base rc_v34_more18 --out ctx)   # the 2025 side's departure-stream context (the phases build theirs in stage 3b / 4)
    [ "$BND_CTX" = 1 ] && bash pipeline/38_stx_members.sh oof_ctx
    echo "39_bnd1 base done" ;;
  members)
    : "${EC_PHASE:?set EC_PHASE=pseudo2..pseudo6 or final}"
    [ "$BND_KDU" = 1 ] && bash pipeline/38_stx_members.sh kdu
    [ "$BND_CTX" = 1 ] && bash pipeline/38_stx_members.sh ctx
    [ "$BND_NB" = 1 ] && bash pipeline/38_stx_members.sh nb
    echo "39_bnd1 members done ($EC_PHASE)" ;;
  chain)
    export EC_PHASE=${EC_PHASE:-final}
    for e in ${M12_EXTRA//,/ }; do
      for ph in pseudo2 pseudo3 pseudo4 pseudo5 pseudo6 $EC_PHASE; do
        [ -f data/processed_$ph/features_dep_$e.parquet ] || { echo "39_bnd1: data/processed_$ph/features_dep_$e.parquet is missing (run the stage that makes it in that phase)"; exit 1; }
      done
    done
    bash pipeline/35_m12_stackers.sh x c l w avg
    # the final chain (BND_NB=1) ends its tail at <id>_dup: the monthly runway calibration (prc2, side branch prc25) is not part of it
    [ "$BND_NB" = 1 ] && export TAIL_SKIP=${TAIL_SKIP-lplan,prc25,prc2}
    (cd overnight && mkdir -p ../logs && bash tail_chain.sh $TAIL_ID ${M12_ID}_avg4)
    L=prc2; [ "$BND_NB" = 1 ] && L=dup
    echo "39_bnd1 chain done: outputs/overnight_$EC_PHASE/test/${TAIL_ID}_$L.parquet" ;;
  env)   # for a seed-bag draw (pipeline/45_bag_draw.sh): the ids and stacker inputs of this switch setting, as exports
    echo "export M12_ID=$M12_ID TAIL_ID=$TAIL_ID M12_EXTRA=$M12_EXTRA" ;;
  *) echo "usage: 39_bnd1.sh base | members | chain | env   (see the header)"; exit 2 ;;
esac
