#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Stage 3a: the stream-transformer pieces of the bundle chain (39_bnd1.sh): two members, each its own switch, each producing the feature file(s) the m12 stackers
# (35_m12_stackers.sh, M12_EXTRA) read.  Both are test-side refits on the recorded 2025 OOF, like 50_final_phase.sh's members stage; no 2026 label exists.
#   kdu   ts_kdu3 = the stream transformer overnight/stx.py (every movement of an airport is a token; attention over 512-token chunks) with ADS-B modality dropout and
#         privileged self-distillation (teacher = the EMA copy that sees the full chunk; also on the label-free months' tracked tokens), 3 seeds, speed flags.
#         refit on all 2025 with the recorded 23 epochs -> outputs/overnight_<phase>/test/ts_kdu3_raw.parquet -> features_dep_ts_kdu_f (stxf_reg, stxf_reg_gap,
#         stxf_sig, stxf_p1800, stxf_p3600, features_stx.py).  Recipe: recipes/ts_kdu3.json.  Fixed seeds: the columns are not part of the seed bag.
#   ctx   stx_ctx = the same transformer whose departure tokens carry the label-free departure-stream windows (features_dep_ctxm = features_dep_ctx without the
#         16 columns that hold the base chain's predictions, ctx_member.py) -> test refit (recorded 14 epochs) -> member-disagreement features with stx_ctx added to
#         the 11 regressors (features_memb.py --add stx_ctx) = features_dep_ef_10111040_stxctx_memb, which replaces memb2 in the stackers' extras.
#   nb    stx_nb = ts_kdu3's recipe plus the neighbour-label head (--nbaux 0.3 --nb_win 1800: each loss-row departure also regresses the shrunk leave-one-out level of
#         its airport's and its runway's clipped residuals within +-30 min; built per fit from that fit's own training months, so no validation or 2026 label enters)
#         -> test refit (recorded 23 epochs) -> features_dep_stxnb_f (stxf_reg, stxf_reg_gap, stxf_sig, stxf_p1800, stxf_p3600 + the level estimates nbm30, nbr30), which
#         replaces ts_kdu_f in the stackers' extras (39_bnd1.sh BND_NB=1).  Recipe: recipes/stx_nb.json.  Fixed seeds, like kdu.
#   oof_kdu / oof_ctx / oof_nb   the phase-independent 2025 CV (Jan/Jul folds) that the recorded OOF came from; run once on the base data (EC_PHASE unset), needs the GPU,
#         about 16 and 25 minutes.  The phase stages below copy the recorded OOF and results json from SEED_ROOT (default: the repository root) when
#         the phase directory lacks them; a missing recorded OOF stops the stage and names these two modes.
# The phase is EC_PHASE (final, or pseudo2..pseudo6 for the rehearsals whose pairs the stackers train on; the base data when unset).
#   EC_PHASE=final bash pipeline/38_stx_members.sh kdu|ctx|nb [dry]
# GPU (torch); about 15-17 min per member on one RTX 5070 Ti, 4 CPU threads, 14 GB RAM.  `dry` prints the commands and checks the inputs.
set -e
cd "$(dirname "$0")/../overnight"
mkdir -p ../logs
PY=${PY:-python}
PH=${EC_PHASE:-}
MODE=$1; DRY=${2:-}
SEED_ROOT=${SEED_ROOT:-$(cd .. && pwd)}
EX=ext_metar,ext_prc,ext_osm,ext_opdiev,ext_ord,adsb2fullrel
V3="--tdir 1 --rope 1 --mtm 0.15 --mtm_w 0.3 --adrop 0.4 --aux_w 0.25 --sig 1 --tail 1800,3600 --ema 0.999"
KDU="$V3 --kd_w 1 --kd_unl 1 --nosync 1 --fused 1 --compile 1"
NB="--nbaux 0.3 --nb_win 1800"
SFX=${PH:+_$PH}
D=../data/processed$SFX; O=../outputs/overnight$SFX; R=../reports/overnight$SFX
run() { echo "+ $*"; [ "$DRY" = dry ] || "$@"; }
need() { [ -f "$1" ] || { echo "38_stx_members: missing $1 ($2)"; [ "$DRY" = dry ] || exit 1; }; }
seed() {   # seed <id>: the recorded 2025 results json and OOF of a member into this phase's directories
  for pair in "$O/oof/$1.parquet:$SEED_ROOT/outputs/overnight/oof/$1.parquet" "$R/results/$1.json:$SEED_ROOT/reports/overnight/results/$1.json"; do
    [ -f "${pair%%:*}" ] || [ -f "${pair#*:}" ] || { echo "38_stx_members: the recorded 2025 file ${pair#*:} does not exist: it comes from the phase-independent CV of $1. Run 'bash pipeline/39_bnd1.sh base' once first with EC_PHASE unset (it runs 38_stx_members.sh oof_kdu and oof_ctx), or point SEED_ROOT at the tree that holds it." >&2; [ "$DRY" = dry ] || exit 1; }
  done
  [ -f $R/results/$1.json ] || { [ "$DRY" = dry ] && echo "+ cp $SEED_ROOT/reports/overnight/results/$1.json $R/results/" || { mkdir -p $R/results; cp $SEED_ROOT/reports/overnight/results/$1.json $R/results/; }; }
  [ -f $O/oof/$1.parquet ] || { [ "$DRY" = dry ] && echo "+ cp $SEED_ROOT/outputs/overnight/oof/$1.parquet $O/oof/" || { mkdir -p $O/oof; cp $SEED_ROOT/outputs/overnight/oof/$1.parquet $O/oof/; }; }
}
case $MODE in
  kdu)
    seed ts_kdu3
    [ -f $O/test/ts_kdu3_raw.parquet ] || run $PY -u stx.py --id ts_kdu3 --extra $EX --ctx 1 --epochs 24 --seeds 3 --refit_only 1 $KDU
    [ -f $D/features_dep_ts_kdu_f.parquet ] || run $PY -u features_stx.py --id ts_kdu3 --out ts_kdu_f --cols sig,p1800,p3600 ;;
  ctx)
    seed stx_ctx
    run $PY -u ctx_member.py
    [ -f $O/test/stx_ctx_raw.parquet ] || run $PY -u stx.py --id stx_ctx --extra $EX,ctxm --ctx 1 --epochs 24 --seeds 3 --refit_only 1
    [ -f $D/features_dep_ef_10111040_stxctx_memb.parquet ] || run $PY -u features_memb.py --all 1 --add stx_ctx --out ef_10111040_stxctx_memb ;;
  nb)
    seed stx_nb
    [ -f $O/test/stx_nb_raw.parquet ] || run $PY -u stx.py --id stx_nb --extra $EX --ctx 1 --epochs 24 --seeds 3 --refit_only 1 $KDU $NB
    [ -f $D/features_dep_stxnb_f.parquet ] || run $PY -u features_stx.py --id stx_nb --out stxnb_f --cols sig,p1800,p3600,nbm30,nbr30 ;;
  oof_kdu)
    [ -z "$PH" ] || { echo "38_stx_members oof_kdu: phase independent, run with EC_PHASE unset"; exit 2; }
    run $PY -u stx.py --id ts_kdu3 --extra $EX --ctx 1 --epochs 24 --seeds 3 --final 0 --mask_val final $KDU ;;
  oof_nb)
    [ -z "$PH" ] || { echo "38_stx_members oof_nb: phase independent, run with EC_PHASE unset"; exit 2; }
    run $PY -u stx.py --id stx_nb --extra $EX --ctx 1 --epochs 24 --seeds 3 --final 0 --mask_val final $KDU $NB ;;
  oof_ctx)
    [ -z "$PH" ] || { echo "38_stx_members oof_ctx: phase independent, run with EC_PHASE unset"; exit 2; }
    run $PY -u ctx_member.py
    run $PY -u stx.py --id stx_ctx --extra $EX,ctxm --ctx 1 --epochs 24 --seeds 3 --final 1 ;;
  *) echo "usage: 38_stx_members.sh kdu|ctx|nb|oof_kdu|oof_ctx|oof_nb [dry]"; exit 2 ;;
esac
echo "38_stx_members $MODE ${PH:-base} done"
