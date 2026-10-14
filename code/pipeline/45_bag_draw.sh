#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Stage 5a: one extra draw of the seed bag (draw k = 2, 3, ...): the whole final chain refitted with a new seed offset, in its own phase.
#   bash pipeline/45_bag_draw.sh <k>        phase finalb<k>, BAG_SEED = 100 * k;  PLAN=1 prints the plan and touches nothing
# Draw 1 is the final phase itself (`make final`, seeds as recorded: BAG_SEED unset = 0).  Draw k refits every test-side member (overnight/final.py, dae.py, mlp.py: seeds of the
# members), the delayed-row gate's five seeds (gate2.py) and the stackers' final fits (adsb_stack.py) at the offset BAG_SEED, then runs the final chain on them and
# its four post-steps (ar_134naclh_tww).  What stays fixed across draws: the 2025 out-of-fold side and the stream-transformer columns (stx_ctx, stx_nb: stx.py has no seed offset).
# The phase starts as a copy of the final phase's features, out-of-fold files and caches (no member test file, no gate2 q cache), so it needs stages 3b-3d finished in
# the final phase and in pseudo2 ... pseudo6 (the post-steps' coefficient pools read the rehearsal phases, as in `make final`).  It is resumable: a member whose
# test file exists is skipped.  One CUDA GPU, 10 threads; about 75 min of members, about 5 h of stackers (four, one after the other), a few minutes of post-steps.
# Two draws can run side by side on two cards (CUDA_VISIBLE_DEVICES=0 / 1).  Then: bash pipeline/46_bag_file.sh <chain id> <K>.
set -e
K=${1:?usage: bash pipeline/45_bag_draw.sh <k>   (k = 2, 3, ...)}
[ "$K" -ge 2 ] || { echo "45_bag_draw: draw 1 is the final phase; k must be 2 or more"; exit 2; }
cd "$(dirname "$0")/.."
PY=${PY:-python}
export EC_PHASE=finalb$K BAG_SEED=$((100 * K)) THREADS=${THREADS:-10} GATE2_SEEDS=${GATE2_SEEDS:-5}
SRC=final; P=finalb$K
T=outputs/overnight_$P/test
EXT=ext_metar,ext_prc,ext_osm,ext_opdiev,ext_ord
S0=$((BAG_SEED / 10))   # the mlp members' first seed: the recorded fits use 100 + seed0 + s
MEMBERS="mix_v18_fs3_deep mix_v8_cat_it16k mix_v32_cat_d10_opdiev mix_v34_cat_lr15_ord mix_v37_cat_lobt_ord mix_v36_cat_adsbdec mix_v39_cat_adsb_lobt mix_v36s2_cat_adsbdec"
if [ -n "${PLAN:-}" ]; then
  echo "bag draw $K: phase $P, BAG_SEED=$BAG_SEED, copy of phase $SRC"
  for id in $MEMBERS; do echo "  final.py --from $id --scale 1.15"; done
  echo "  dae.py dae_v6_fs3_h2048_sw25; mlp.py mlp_v1_fs3, mlp_v2_adsb, mlp_v4_lobt_adsb (--seed0 $S0)"
  echo "  pipeline/30_chain.sh with the stage-3d ids (BND_NB=1: M12_ID, TAIL_ID, M12_EXTRA from 39_bnd1.sh env), tail without the runway calibration"
  echo "  pipeline/39_post_steps.sh run  ->  $T/ar_134naclh_tww.parquet"
  exit 0
fi
# the phase: features, out-of-fold files and caches of the final phase
[ -d data/processed_$P ] || cp -r data/processed_$SRC data/processed_$P
if [ ! -d outputs/overnight_$P/oof ]; then
  mkdir -p outputs/overnight_$P/test
  cp -r outputs/overnight_$SRC/oof outputs/overnight_$P/oof
  cp -r outputs/overnight_$SRC/cache outputs/overnight_$P/cache
  rm -f outputs/overnight_$P/cache/gate2_v1_q.parquet
  cp outputs/overnight_$SRC/*.pt outputs/overnight_$P/
fi
[ -d reports/overnight_$P/results ] || { mkdir -p reports/overnight_$P; cp -r reports/overnight_$SRC/results reports/overnight_$P/results; }
echo "[$(date +%T)] phase $P, BAG_SEED=$BAG_SEED"
cd overnight
for id in $MEMBERS; do
  [ -f ../$T/$id.parquet ] || { echo "[$(date +%T)] member $id"; $PY -u final.py --from $id --scale 1.15; }
done
[ -f ../$T/dae_v6_fs3_h2048_sw25_raw.parquet ] || $PY -u dae.py --id dae_v6_fs3_h2048_sw25 --fs3 1 --hidden 2048 --swap 0.25 --dae_epochs 60 --head_epochs 15 --clip 5000 --final 1 --refit_only 1
[ -f ../$T/mlp_v1_fs3_raw.parquet ] || $PY -u mlp.py --id mlp_v1_fs3 --fs3 1 --seeds 2 --epochs 12 --final 1 --refit_only 1 --seed0 $S0
[ -f ../$T/mlp_v2_adsb_raw.parquet ] || $PY -u mlp.py --id mlp_v2_adsb --fs3 1 --seeds 3 --epochs 12 --final 1 --extra $EXT,adsb2fullrel --refit_only 1 --seed0 $S0
[ -f ../$T/mlp_v4_lobt_adsb_raw.parquet ] || $PY -u mlp.py --id mlp_v4_lobt_adsb --anchor lobt --fs3 1 --seeds 2 --epochs 12 --final 1 --extra adsb2fullrel --refit_only 1 --seed0 $S0
echo "[$(date +%T)] members done; chain"
cd ..
# the final chain's ids and stacker inputs (BND_NB=1), the tail stopping at <id>_dup as in `make final`
eval "$(BND_NB=1 bash pipeline/39_bnd1.sh env)"
export TAIL_SKIP=${TAIL_SKIP-lplan,prc25,prc2}
bash pipeline/30_chain.sh
bash pipeline/39_post_steps.sh run
echo "[$(date +%T)] draw $K done: $T/${TAIL_ID}clh_tww.parquet"
