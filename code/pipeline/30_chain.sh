#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Stage 3 / 3b, the first chain of the final phase: blend -> rules chain -> member-aware ADS-B stackers (+ departure-stream context)
# -> LIRF steps -> calibration tail (prc_cal2 on <tag>_dup, in place of the prc_cal side branch <tag>_prc25).  With M12=1 it ends in ar_132_prc2.
# The chain that ships is the later stage 3d (README.md): the same stackers with two more stream-transformer inputs (pipeline/39_bnd1.sh, ids ar_134na_*), a tail that stops at
# ar_134na_dup (no prc_cal2) and the four post-steps of pipeline/39_post_steps.sh.  The ids ar_129 .. ar_132 below are this script's own stages, not the shipped chain's.
# FEAT picks the ADS-B feature set of the stackers (default adsb2m: the extended-match ADS-B v2 events of 10_features.sh).
# EC_PHASE=final|pseudo|pseudo2 runs the test side on that phase's rows (OOF artefacts come from scripts/phase_seed.py).
# CTX=1 (default, 11-10-2026): the departure-stream context features (features_ctx.py, rebuilt per phase from that phase's own rc_v34_more18) join all four
#   stackers' --extra; ids ar_130_* and final candidate ar_130_prc2 (twelve-month check: -0.58 RMSE).  CTX=0 is the ar_129_* chain of the first q5 final.
# M12=1 (default, 11-10-2026): the four stackers are the month-complete family of pipeline/35_m12_stackers.sh (Jan/Jul OOF + the five pseudo-phase held-out pairs,
#   twelve-month check: -0.71 RMSE), stackers ar_131_*, tail without lplan ar_132_*, final candidate ar_132_prc2; needs the pseudo phases (PAIRS_ROOT, see 35_m12_stackers.sh).  M12=0 gives ar_130_prc2.
# FROM=stack starts at the stackers (features_ctx of the phase already built).
# FROM=ctx starts at the context step (blend, gate2, rules chain and memb2 of the phase already built).
set -e
cd "$(dirname "$0")/../overnight"
mkdir -p ../logs
PY=${PY:-python}
FEAT=${FEAT:-adsb2m}
N=${THREADS:-10}
CTX=${CTX:-1}; FROM=${FROM:-all}
M12=${M12:-$CTX}   # 1 (default with CTX=1): month-complete stackers (35_m12_stackers.sh) -> ar_131_*; M12=0 CTX=1 = the ar_130 chain (Jan/Jul-only stackers)
[ "$CTX" = 1 ] && TAG=ar_130 || TAG=ar_129
[ "$CTX" = 1 ] && [ "$M12" = 1 ] && TAG=ar_131
R=mix_v34_cat_lr15_ord,mix_v32_cat_d10_opdiev,mix_v8_cat_it16k,dae_v6_fs3_h2048_sw25,mix_v37_cat_lobt_ord,mix_v36_cat_adsbdec,mlp_v1_fs3,mlp_v2_adsb,mlp_v4_lobt_adsb,mix_v39_cat_adsb_lobt,mix_v36s2_cat_adsbdec
R20=mix_v31_cat_lr15_opdiev,mix_v32_cat_d10_opdiev,mix_v8_cat_it16k,dae_v6_fs3_h2048_sw25

if [ "$FROM" = all ]; then
# gate2 q cache (delayed-row copy gate, used by the rules chain) from the blend_v20 members.
if [ -z "$EC_PHASE" ]; then
  $PY -u blend.py --id blend_v20 --p_from mix_v18_fs3_deep --regs $R20
  # the clock-rule reference of the gate2 step below (clk3_*): postproc alpha 0.5, then the clock rules R1-R3, on the OOF blend and on the test blend
  $PY -u postproc.py --src blend_v20_cf --id pp_blend_v20_cf_a05 --alpha 0.5
  $PY -u clockfix.py --src pp_blend_v20_cf_a05 --id clk3_blend_v20 --lirf 1
  $PY -u final_blend.py --id final_blend_v20 --parts $R20 --weights_from blend_v20
  $PY -u clockfix.py --src pp_blend_v20_cf_a05 --id clk3_final_blend_v20 --test final_blend_v20 --lirf 1
  $PY -u gate2.py --id gate2_v1 --blend blend_v20_cf --test_raw final_blend_v20
  (cd .. && $PY scripts/build_witness_cache.py)   # arrival-witness table the rules chain reads (kept if it exists)
  # the first rules chain (the same rules on blend_v20_cf): its p_imp column marks the rows the plan-window calibrations (prc_cal*.py) treat as regression rows
  $PY -u rules_chain.py --id rc_v4_gate2_rules --blend blend_v20_cf --cliff 3600 --nocross 1 --window 3606 --sym 1 --q_cache gate2_v1 --wl 0.75 --wo 0 --test_raw final_blend_v20
  $PY -u blend.py --id blend_v48 --p_from mix_v18_fs3_deep --regs $R
else
  $PY -u gate2.py --id gate2_v1 --blend blend_v20_cf --q_only 1 --seeds ${GATE2_SEEDS:-5}   # gate2 q = mean over 5 CatBoost seeds (GPU CatBoost noise moves CV by ~0.1; GATE2_SEEDS=1 is the single-seed chain of the first final run)
fi
# Cross-fitted NNLS blend of the 11 regressors under the shared p_imp (OOF side), the same weights on the test refits.
$PY -u final_blend.py --id final_blend_v48 --parts $R --weights_from blend_v48
# Rules chain (LOBT cliff and window, gate2 LIRF q, clock rules R1-R3); --test_raw applies it to the test blend.
(cd .. && $PY scripts/build_witness_cache.py)   # arrival-witness table (kept if the phase seed already copied it)
$PY -u rules_chain.py --id rc_v34_more18 --blend blend_v48_cf --cliff 3600 --nocross 1 --window 3606 --sym 1 \
  --q_cache gate2_v1 --wl 0.75 --wo 0 --test_raw final_blend_v48

# Member-disagreement features: per member the regressor's gap to the blend regressor, plus spread (features_dep_memb2).
$PY -u features_memb.py --all 1 --out memb2

fi

# Four ADS-B residual stackers on the rules-chain output (member-aware: memb2), averaged.
S="--base rc_v34_more18 --feat $FEAT --cv day5 --seeds 3 --train_ymax 20000 --skip LIRF --unmatched 1"
P="{\"max_depth\":7,\"device\":\"cpu\",\"nthread\":$N}"
E2="--extra adsbdir4,adsbrev,memb2"
if [ "$CTX" = 1 ]; then
  # Departure-stream context: windowed delay, backlog, throughput and the neighbours' base predictions (reads this phase's rc_v34_more18 OOF + test).
  [ "$FROM" = stack ] || $PY -u features_ctx.py --base rc_v34_more18 --out ctx
  E2="$E2,ctx"
fi
if [ "$TAG" = ar_131 ]; then
  bash ../pipeline/35_m12_stackers.sh x c l w avg
elif [ "$CTX" = 1 ]; then ID_X=ar_130_xgb_ctx; ID_C=ar_130_cat_ctx; ID_L=ar_130_lgb2_ctx; ID_W=ar_130_covw_ctx
else ID_X=ar_127_xgb_memb2; ID_C=ar_128_cat_memb2; ID_L=ar_128_lgb2_memb2; ID_W=ar_128_covw_memb2; fi
if [ "$TAG" != ar_131 ]; then
$PY -u adsb_stack.py --id $ID_X $S $E2 --params "$P"
STACK_CAT_CPU=$N $PY -u adsb_stack.py --id $ID_C $S $E2 --model cat
$PY -u adsb_stack.py --id $ID_L $S $E2 --model lgb --params "{\"num_threads\":$N,\"extra_trees\":true,\"num_leaves\":127,\"min_data_in_leaf\":30}"
$PY -u adsb_stack.py --id $ID_W $S $E2 --covw 1 --params "$P"
$PY -u avg_multi.py --id ${TAG}_avg4 --ids $ID_X,$ID_C,$ID_L,$ID_W
fi

# Tail: lirf_far (ok) -> lirf_bel -> [lirf_far --plan, left out of ar_132] -> dly_cal (mul + hedge) -> dup_pair -> prc_cal 0.25 (side branch ar_129_prc25, not in the final)
# and, on the same ar_129_dup: prc_cal2 (invalid-share offset, alpha 0.25, oor 0.02) => ${TAG}_prc2, the chain's last step (the final file adds scripts/witness_post.py zf3, pipeline/40_final_file.sh).
# The tail of the m12 chain leaves out lplan (tail_chain.sh's default TAIL_SKIP=lplan: its gain changes sign between months) and writes ar_132_* from ar_131_avg4,
# so the recorded ar_131_* files (with lplan) are never overwritten.  The CTX=0 / M12=0 replays keep every step (TAIL_SKIP=), as their recorded ar_129 / ar_130 chains did.
if [ "$TAG" = ar_131 ]; then TT=${TAIL_ID:-ar_132}; else TT=$TAG; export TAIL_SKIP=${TAIL_SKIP-}; fi
[ "$TAG" = ar_131 ] && AVG=${M12_ID:-ar_131}_avg4 || AVG=${TAG}_avg4   # M12_ID: the bundle chain's stackers (pipeline/39_bnd1.sh), names its average
bash tail_chain.sh $TT $AVG
echo "test predictions: outputs/overnight${EC_PHASE:+_$EC_PHASE}/test/${TT}_prc2.parquet"
