#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Phase runs: the same pipeline on the four test months (EC_PHASE=final) and on the five rehearsals that hide two 2025 months each.
#   EC_PHASE=<phase> bash pipeline/50_final_phase.sh [stage ...]       stages (default: all, in this order): data seed features members chain file
# Phases and the months they hide (the pair is masked like the test months, every member is refitted on the other ten):
#   pseudo2 = Feb + Jun 2025, pseudo3 = Mar + Sep, pseudo4 = Apr + Oct, pseudo5 = May + Nov, pseudo6 = Aug + Dec; final = Jan, Feb, Jun, Jul 2026.
# Run order for the shipped chain: stages 0-3 once (the 2025 side), then pseudo2..6 (data seed features members chain), then final (all stages).
# Everything lives in data/processed_<phase>, reports/overnight_<phase>, outputs/overnight_<phase>.  The 2025 out-of-fold side is phase independent
# (scripts/phase_seed.py copies it from stages 1-3); only test-side work reruns.  Rehearsal phases end with their chain's score on the held-out months.
set -e
cd "$(dirname "$0")/.."
export EC_PHASE=${EC_PHASE:?set EC_PHASE=final or EC_PHASE=pseudo2..pseudo6}
PY=${PY:-python}
STAGES=${*:-data seed features members chain file}
OV=overnight

case "$EC_PHASE" in
  final) ;;
  pseudo2) PM="2025-02,2025-06";;
  pseudo3) PM="2025-03,2025-09";;
  pseudo4) PM="2025-04,2025-10";;
  pseudo5) PM="2025-05,2025-11";;
  pseudo6) PM="2025-08,2025-12";;
  *) echo "50_final_phase.sh: EC_PHASE must be final or pseudo2..pseudo6"; exit 2;;
esac

stage() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }

if stage data; then
  if [ "$EC_PHASE" = final ]; then
    $PY scripts/fetch_final.py ${FINAL_DIR:+--from-dir "$FINAL_DIR"}   # final_ranking.parquet + final_submitting.parquet (pipeline/00_data.sh fetches them too)
    $PY scripts/fetch_external.py                       # METAR / OPDI / EUROCONTROL tables for the test months (earlier copies kept in data/external/_superseded)
    $PY scripts/adsb_fetch.py --final --workers ${ADSB_WORKERS:-6}   # raw adsb.lol days of the test months (~3 GB/day streamed, resumable)
  else
    $PY scripts/make_pseudo_final.py --phase "$EC_PHASE" --months "$PM"
  fi
fi

if stage seed; then
  $PY scripts/phase_seed.py "$EC_PHASE" --oof           # 2025-only artifacts and OOF predictions copied into the phase dirs
fi

if stage features; then
  $PY $OV/prep.py                                       # movements.parquet first (split 'rank' = the phase's final_ranking rows)
  bash $OV/pf_feats1.sh
  bash $OV/pf_feats2.sh ${FEAT_WORKERS:-6}
  $PY $OV/adsb_direct.py --geo 2                        # GPU xgb, phase generic -> features_dep_adsbdir4.parquet
fi

if stage members; then
  # Test-side refits only: the OOF of every member is the 2025 result of stage 2.  Same configs as pipeline/20_members.sh.
  cd $OV
  EXT=ext_metar,ext_prc,ext_osm,ext_opdiev,ext_ord
  for id in mix_v18_fs3_deep mix_v8_cat_it16k mix_v32_cat_d10_opdiev mix_v34_cat_lr15_ord mix_v37_cat_lobt_ord \
            mix_v36_cat_adsbdec mix_v39_cat_adsb_lobt mix_v36s2_cat_adsbdec; do
    $PY -u final.py --from $id --scale 1.15
  done
  $PY -u dae.py --id dae_v6_fs3_h2048_sw25 --fs3 1 --hidden 2048 --swap 0.25 --dae_epochs 60 --head_epochs 15 --clip 5000 --final 1 --refit_only 1
  $PY -u mlp.py --id mlp_v1_fs3 --fs3 1 --seeds 2 --epochs 12 --final 1 --refit_only 1
  $PY -u mlp.py --id mlp_v2_adsb --fs3 1 --seeds 3 --epochs 12 --final 1 --extra $EXT,adsb2fullrel --refit_only 1
  $PY -u mlp.py --id mlp_v4_lobt_adsb --anchor lobt --fs3 1 --seeds 2 --epochs 12 --final 1 --extra adsb2fullrel --refit_only 1
  $PY -u final.py --from mix_v31_cat_lr15_opdiev --scale 1.15
  $PY -u gate2.py --id gate2_v1 --blend blend_v20_cf --q_only 1
  cd ..
fi

if stage chain; then
  if [ "$EC_PHASE" = final ]; then
    bash pipeline/30_chain.sh                           # CTX=1 M12=1: needs the five rehearsal phases (35_m12_stackers.sh) -> ar_132_prc2
  else
    CTX=0 bash pipeline/30_chain.sh                     # the rehearsal's own chain (Jan/Jul-only stackers) -> ar_129_prc2; its rc_v34_more18 is the pair's base prediction
    (cd $OV && $PY -u features_ctx.py --base rc_v34_more18 --out ctx)   # the departure-stream context of the pair, built on that base prediction; read by 35_m12_stackers.sh
  fi
fi

if stage file; then
  if [ "$EC_PHASE" = final ]; then
    bash pipeline/40_final_file.sh outputs/overnight_final/test/ar_132_prc2.parquet   # the chain stage's output (stage 3b); the shipped chain's file is made at stage 5
  else
    $PY scripts/validate_file.py outputs/overnight_$EC_PHASE/test/ar_129_prc2.parquet
    $PY scripts/score_phase.py --source outputs/overnight_$EC_PHASE/test/ar_129_prc2.parquet
  fi
fi
