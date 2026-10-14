#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Stage 2: base classifier P(BLOCK = SCHED) and the 11 regressors of blend_v48 (OOF on the two season folds + test refit).
# Each mix/mlp/dae run writes outputs/overnight/oof/<id>.parquet and reports/overnight/results/<id>.json;
# final.py refits on all of 2025 and writes outputs/overnight/test/<id>.parquet (iterations = fold mean x 1.15).
set -e
PY=${PY:-python}
cd "$(dirname "$0")/../overnight"
EXT=ext_metar,ext_prc,ext_osm,ext_opdiev,ext_ord
EXTA=$EXT,adsb2fullrel
CAT15='{"learning_rate":0.15,"iterations":8000}'
COMMON="--fs3 1 --p_from mix_v18_fs3_deep --reg_model cat --reg_clip 5000 --es_clip 3600"

# p_imp classifier + XGBoost regressor (recorded config: recipes/mix_v18_fs3_deep.json). Everything below reuses its p_imp.
$PY -u mix.py --id mix_v18_fs3_deep --fs3 1 --reg_target resid --reg_clip 5000 --es_clip 3600 \
  --reg_params '{"eta":0.03,"max_depth":10,"min_child_weight":50,"colsample_bytree":0.5}' --note "fs3+deep+clip5000+es3600"
$PY -u final.py --from mix_v18_fs3_deep --scale 1.15

# CatBoost regressors (recorded configs in recipes/).  Weight in blend_v48: v8 0, v32 0, v34 0 (kept for the exact reproduction).
$PY -u mix.py --id mix_v8_cat_it16k --fs3 1 --p_from mix_v18_fs3_deep --reg_model cat --reg_target resid --reg_clip 5000 \
  --es_clip 3600 --reg_params '{"iterations":16000}' --note catboost-lr0.08-uncapped-16k
$PY -u final.py --from mix_v8_cat_it16k --scale 1.15
# mix_v31 (CatBoost, lr 0.15, opdiev features, no ord) belongs to the blend_v20 members that feed the gate2 q cache (30_chain.sh).
$PY -u mix.py --id mix_v31_cat_lr15_opdiev $COMMON --reg_target resid --reg_params "$CAT15" --extra ext_metar,ext_prc,ext_osm,ext_opdiev
$PY -u final.py --from mix_v31_cat_lr15_opdiev --scale 1.15
$PY -u mix.py --id mix_v32_cat_d10_opdiev $COMMON --reg_target resid --reg_params '{"depth":10,"iterations":8000}' \
  --extra ext_metar,ext_prc,ext_osm,ext_opdiev
$PY -u final.py --from mix_v32_cat_d10_opdiev --scale 1.15
$PY -u mix.py --id mix_v34_cat_lr15_ord $COMMON --reg_target resid --reg_params "$CAT15" --extra $EXT
$PY -u final.py --from mix_v34_cat_lr15_ord --scale 1.15
$PY -u mix.py --id mix_v37_cat_lobt_ord $COMMON --reg_target resid_lobt --reg_params "$CAT15" --extra $EXT
$PY -u final.py --from mix_v37_cat_lobt_ord --scale 1.15
$PY -u mix.py --id mix_v36_cat_adsbdec $COMMON --reg_target resid --reg_params "$CAT15" --extra $EXTA
$PY -u final.py --from mix_v36_cat_adsbdec --scale 1.15
$PY -u mix.py --id mix_v39_cat_adsb_lobt $COMMON --reg_target resid_lobt --reg_params "$CAT15" --extra $EXTA
$PY -u final.py --from mix_v39_cat_adsb_lobt --scale 1.15
$PY -u mix.py --id mix_v36s2_cat_adsbdec $COMMON --reg_target resid \
  --reg_params '{"learning_rate":0.15,"iterations":8000,"random_seed":2}' --extra $EXTA
$PY -u final.py --from mix_v36s2_cat_adsbdec --scale 1.15

# Neural members
$PY -u dae.py --id dae_v6_fs3_h2048_sw25 --fs3 1 --hidden 2048 --swap 0.25 --dae_epochs 60 --head_epochs 15 --clip 5000 --final 1
$PY -u mlp.py --id mlp_v1_fs3 --fs3 1 --seeds 2 --epochs 12 --final 1
$PY -u mlp.py --id mlp_v2_adsb --fs3 1 --seeds 3 --epochs 12 --final 1 --extra $EXTA
$PY -u mlp.py --id mlp_v4_lobt_adsb --anchor lobt --fs3 1 --seeds 2 --epochs 12 --final 1 --extra adsb2fullrel
