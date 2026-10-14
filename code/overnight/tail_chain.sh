#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# The post-stacker tail of the honest chain, from a stacker average to the final file:
#   <tag>_lirf (lirf_far ok step) -> <tag>_fin (BEL rule) -> [<tag>_lplan (plan-row copy classifier 1200-5400), skipped by default]
#   -> <tag>_dlyh (dly_cal mul+hedge) -> <tag>_dup -> <tag>_prc25 (prc_cal 0.25, side branch) and <tag>_dup -> <tag>_prc2 (prc_cal2 invalid-share offset, the last step of the earlier chains).
#   The final chain (pipeline/39_bnd1.sh with BND_NB=1) skips prc25 and prc2 and ends at <tag>_dup.
#   bash overnight/tail_chain.sh <tag> <avg id>      log: logs/<tag>_tail.log (logs/<tag>_<phase>_tail.log under EC_PHASE)
# TAIL_SKIP = comma list of steps left out (lirf,fin,lplan,dlyh,dup,prc25,prc2; the next step takes the last kept output).
#   TAIL_SKIP by caller (each caller sets the list of the one chain it builds; the lists differ on purpose):
#     lplan              this script run by hand (the default below), and the chain of the stage-3b replay with TAG=ar_131 in pipeline/30_chain.sh
#     lplan,prc25,prc2   the shipped final chain: pipeline/39_bnd1.sh (BND_NB=1) and pipeline/45_bag_draw.sh; the tail ends at <tag>_dup
#     (empty)            the replays of the earlier chains ar_129/ar_130/ar_131 (pipeline/30_chain.sh with CTX=0 or M12=0), which kept every step
#   Default here: lplan.  The LIRF plan-row classifier gains on the two 2025 folds (-217 Jan, -212 Jul MSE), but its effect changes sign between
#   months (it loses in June 2025, where its moves no longer sort copies; between-month spread larger than its pooled gain, docs/model.md),
#   so the final leaves it out (from ar_132 on).  TAIL_SKIP= (set, empty) keeps every step, which replays the recorded ar_129/ar_130/ar_131 chains that had it.
# TAIL_CHAIN = the pre-stack rules chain (default rc_v34_more18); TAIL_ARGS_<step> = arguments appended to that step (argparse keeps the last value).
T=$1; AVG=$2
cd "$(dirname "$0")"
PY=${PY:-python}
TAIL_SKIP=${TAIL_SKIP-lplan}
L=../logs/${T}${EC_PHASE:+_$EC_PHASE}_tail.log; : > $L
C=${TAIL_CHAIN:-rc_v34_more18}
P=$AVG   # the last kept output: the next step's --base
step() {   # step <name> <script> <args...>: writes ${T}_<name> from $P
  local s=$1 x=$2 v=TAIL_ARGS_$1; shift 2
  case ",$TAIL_SKIP," in *",$s,"*) echo "[tail] skip $s" >> $L; return 0;; esac
  $PY -u $x --id ${T}_$s --base $P "$@" ${!v} >> $L 2>&1 || exit 1
  P=${T}_$s
}
step lirf lirf_far.py --chain $C --arrmo 1 --w 0.5 --adsb ok
step fin lirf_bel.py --dmin 0
step lplan lirf_far.py --chain $C --plan 1 --lo 1200 --hi 5400
step dlyh dly_cal.py --mode mul --hedge 1
step dup dup_pair.py
D=$P
step prc25 prc_cal.py --alpha 0.25
P=$D   # prc_cal2 starts from dup_pair, not from prc25
step prc2 prc_cal2.py --cv_offset inv --test_offset inv --alpha 0.25 --oor 0.02
exit 0
