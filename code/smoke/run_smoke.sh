#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# CPU smoke test: the whole pipeline (stages 0-4, the five rehearsal phases, the final phase and the bundle chain) on small synthetic data, in a scratch copy of
# this tree.  Nothing real is read or written: the data is made up (smoke/make_*.py), no network is used, no GPU is needed.
#   bash smoke/run_smoke.sh [work dir]          default work dir: ${TMPDIR:-/tmp}/prc_smoke   (needs ~2 GB; the python on PATH must have requirements.txt installed)
#   SMOKE_DAYS=<n>                              days of synthetic data per month (default 8; a smaller slice runs faster and checks the same stages)
# What it checks: every stage runs to its end with the shipped code and the pinned libraries, the final file passes the template check, and its
# error is below that of a constant predictor on the synthetic test months.  It does NOT check accuracy of the real models (the data has no real signal).
# Speed-ups that apply to the smoke only (smoke/sitecustomize.py, active through PYTHONPATH): boosting rounds capped at EC_SMOKE_CAP, neural-net
# epochs capped at 3 and the hidden width at 128; EC_DEVICE=cpu moves every GPU model (XGBoost, CatBoost, torch) to the CPU.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
WORK=$(mkdir -p "${1:-${TMPDIR:-/tmp}/prc_smoke}" && cd "${1:-${TMPDIR:-/tmp}/prc_smoke}" && pwd)
CODE="$WORK/code"
rm -rf "$CODE" "$WORK/synth"
mkdir -p "$CODE" "$WORK/synth"
for d in overnight pipeline scripts recipes smoke; do cp -r "$SRC/$d" "$CODE/$d"; done
cp "$SRC/requirements.txt" "$CODE/"
cd "$CODE"
export EC_DEVICE=cpu EC_SMOKE_CAP=${EC_SMOKE_CAP:-40} PYTHONPATH="$CODE/smoke" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=
# the plan-window calibrations skip runway-month cells under 100 rows: on the small synthetic data they would all be skipped
export TAIL_ARGS_prc25="--min_rows 3 --min_flt 50" TAIL_ARGS_prc2="--min_rows 3 --min_flt 50"
export PRC_REAL_PYTHON=$(command -v python) PATH="$CODE/smoke/bin:$PATH"   # smoke/bin/python: runs the interpreter and kills a finished script that hangs at exit
export PY=${PY:-python}
T0=$(date +%s)
trap 'rc=$?; echo "SMOKE FAILED (exit $rc); newest stage log, logs/$(ls -t logs 2>/dev/null | head -1):"; tail -n 25 "logs/$(ls -t logs 2>/dev/null | head -1)" 2>/dev/null; exit $rc' ERR
step() { echo; echo "=== [$(( $(date +%s) - T0 )) s] $*"; }

step "synthetic challenge files, ADS-B points, open external data"
$PY smoke/make_synthetic.py --days "${SMOKE_DAYS:-8}" --out "$WORK/synth/raw" --zip "$WORK/synth/files_list.zip" --truth "$WORK/synth/truth.parquet"
mkdir -p data/external
$PY smoke/make_adsb.py --raw "$WORK/synth/raw" --truth "$WORK/synth/truth.parquet" --out data/external
$PY smoke/make_external.py --raw "$WORK/synth/raw" --truth "$WORK/synth/truth.parquet" --out data/external

step "stage 0: import, movement table (the downloads of pipeline/00_data.sh are replaced by the synthetic files)"
$PY scripts/import_data.py "$WORK/synth/files_list.zip"
$PY scripts/fetch_final.py --from-dir "$WORK/synth/raw"      # the bucket download is replaced by the synthetic test-month files
( cd overnight && $PY -u prep.py )

step "stage 1: features"
bash pipeline/10_features.sh
step "stage 2: members"
bash pipeline/20_members.sh
step "stage 3: chain on the 2025 side (CTX=0)"
CTX=0 bash pipeline/30_chain.sh

for p in pseudo2 pseudo3 pseudo4 pseudo5 pseudo6; do
  step "rehearsal phase $p"
  EC_PHASE=$p bash pipeline/50_final_phase.sh data seed features members chain file
done

step "final phase (m12 stackers, chain, post-step, file check)"
EC_PHASE=final bash pipeline/50_final_phase.sh seed features members chain
bash pipeline/40_final_file.sh outputs/overnight_final/test/ar_132_prc2.parquet   # stage 3b chain; the script defaults to the stage 3c file

step "score against the synthetic truth"
$PY smoke/score_smoke.py outputs/overnight_final/test/ar_132_prc2_wit_zf3.parquet "$WORK/synth/truth.parquet" data/raw

step "bundle chain ar_133 (stage 3c): stream-transformer members on the base data and on every phase, stackers, tail, file"
bash pipeline/39_bnd1.sh base
for p in pseudo2 pseudo3 pseudo4 pseudo5 pseudo6 final; do
  EC_PHASE=$p bash pipeline/39_bnd1.sh members
done
EC_PHASE=final bash pipeline/39_bnd1.sh chain
bash pipeline/40_final_file.sh outputs/overnight_final/test/ar_133_prc2.parquet
$PY smoke/score_smoke.py outputs/overnight_final/test/ar_133_prc2_wit_zf3.parquet "$WORK/synth/truth.parquet" data/raw
step "done"
