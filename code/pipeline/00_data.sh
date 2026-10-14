#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Stage 0: challenge data, open external data, ADS-B ground points.  Run from the code root: bash pipeline/00_data.sh path/to/files_list.zip   (or a directory that holds the unpacked parquet files)
# Nothing in this repository is data: every file below is downloaded from its owner.  Sources, licences and the date each was retrieved: docs/data-sources.md.
set -e
cd "$(dirname "$0")/.."
ZIP=${1:?usage: bash pipeline/00_data.sh <the challenge archive files_list.zip, or a directory holding its parquet files>}
PY=${PY:-python}

# 0a. Challenge files: the organisers' archive (training_2025-*.parquet, ranking.parquet, submitting.parquet) -> data/raw/.
$PY scripts/import_data.py "$ZIP"
#     The four test months (final_ranking.parquet + final_submitting.parquet) come from the organisers' read-only bucket (S3 credentials in the
#     environment, see scripts/fetch_final.py; FINAL_DIR=<directory> uses two files you already have instead).  SKIP_FINAL_FETCH=1 skips the line
#     to build everything except the final phase.
if [ -z "$SKIP_FINAL_FETCH" ] && ! $PY scripts/fetch_final.py ${FINAL_DIR:+--from-dir "$FINAL_DIR"}; then
  cat >&2 <<'MSG'

00_data.sh: the four test months (final_ranking.parquet, final_submitting.parquet) could not be fetched. Pick one:
  - S3 credentials: export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (or AWS_PROFILE; S3_ENDPOINT overrides the endpoint), then run this script again;
  - the two files are already on disk: FINAL_DIR=<directory holding them> bash pipeline/00_data.sh <archive>;
  - everything except the final phase (stages 1 to 3 and the rehearsals): SKIP_FINAL_FETCH=1 bash pipeline/00_data.sh <archive>.
Step 0a (the challenge archive) has already run; running the script again does not overwrite what it imported.
MSG
  exit 1
fi

# 0b. Combined movement table (2025 training + 2026 ranking) -> data/processed/movements.parquet
( cd overnight && $PY -u prep.py )

# 0c. Open external data -> data/external/<source>/ with SOURCE.md and files.jsonl per source (URL, date, sha256).
$PY scripts/fetch_external.py                 # no argument: every source in the script
$PY scripts/fetch_prc_dashboard.py            # ansperformance.eu airport dashboards (monthly taxi-out tables)

# 0d. adsb.lol ground and near-airport points (terms: docs/data-sources.md), one parquet per day under data/external/adsblol/points/.  Resumable per day; --days shards the work.
#     Days of 2025 and Jan/Jul 2026 (the ranking phase), then the days of the four test months.
$PY scripts/adsb_fetch.py --workers ${ADSB_WORKERS:-3}
$PY scripts/adsb_fetch.py --final --workers ${ADSB_WORKERS:-3}
# 0e. Plausibility check of every day file (exit 1 on a partial or thin day).
$PY scripts/validate_days.py
