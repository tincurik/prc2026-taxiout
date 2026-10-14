# SPDX-License-Identifier: GPL-3.0-or-later
"""Download final_ranking.parquet and final_submitting.parquet (the four test months and the rows to predict) from the organisers'
read-only datasets bucket into data/raw/.  Never overwrites a file already there unless --refresh.

S3 credentials come from the usual boto3 sources: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the environment (or AWS_PROFILE naming a profile
in ~/.aws/credentials); nothing is stored in this repository.  S3_ENDPOINT overrides the endpoint (default: the challenge's S3 service).
Without credentials the script stops with that message.  If the two files were already downloaded some other way, point --from-dir at the
directory that holds them: they are copied into data/raw/ and checked, and no network or credentials are needed.

  python scripts/fetch_final.py [--refresh] [--from-dir DIR]
"""
import argparse
import hashlib
import os
import shutil
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.opensky-network.org")
BUCKET = "prc-2026-datasets"
NAMES = ("final_ranking.parquet", "final_submitting.parquet")


def fetch(a):
    session = boto3.Session()
    if session.get_credentials() is None:
        raise SystemExit("fetch_final: no S3 credentials found.  Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (or AWS_PROFILE) for the challenge's "
                         f"read-only bucket {BUCKET} (endpoint {ENDPOINT}, S3_ENDPOINT overrides it), or pass --from-dir <directory holding "
                         f"{' and '.join(NAMES)}> if you already have the files.")
    s3 = session.client("s3", endpoint_url=ENDPOINT, region_name="us-east-1",
                        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}, connect_timeout=10, read_timeout=60,
                                      retries={"max_attempts": 3}))
    try:
        keys = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET).get("Contents", [])}
    except (BotoCoreError, ClientError) as e:
        raise SystemExit(f"fetch_final: cannot list bucket {BUCKET} at {ENDPOINT}: {e}.  Check the credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) "
                         "and the endpoint (S3_ENDPOINT), or pass --from-dir.")
    for name in NAMES:
        if name not in keys:
            raise SystemExit(f"{name} is not in the bucket; keys: {sorted(keys)}")
        if (RAW / name).exists() and not a.refresh:
            print(f"{name} already in data/raw (use --refresh to download again)")
        else:
            s3.download_file(BUCKET, name, str(RAW / name))


def from_dir(a):
    src = Path(a.from_dir)
    missing = [n for n in NAMES if not (src / n).is_file()]
    if missing:
        raise SystemExit(f"fetch_final: {', '.join(missing)} not found in {src}; the directory must hold {' and '.join(NAMES)}")
    for name in NAMES:
        if (RAW / name).exists() and not a.refresh and (RAW / name).read_bytes() != (src / name).read_bytes():
            raise SystemExit(f"fetch_final: data/raw/{name} exists and differs from {src / name}; use --refresh to replace it")
        if not (RAW / name).exists() or a.refresh:
            shutil.copyfile(src / name, RAW / name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="download or copy again even if the files exist")
    ap.add_argument("--from-dir", help="directory that already holds final_ranking.parquet and final_submitting.parquet (no network needed)")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    from_dir(a) if a.from_dir else fetch(a)
    for name in NAMES:
        dest = RAW / name
        print(f"{name}: {dest.stat().st_size} bytes, sha256 {hashlib.sha256(dest.read_bytes()).hexdigest()}")
    template = pd.read_parquet(RAW / NAMES[1])
    ranking = pd.read_parquet(RAW / NAMES[0], columns=["MVT_ID_mvt", "PHASE_mvt"])
    deps = set(ranking.loc[ranking["PHASE_mvt"].eq("DEP"), "MVT_ID_mvt"])
    missing = int((~template["MVT_ID_mvt"].isin(deps)).sum())
    if missing:
        raise SystemExit(f"{missing} template ids are not departures of {NAMES[0]}")
    print(f"{NAMES[1]}: {len(template):,} rows, all departures of {NAMES[0]} ({len(ranking):,} rows)")


if __name__ == "__main__":
    main()
