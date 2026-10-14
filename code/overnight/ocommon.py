# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared helpers for the overnight experiments: paths, progress reporting, folds, metrics."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Phase switch (see README.md, stages 3 and 4). EC_PHASE unset = the ranking phase (ranking.parquet = Jan/Jul 2026, template
# submitting.parquet). EC_PHASE=final = the blind final (final_ranking.parquet is the 'rank' split, template
# final_submitting.parquet); EC_PHASE=pseudo = a rehearsal of it built from labelled 2025 months (scripts/make_pseudo_final.py).
# Every phase writes to its own data/processed_<phase>, reports/overnight_<phase>, outputs/overnight_<phase>; the split name
# stays 'rank' in all of them, so the stage scripts do not change. Seed a phase with scripts/phase_seed.py.
PHASE = os.environ.get("EC_PHASE", "")
SFX = f"_{PHASE}" if PHASE else ""
# EC_PHASE=<phase>_hm: an analysis option for scoring a phase with the ADS-B tracks hidden on a fixed set of rows (stx.py --mask_val hm).
# No stage of this tree uses it: every shipped stage runs on the plain phase names, so HM is False and PLAIN_* equal the phase's own dirs.
# Using it needs data/hide/<phase>_hm.parquet (one column MVT_ID_mvt, the rows to hide); the script that drew those rows is not included.
# An _hm phase reads the unmasked phase's files: PLAIN_* are that phase's dirs.
HM = PHASE.endswith("_hm")
PLAIN_PHASE = PHASE[:-3] if HM else PHASE
PLAIN_SFX = f"_{PLAIN_PHASE}" if PLAIN_PHASE else ""
RAW = ROOT / (f"data/{PLAIN_PHASE}/raw" if PLAIN_PHASE.startswith("pseudo") else "data/raw")
TEST_FILE = RAW / ("final_ranking.parquet" if PHASE else "ranking.parquet")
TEMPLATE = RAW / ("final_submitting.parquet" if PHASE else "submitting.parquet")
PROC = ROOT / f"data/processed{SFX}"
REP = ROOT / f"reports/overnight{SFX}"
OUTD = ROOT / f"outputs/overnight{SFX}"
PLAIN_PROC = ROOT / f"data/processed{PLAIN_SFX}"
PLAIN_OUTD = ROOT / f"outputs/overnight{PLAIN_SFX}"
for d in (PROC,):
    d.mkdir(parents=True, exist_ok=True)
for d in (REP, OUTD, REP / "results", OUTD / "oof", OUTD / "test"):
    d.mkdir(parents=True, exist_ok=True)

# EC_DEVICE=cpu: every step that asks for a CUDA card (XGBoost, CatBoost, PyTorch) runs on the CPU instead (smoke test, boxes without a GPU).
# Unset = the cards of the recorded run.  CatBoost on the CPU and on a GPU give different trees; only the GPU run is the recorded one.
CPU = os.environ.get("EC_DEVICE", "") == "cpu"
XGB_DEV = "cpu" if CPU else "cuda"
CB_TASK = "CPU" if CPU else "GPU"
CB_GPU = {} if CPU else {"gpu_ram_part": 0.9}
TORCH_DEV = "cpu" if CPU else "cuda"

# Worker pools (ProcessPoolExecutor) must not be forked after polars has started its threads: on Linux, Python before 3.14 forks by default and the
# workers then hang.  Python 3.14 defaults to the forkserver start method; use the same one on older versions (the scripts do not rely on fork inheritance).
import multiprocessing as _mp  # noqa: E402
import sys as _sys  # noqa: E402
if _sys.platform.startswith("linux") and _sys.version_info < (3, 14):
    try:
        _mp.set_start_method("forkserver")
    except RuntimeError:
        pass

_progress = None


def report(run, done, total, detail, finished=False, purpose=""):
    print(f"[{time.strftime('%H:%M:%S')}] [{done}/{total}] {detail}", flush=True)
    if _progress:
        # Lab > Runs files a live card under a project only when its id starts with "<project>-"
        rid = run if run.startswith("eurocontrol-") else "eurocontrol-" + run
        _progress.report(rid, "EUROCONTROL " + run, done, total, detail=detail,
                         finished=finished, purpose=purpose)


# Season-matched folds with 1-month buffers (docs/model.md).
FOLDS = {
    "jan": {"val": [1], "train": [3, 4, 5, 6, 7, 8, 9, 10, 11]},
    "jul": {"val": [7], "train": [1, 2, 3, 4, 5, 9, 10, 11, 12]},
}
if os.environ.get("EC_FOLDS"):  # B1 month-complete OOF: EC_FOLDS=2,3,4 -> one fold per month, 1-month buffers each side
    FOLDS = {f"m{m:02d}": {"val": [m], "train": [k for k in range(1, 13) if min((k - m) % 12, (m - k) % 12) > 1]}
             for m in map(int, os.environ["EC_FOLDS"].split(","))}


def write_merged(path, new, key="MVT_ID_mvt"):
    """Write per-row features `new` (a pandas frame). With EC_MERGE=1 the rows already in `path` (the 2025 rows seeded by
    scripts/phase_seed.py, whose raw ADS-B points are not on this host) are kept unless `new` has the key."""
    import pandas as pd
    if os.environ.get("EC_MERGE") == "1" and Path(path).exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[~old[key].isin(new[key])], new], ignore_index=True)
    new.to_parquet(path, index=False)
    return new


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def save_result(res):
    (REP / "results" / f"{res['id']}.json").write_text(json.dumps(res, indent=2))
    with open(REP / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in res.items() if k != "per_airport"}) + "\n")


def hide_ids(phase=None):
    """The fixed hidden-track row ids of a phase's masked mirror (analysis option only, see EC_PHASE=<phase>_hm above): data/hide/<phase>_hm.parquet."""
    import pandas as pd
    ph = PLAIN_PHASE if phase is None else phase
    return pd.read_parquet(ROOT / f"data/hide/{ph or 'base'}_hm.parquet").MVT_ID_mvt.to_numpy()
