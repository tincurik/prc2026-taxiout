# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared paths and run settings (EC_PHASE) and the raw-data directory."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE = os.environ.get("EC_PHASE", "")  # "" | final | pseudo, see overnight/ocommon.py
RAW = ROOT / (f"data/{PHASE}/raw" if PHASE.startswith("pseudo") else "data/raw")
ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"

_progress = None


def report(run, done, total, detail, finished=False):
    print(f"[{done}/{total}] {detail}", flush=True)
    if _progress:
        _progress.report(run, "EUROCONTROL " + run.split("-", 1)[-1],
                         done, total, detail=detail, finished=finished,
                         purpose="Set up and verify the PRC 2026 taxi-out prediction workflow")
