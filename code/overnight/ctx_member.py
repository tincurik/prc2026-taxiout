# SPDX-License-Identifier: GPL-3.0-or-later
"""features_dep_ctxm: features_ctx.py without the 16 columns that carry the base chain's predictions (cx_nb_*pred*, own_m_nbpred_pm30), i.e. only
the label-free departure-stream windows (delay vs SCHED / EOBT / LOBT, AOBT off the synthetic mode, backlog, throughput, arrivals).
Those predictions exist only on the Jan/Jul OOF rows and the test rows, so a member trained on all 2025 months cannot use them.
Phase-aware.   [EC_PHASE=pseudo2] python ctx_member.py [--src ctx] [--out ctxm]"""
import argparse

import pandas as pd

from ocommon import PROC

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="ctx")
ap.add_argument("--out", default="ctxm")
a = ap.parse_args()
d = pd.read_parquet(PROC / f"features_dep_{a.src}.parquet")
drop = [c for c in d.columns if "pred" in c]
d = d.drop(columns=drop)
d.to_parquet(PROC / f"features_dep_{a.out}.parquet", index=False)
print(f"{a.out}: {d.shape}, dropped {len(drop)} prediction columns", flush=True)
