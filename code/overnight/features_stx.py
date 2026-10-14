# SPDX-License-Identifier: GPL-3.0-or-later
"""Stream-transformer output as a raw stacker feature (not a member): stxf_reg = stx.py's regressor output (the departure's own
prediction after attending to the airport's departure + arrival stream) and stxf_reg_gap = stxf_reg minus the member-blend mean
(features_memb mb_reg, when present), as columns the stackers can split on. bg_stxA put the same member in features_memb's gap /
spread columns; this gives the four stackers the output itself.
OOF rows: outputs/overnight<phase>/oof/<id>.parquet (month OOF); test rows: outputs/overnight<phase>/test/<id>_raw.parquet (refit).
Writes PROC/features_dep_<out>.parquet.   [EC_PHASE=pseudo2] python features_stx.py [--id stx_v2] [--out stxf] [--cols sig,p1800,p3600]
--prefix names the columns of another reg member (mlp.py --model tabm: --prefix tbm -> tbm_reg, tbm_reg_gap) beside the stx ones.
"""
import argparse

import numpy as np
import pandas as pd

from ocommon import OUTD, PROC

ap = argparse.ArgumentParser()
ap.add_argument("--id", default="stx_v2")
ap.add_argument("--out", default="stxf")
ap.add_argument("--cols", default="", help="more columns of the stx parquets to export as stxf_<col> (stx_v3: sig,p1800,p3600; knn_m50,...)")
ap.add_argument("--memb", default="memb2", help="features_dep_<name> whose mb_reg gives the gap column ('' = none)")
ap.add_argument("--prefix", default="stxf", help="column prefix (<prefix>_reg, <prefix>_reg_gap, <prefix>_<col>)")
a = ap.parse_args()
cols = [c for c in a.cols.split(",") if c]
rows = []
for p in (OUTD / "oof" / f"{a.id}.parquet", OUTD / "test" / f"{a.id}_raw.parquet"):
    rows.append(pd.read_parquet(p)[["MVT_ID_mvt", "reg"] + cols].drop_duplicates("MVT_ID_mvt"))
f = pd.concat(rows, ignore_index=True).drop_duplicates("MVT_ID_mvt")
P = a.prefix
f = f.rename(columns={c: f"{P}_{c}" for c in ["reg"] + cols})
if a.memb and (PROC / f"features_dep_{a.memb}.parquet").exists():
    m = pd.read_parquet(PROC / f"features_dep_{a.memb}.parquet", columns=["MVT_ID_mvt", "mb_reg"])
    f = f.merge(m, on="MVT_ID_mvt", how="left")
    f[f"{P}_reg_gap"] = f[f"{P}_reg"] - f.pop("mb_reg")
f = f.astype({c: "float32" for c in f.columns if c != "MVT_ID_mvt"})
assert f.MVT_ID_mvt.is_unique and f[f"{P}_reg"].notna().all()
f.to_parquet(PROC / f"features_dep_{a.out}.parquet", index=False)
print(f.shape, f.describe().T.round(1))
