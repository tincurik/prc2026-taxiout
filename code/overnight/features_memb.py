# SPDX-License-Identifier: GPL-3.0-or-later
"""Member-disagreement features for the stacker (level-2 signal the NNLS blend discards):
per blend_v48 (--blend) member the regressor's gap to the blend regressor, plus the spread (weighted std, range) and the blend reg.
OOF rows from outputs/overnight/oof/<member>.parquet (month-OOF), test rows from outputs/overnight/test/<member>_raw.parquet
(full refit). Writes data/processed/features_dep_memb.parquet (OOF and test ids stacked; ids never overlap).
   python features_memb.py [--all 1 --out memb2]"""
import argparse
import json

import numpy as np
import pandas as pd

from ocommon import OUTD, PROC, REP

ap = argparse.ArgumentParser()
ap.add_argument("--all", type=int, default=0, help="1 = every blend_v48 member incl. the zero-weight ones (spread from equal weights)")
ap.add_argument("--out", default="memb")
ap.add_argument("--add", default="", help="comma list of further member ids (oof + test _raw with reg), weight 0.05 in the mean/spread")
ap.add_argument("--blend", default="blend_v48", help="the blend whose members and weights to use")
a = ap.parse_args()
W = json.loads((REP / "results" / f"{a.blend}.json").read_text())["weights"]
W = {k: (v if v > 0 else (0.05 if a.all else 0)) for k, v in W.items()}
W.update({k: 0.05 for k in a.add.split(",") if k})
W = {k: v for k, v in W.items() if v > 0}
rows = []
for split in ("oof", "test"):
    path = (lambda m: OUTD / "oof" / f"{m}.parquet") if split == "oof" else (lambda m: OUTD / "test" / f"{m}_raw.parquet")
    ids = None
    R = []
    for m in W:
        r = pd.read_parquet(path(m))[["MVT_ID_mvt", "reg"]].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").reg
        ids = r.index if ids is None else ids
        R.append(r.reindex(ids).to_numpy())
    R = np.stack(R, 1)
    w = np.array(list(W.values()))
    w = w / w.sum()
    mean = R @ w
    f = pd.DataFrame({"MVT_ID_mvt": ids.to_numpy()})
    for i, m in enumerate(W):
        f[f"mb_{m[:12]}_{i}"] = (R[:, i] - mean).astype("float32")
    f["mb_std"] = np.sqrt(((R - mean[:, None]) ** 2) @ w).astype("float32")
    f["mb_range"] = (R.max(1) - R.min(1)).astype("float32")
    f["mb_reg"] = mean.astype("float32")
    rows.append(f)
out = pd.concat(rows, ignore_index=True)
assert out.MVT_ID_mvt.is_unique and out.drop(columns="MVT_ID_mvt").notna().all().all()
out.to_parquet(PROC / f"features_dep_{a.out}.parquet", index=False)
print(out.shape, list(out.columns))
print(out.groupby(out.MVT_ID_mvt.isin(pd.read_parquet(OUTD / "oof" / "mix_v39_cat_adsb_lobt.parquet").MVT_ID_mvt))[["mb_std", "mb_range"]].describe().T.round(1))
