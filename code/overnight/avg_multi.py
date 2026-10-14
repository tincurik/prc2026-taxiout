# SPDX-License-Identifier: GPL-3.0-or-later
"""Equal (or weighted) average of N finished runs (OOF + test), saved as a new run:
   python avg_multi.py --id X --ids a,b,c [--w 1,1,2]  [--dry 1]   (dry: print CV only, write nothing)"""
import argparse
import sys

import numpy as np
import pandas as pd

from ocommon import OUTD, ROOT, rmse, save_result

ap = argparse.ArgumentParser()
ap.add_argument("--id", required=True)
ap.add_argument("--ids", required=True)
ap.add_argument("--w", default="")
ap.add_argument("--dry", type=int, default=0)
a = ap.parse_args()
ids = a.ids.split(",")
w = np.array([float(x) for x in a.w.split(",")]) if a.w else np.ones(len(ids))
w = w / w.sum()
o0 = pd.read_parquet(OUTD / "oof" / f"{ids[0]}.parquet")
P = np.stack([o0.pred.to_numpy()] + [pd.read_parquet(OUTD / "oof" / f"{i}.parquet").set_index("MVT_ID_mvt").pred.reindex(o0.MVT_ID_mvt).to_numpy() for i in ids[1:]], 1)
assert not np.isnan(P).any()
pred = P @ w
y = o0.y.to_numpy()
cv = rmse(pred, y)
print(a.id, "CV", round(cv, 3), {i: round(rmse(P[:, k], y), 3) for k, i in enumerate(ids)})
if a.dry:
    sys.exit()
o = o0.copy()
o["pred"] = pred
o[["MVT_ID_mvt", "fold", "y", "pred"]].to_parquet(OUTD / "oof" / f"{a.id}.parquet", index=False)
t0 = pd.read_parquet(OUTD / "test" / f"{ids[0]}.parquet")
T = np.stack([t0.TAXITIME_SEC_mvt.to_numpy()] + [t0.MVT_ID_mvt.map(pd.read_parquet(OUTD / "test" / f"{i}.parquet").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt).to_numpy() for i in ids[1:]], 1)
assert not np.isnan(T).any()
t0["TAXITIME_SEC_mvt"] = T @ w
path = OUTD / "test" / f"{a.id}.parquet"
t0.to_parquet(path, index=False)
sys.path.insert(0, str(ROOT / "scripts"))
from baseline import validate_submission
validate_submission(path)
res = {"id": a.id, "model": "avg_multi", "ids": ids, "w": w.tolist(), "cv_rmse": cv, "cv_trim_rmse": rmse(pred[y <= 20000], y[y <= 20000]),
       "cv_clean_rmse": rmse(pred[y <= 7200], y[y <= 7200]), "member_cv": {i: rmse(P[:, k], y) for k, i in enumerate(ids)}, "folds": {}}
for f in ("jan", "jul"):
    k = (o0.fold == f).to_numpy()
    res["folds"][f] = {"rmse": rmse(pred[k], y[k]), "n": int(k.sum())}
save_result(res)
