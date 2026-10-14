# SPDX-License-Identifier: GPL-3.0-or-later
"""Average the draws of the seed bag, row by row (the shipped file is the average of K = 3 draws of the final chain).

  python scripts/bag_build.py --id ar_134naclh_tww [--k 3]
      reads   outputs/overnight_<phase>/test/<id>.parquet   for the phases final, finalb2, ..., finalb<k>   (draw 1 is the final phase of `make final`, draws 2..k are pipeline/45_bag_draw.sh)
      writes  outputs/overnight_final/test/<id>_bag<k>.parquet           the average (the MVT_ID_mvt order of draw 1, the TAXITIME_SEC_mvt column averaged)
              outputs/overnight_final/test/<id>_bag<k>.bag.json          sha256 of every draw, the draws' pairwise rms difference, the estimated gain of the average
  then (pipeline/46_bag_file.sh does all three steps): scripts/witness_post.py --source <that> --ideas zf3   (plan-window step), scripts/check_final.py.

Each draw has run the whole chain, post-steps included, so the average is of finished prediction files; the post-steps are not linear, so
averaging the files and running the post-steps on an averaged base would differ slightly (the shipped order is the first).
The estimated gain is the variance argument: if the draws differ by independent noise of variance s^2 each (s^2 = half the mean squared pairwise difference), the average
of K of them has a mean squared error lower by (1 - 1/K) s^2, so the RMSE falls by about (1 - 1/K) s^2 / (2 RMSE).  It is an estimate: the draws share one
out-of-fold side (the 2025 files are not redrawn), so the bag has no cross-validation score of its own.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COL = "TAXITIME_SEC_mvt"


def phases(k):
    return ["final"] + [f"finalb{i}" for i in range(2, k + 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="the chain output to average (after its post-steps), e.g. ar_134naclh_tww")
    ap.add_argument("--k", type=int, default=3, help="number of draws (default 3: the final phase plus finalb2 and finalb3)")
    ap.add_argument("--rmse0", type=float, default=279.1, help="RMSE of the chain on the two validation folds: only the scale of the gain estimate")
    a = ap.parse_args()
    files = {p: ROOT / f"outputs/overnight_{p}/test/{a.id}.parquet" for p in phases(a.k)}
    missing = [str(f) for f in files.values() if not f.exists()]
    if missing:
        raise SystemExit("bag_build: missing draw file(s): " + ", ".join(missing) + "  (draw k: bash pipeline/45_bag_draw.sh k)")
    d = {p: pd.read_parquet(f) for p, f in files.items()}
    ref = d["final"]
    for p, v in d.items():
        assert v.MVT_ID_mvt.equals(ref.MVT_ID_mvt) and v.shape == ref.shape and list(v.columns) == list(ref.columns), p
        assert np.isfinite(v[COL]).all(), p
    X = np.stack([v[COL].to_numpy(float) for v in d.values()])
    K = len(d)
    out = ref.copy()
    out[COL] = X.mean(0)
    dst = ROOT / f"outputs/overnight_final/test/{a.id}_bag{K}.parquet"
    out.to_parquet(dst, index=False)
    rms = {f"{i}~{j}": float(np.sqrt(np.mean((X[p] - X[q]) ** 2))) for (p, i), (q, j) in itertools.combinations(list(enumerate(d)), 2)}
    sig2 = np.mean([v ** 2 for v in rms.values()]) / 2
    info = dict(K=K, draws={p: hashlib.sha256(f.read_bytes()).hexdigest() for p, f in files.items()}, rms_diff=rms,
                est_gain_rmse=-(1 - 1 / K) * sig2 / (2 * a.rmse0), mean_abs_bag_minus_draw1=float(np.abs(X.mean(0) - X[0]).mean()),
                out=str(dst.relative_to(ROOT)))
    dst.with_suffix(".bag.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
