# SPDX-License-Identifier: GPL-3.0-or-later
"""Blend the regressors of several final.py refits (test predictions), mirroring blend.py on the OOF side.

pred = p_imp*(d_sched+delta) + (1-p_imp)*sum_i w_i reg_i, then LIRF cell post-processing (alpha 0.5), clip at 0,
template order, validated. Weights: --weights, or the NNLS weights stored in a blend result (--weights_from).

python final_blend.py --id blend_seeds --parts mix_v18_fs3_deep,mix_v18_s1,mix_v18_s2 [--weights_from blend_seeds]
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import polars as pl

import postproc
from ocommon import OUTD, PROC, REP, ROOT, report, TEMPLATE

sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--parts", required=True, help="final.py output ids (<id>_raw.parquet); p_imp taken from the first")
    ap.add_argument("--weights", default="")
    ap.add_argument("--weights_from", default="")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--p_test", default="", help="take test p_imp from outputs/overnight/test/<this>.parquet (an earlier classifier's file; not used by the shipped chain) instead of parts[0]")
    a = ap.parse_args()
    parts = a.parts.split(",")
    run = f"final-blend-{a.id}"
    report(run, 0, 2, "loading", purpose="Blend refit regressors into one test-month prediction")
    if a.weights_from:
        wd = json.loads((REP / "results" / f"{a.weights_from}.json").read_text())["weights"]
        w = np.array([wd[k] for k in wd])
        assert len(w) == len(parts), (wd, parts)
    elif a.weights:
        w = np.array([float(x) for x in a.weights.split(",")])
    else:
        w = np.ones(len(parts)) / len(parts)
    base = pd.read_parquet(OUTD / "test" / f"{parts[0]}_raw.parquet")[["MVT_ID_mvt", "p_imp"]]
    if a.p_test:
        pt = pd.read_parquet(OUTD / "test" / f"{a.p_test}.parquet")[["MVT_ID_mvt", "p_imp"]]
        base = base[["MVT_ID_mvt"]].merge(pt, on="MVT_ID_mvt", how="left")
        assert not base.p_imp.isna().any()
    R = []
    for pid in parts:
        r = pd.read_parquet(OUTD / "test" / f"{pid}_raw.parquet")[["MVT_ID_mvt", "reg"]]
        R.append(base[["MVT_ID_mvt"]].merge(r, on="MVT_ID_mvt", how="left")["reg"].to_numpy())
    R = np.stack(R, 1)
    assert not np.isnan(R).any()
    meta = postproc.load_meta()
    tr = meta[meta.split == "train"]
    delta = float(np.mean(tr.TAXITIME_SEC_mvt[tr.kind == "imp"] - tr.d_sched[tr.kind == "imp"]))
    out = base.merge(meta[["MVT_ID_mvt", "cell", "d_sched"]], on="MVT_ID_mvt", how="left")
    out["reg"] = R @ w
    out["pred"] = out.p_imp * (out.d_sched + delta) + (1 - out.p_imp) * out.reg
    rt, taxi = postproc.rates(tr, 8, 100)
    out["pred_model"] = out["pred"]
    out["pred"] = postproc.apply(out, rt, taxi, delta, alpha=a.alpha)
    n_over = int((out.pred != out.pred_model).sum())
    out["pred"] = out["pred"].clip(lower=0)
    out.drop(columns=["cell", "d_sched"]).to_parquet(OUTD / "test" / f"{a.id}_raw.parquet", index=False)
    template = pd.read_parquet(TEMPLATE)
    template["TAXITIME_SEC_mvt"] = template["MVT_ID_mvt"].astype("int64").map(out.set_index("MVT_ID_mvt")["pred"]).to_numpy()
    path = OUTD / "test" / f"{a.id}.parquet"
    template.to_parquet(path, index=False)
    from baseline import validate_submission
    validate_submission(path)
    pred = out.pred.to_numpy()
    report(run, 2, 2, f"wrote {path.name}: weights {np.round(w, 3).tolist()}, {n_over} rows overridden, mean {pred.mean():.1f}s, p99 {np.quantile(pred, .99):.0f}s", finished=True)


if __name__ == "__main__":
    main()
