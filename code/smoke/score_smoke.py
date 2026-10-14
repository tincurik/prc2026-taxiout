# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke-test verdict: the final-phase prediction file against the synthetic truth of the test months.
Checks the file is complete and not garbage: its RMSE must be below the constant predictor that already knows the test months' mean (a misaligned or
broken file fails this). Accuracy is not tested: the data are made up and carry little real signal. Two airport means are printed for reference:
the one fitted on the synthetic 2025 training months (the baseline a real model has to beat) and the one fitted on the test labels themselves (an oracle no model
can use, so a file above it is not "worse than the baseline").

  python smoke/score_smoke.py <prediction file> <synthetic_truth.parquet> <data/raw>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(pred_path, truth_path, raw):
    p = pd.read_parquet(pred_path)
    t = pd.read_parquet(truth_path).rename(columns={"TAXITIME_SEC_mvt": "y"})
    tmpl = pd.read_parquet(f"{raw}/final_submitting.parquet")
    assert len(p) == len(tmpl) == len(t), (len(p), len(tmpl), len(t))
    d = t.merge(p, on="MVT_ID_mvt", how="left")
    assert d.TAXITIME_SEC_mvt.notna().all(), "predictions missing"
    rank = pd.read_parquet(f"{raw}/final_ranking.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"])
    d = d.merge(rank, on="MVT_ID_mvt", how="left")
    rmse = float(np.sqrt(((d.TAXITIME_SEC_mvt - d.y) ** 2).mean()))
    const = float(np.sqrt(((d.y - d.y.mean()) ** 2).mean()))
    apt = float(np.sqrt(((d.y - d.groupby("ADEP_mvt").y.transform("mean")) ** 2).mean()))
    tr = pd.concat([pd.read_parquet(f, columns=["ADEP_mvt", "TAXITIME_SEC_mvt"]) for f in sorted(Path(raw).glob("training_2025-*.parquet"))])
    base = float(np.sqrt(((d.y - d.ADEP_mvt.map(tr.groupby("ADEP_mvt").TAXITIME_SEC_mvt.mean()).fillna(tr.TAXITIME_SEC_mvt.mean())) ** 2).mean()))
    print(f"smoke final file: {len(d):,} departures, RMSE {rmse:.1f} s (constant at the test mean {const:.1f}; airport mean of the training months {base:.1f}, "
          f"the baseline of the real data; airport mean of the test labels {apt:.1f}, an oracle)")
    assert rmse < const, "the pipeline's file is worse than a constant: something is broken"
    print("SMOKE OK")


if __name__ == "__main__":
    main(*sys.argv[1:4])
