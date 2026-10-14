# SPDX-License-Identifier: GPL-3.0-or-later
"""Counts of (recorded label, prediction) pairs on a 30 s grid, for the predicted-versus-actual figure.

Read-only: it bins predictions that already exist. Rows come from the final chain's validation run on January and July 2025 (out of fold,
344,419 departures, the same rows as the validation table in docs/results.md). Writes writeup/data/pred_actual.csv (columns: y_lo, p_lo, n; the
lower edge of each 30 s cell, 0 to 2,400 s on both axes) and writeup/data/pred_actual_summary.csv (rows outside the grid, quantiles of the error), writeup/data/pred_error_hist.csv (prediction minus label in 10 s bins, the two end bins hold everything beyond 600 s).

  python writeup/pred_actual_table.py --oof <validation run's out-of-fold file> [--out <dir>]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STEP, TOP = 30, 2400
ESTEP, EDGE = 10, 600  # error histogram: 10 s bins, prediction minus label clipped to +-600 s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="parquet with columns y (label, s) and pred (prediction, s)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data"))
    a = ap.parse_args()
    d = pd.read_parquet(a.oof, columns=["y", "pred"])
    inside = d[(d.y >= 0) & (d.y < TOP) & (d.pred >= 0) & (d.pred < TOP)]
    yb = (inside.y // STEP * STEP).astype(int)
    pb = (inside.pred // STEP * STEP).astype(int)
    c = pd.DataFrame({"y_lo": yb, "p_lo": pb}).value_counts().rename("n").reset_index().sort_values(["y_lo", "p_lo"])
    out = Path(a.out)
    c.to_csv(out / "pred_actual.csv", index=False)
    err = d.pred - d.y
    short = d[d.y < 3600]
    s = pd.DataFrame([
        ("rows", len(d)),
        ("rows_in_grid", len(inside)),
        ("rows_label_at_or_above_2400s", int((d.y >= TOP).sum())),
        ("share_of_squared_error_label_at_or_above_2400s", float((err[d.y >= TOP] ** 2).sum() / (err ** 2).sum())),
        ("median_label_s", d.y.median()),
        ("median_prediction_s", d.pred.median()),
        ("median_abs_error_s", err.abs().median()),
        ("p90_abs_error_s", err.abs().quantile(0.90)),
        ("share_within_120s", float((err.abs() <= 120).mean())),
        ("share_within_300s", float((err.abs() <= 300).mean())),
        ("rmse_label_under_3600s", float(np.sqrt(((short.pred - short.y) ** 2).mean()))),
    ], columns=["name", "value"])
    s.to_csv(out / "pred_actual_summary.csv", index=False)
    e = err.clip(-EDGE, EDGE - 1e-6)
    h = pd.Series((e // ESTEP * ESTEP).astype(int)).value_counts().rename("n").rename_axis("err_lo").reset_index().sort_values("err_lo")
    h.to_csv(out / "pred_error_hist.csv", index=False)
    print(c.n.sum(), "rows binned;", s.set_index("name").value.round(4).to_dict())


if __name__ == "__main__":
    main()
