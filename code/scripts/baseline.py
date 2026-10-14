# SPDX-License-Identifier: GPL-3.0-or-later
"""Airport-mean baseline, chronological validation, and local submission export."""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from common import ROOT, RAW, ID, TARGET, PHASE, report


def predict(train, test):
    means = train.groupby("ADEP_mvt")[TARGET].mean()
    return test["ADEP_mvt"].map(means).fillna(train[TARGET].mean()).to_numpy(dtype=float)


def validate_submission(path, template_name="final_submitting.parquet" if PHASE else "submitting.parquet"):
    template = pd.read_parquet(RAW / template_name)
    output = pd.read_parquet(path)
    if output.columns.tolist() != [ID, TARGET]:
        raise ValueError("Submission must have exactly the two template columns in order")
    if not output[ID].equals(template[ID]) or not output[ID].is_unique:
        raise ValueError("Submission must preserve every template ID, dtype and row order")
    values = output[TARGET].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Predictions must be finite, nonnegative seconds")
    print(f"Validated {len(output):,} submission rows: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=str, help="Only validate an existing submission")
    args = parser.parse_args()
    if args.validate:
        validate_submission(args.validate)
        return
    run = "eurocontrol-baseline-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    columns = [ID, "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", TARGET]
    frames = []
    for i, path in enumerate(sorted(RAW.glob("training_*.parquet"))):
        frame = pd.read_parquet(path, columns=columns)
        frames.append(frame.loc[frame["PHASE_mvt"].eq("DEP")].copy())
        report(run, i + 1, 15, "Loaded " + path.name)
    data = pd.concat(frames, ignore_index=True)
    valid = data[TARGET].notna() & np.isfinite(data[TARGET]) & data[TARGET].ge(0)
    excluded = int((~valid).sum())
    data = data.loc[valid]
    cutoff = pd.Timestamp("2025-11-01", tz="UTC")
    train = data.loc[data["MVT_TIME_UTC_mvt"] < cutoff]
    holdout = data.loc[data["MVT_TIME_UTC_mvt"] >= cutoff]
    if train.empty or holdout.empty:
        raise ValueError("Empty chronological training or validation partition")
    predictions = predict(train, holdout)
    errors = (predictions - holdout[TARGET].to_numpy()) ** 2
    scores = holdout[["ADEP_mvt"]].assign(squared_error=errors).groupby("ADEP_mvt").agg(
        rows=("squared_error", "size"), mse=("squared_error", "mean"))
    metrics = {"model": "airport_mean", "validation": "train Jan-Oct 2025; validate Nov-Dec 2025 (movement UTC)",
               "training_departures": len(train), "validation_departures": len(holdout),
               "excluded_missing_or_negative_labels": excluded,
               "rmse_seconds": float(np.sqrt(errors.mean())),
               "global_mean_rmse_seconds": float(np.sqrt(((holdout[TARGET] - train[TARGET].mean()) ** 2).mean())),
               "per_airport": {airport: {"rows": int(row.rows), "rmse_seconds": float(np.sqrt(row.mse))}
                               for airport, row in scores.iterrows()}}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/baseline_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report(run, 13, 15, f"Holdout RMSE: {metrics['rmse_seconds']:.2f} seconds")
    ranking = pd.read_parquet(RAW / "ranking.parquet", columns=[ID, "ADEP_mvt", "PHASE_mvt"])
    departures = ranking.loc[ranking["PHASE_mvt"].eq("DEP")].copy()
    departures[TARGET] = predict(data, departures)
    template = pd.read_parquet(RAW / "submitting.parquet")
    template[TARGET] = template[ID].map(departures.set_index(ID)[TARGET])
    out = ROOT / "outputs/baseline_submission.parquet"
    out.parent.mkdir(exist_ok=True)
    template.to_parquet(out, index=False)
    report(run, 14, 15, "Refit airport means on all valid 2025 departure labels")
    validate_submission(out)
    report(run, 15, 15, "Baseline and local submission verified", finished=True)


if __name__ == "__main__":
    main()
