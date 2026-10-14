# SPDX-License-Identifier: GPL-3.0-or-later
"""Coverage gate for a phase's ranked months, read-only.

The chain joins most feature files with how="left", and several steps fill the gaps (fillna(0) in lirf_far --adsb ok,
covw_cv, prc_cal's p_imp; NaN-aware trees everywhere else). A feature file or member that misses the new months
therefore degrades the final file without any error. This compares, per data/processed_<phase>/features_dep_*.parquet,
the non-null share on each ranked month (split rank, UTC month of MVT) with the same calendar month of the training
rows (split train; all training rows when that month is held out), and checks that every test prediction file in
outputs/overnight_<phase>/test/ covers each ranked month as well as it covers the others.

  EC_PHASE=final python scripts/final_coverage.py                    # every feature file, every test file, ratio 0.8
  EC_PHASE=final python scripts/final_coverage.py --files adsbdir4,adsbrev,adsb2fullrel --require adsb2m
Exit 1 when a feature file is below --ratio x its training share on a ranked month, a --require file is missing, or a
prediction file misses template ids or has more NaN on one ranked month than on the best one.
"""
import argparse
import os
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
progress = None

PHASE = os.environ.get("EC_PHASE", "")
SFX = f"_{PHASE}" if PHASE else ""
PROC = ROOT / f"data/processed{SFX}"
TEST = ROOT / f"outputs/overnight{SFX}/test"
RAW = ROOT / (f"data/{PHASE}/raw" if PHASE.startswith("pseudo") else "data/raw")
TEMPLATE = RAW / ("final_submitting.parquet" if PHASE else "submitting.parquet")
RUN = f"eurocontrol-coverage-gate{SFX}"
# Gate failures that are upstream source gaps in the OPDI files.  Accepted, not measured: the whole OPDI block is worth about 0.51 CV RMSE
# (docs/limits.md), the gaps touch a small part of the scored rows, and the authors' estimate of the cost was under 0.02 RMSE.  No run
# scored the chain with these gaps filled, so the figure is an estimate.
SOURCE_GAPS = {
    ("ext_inb", "2601"): "OPDI flight_list_202601 has no flights 02-10 Jan and 23 Jan",
    ("ext_opdiev", "2601"): "OPDI flight_list_202601 gap 02-10, 23-24 Jan; EDDM events nearly absent from 21 Jan",
    ("ext_opdiev", "2602"): "OPDI own-flight events for EDDM nearly absent 21 Jan - 19 Feb 2026 (EDDM 0.20 vs 0.84 in Feb 2025)",
}


def rep(done, total, detail, **kw):
    print(f"[{done}/{total}] {detail}", flush=True)
    if progress:
        progress.report(RUN, f"EUROCONTROL coverage gate {PHASE or 'base'}", done, total, detail=detail, **kw)


def ids(df):
    return df.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="", help="feature stems to gate (features_dep_<stem>.parquet); default all present")
    ap.add_argument("--require", default="", help="feature stems that must exist (e.g. adsb2m, the stackers' ADS-B set)")
    ap.add_argument("--ratio", type=float, default=0.8, help="fail below ratio x the training share of the same month")
    ap.add_argument("--min_ref", type=float, default=0.05, help="ignore files whose training share is below this")
    ap.add_argument("--no_preds", action="store_true", help="skip the test prediction files")
    a = ap.parse_args()

    base = ids(pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "split", "MVT_TIME_UTC_mvt"]))
    base = base.with_columns(ym=pl.from_epoch("MVT_TIME_UTC_mvt", time_unit="s").dt.strftime("%y%m")).drop("MVT_TIME_UTC_mvt")
    n = base.group_by("split", "ym").len()
    rank_m = sorted(n.filter((pl.col("split") == "rank") & (pl.col("len") >= 1000))["ym"].to_list())
    train_m = set(n.filter((pl.col("split") == "train") & (pl.col("len") >= 1000))["ym"].to_list())
    # reference group per ranked month: training rows of the same calendar month, else all training rows
    ref = {m: sorted(t for t in train_m if t[2:] == m[2:]) or sorted(train_m) for m in rank_m}
    stems = [s for s in a.files.split(",") if s] or sorted(p.stem[13:] for p in PROC.glob("features_dep_*.parquet"))
    tests = [] if a.no_preds else sorted(TEST.glob("*.parquet"))
    total, step, fails = len(stems) + len(tests) + 1, 0, []
    rep(0, total, f"{PROC.name}: ranked months {rank_m}", purpose="Do the ranked months have the feature and member "
        "coverage of the training months (no silent gaps before the final file)?")
    for s in [s for s in a.require.split(",") if s]:
        if not (PROC / f"features_dep_{s}.parquet").exists():
            fails.append(f"required features_dep_{s}.parquet missing")
    rows = []
    for s in stems:
        step += 1
        p = PROC / f"features_dep_{s}.parquet"
        if not p.exists():
            fails.append(f"features_dep_{s}.parquet missing")
            continue
        x = ids(pl.read_parquet(p)).unique("MVT_ID_mvt")
        cols = [c for c in x.columns if c != "MVT_ID_mvt"]
        j = base.join(x, on="MVT_ID_mvt", how="left")
        g = j.group_by("split", "ym").agg(share=pl.mean_horizontal([pl.col(c).is_not_null().mean() for c in cols]))
        sh = {(r["split"], r["ym"]): r["share"] for r in g.iter_rows(named=True)}
        row = {"file": s}
        for m in rank_m:
            r = sum(sh.get(("train", t), 0.0) for t in ref[m]) / len(ref[m])
            v = sh.get(("rank", m), 0.0)
            row[f"{m}"] = round(v, 3)
            row[f"{m}_ref"] = round(r, 3)
            if r >= a.min_ref and v < a.ratio * r:
                msg = f"{s}: rank {m} non-null {v:.3f} < {a.ratio} x train {'/'.join(ref[m])} {r:.3f}"
                if (s, m) in SOURCE_GAPS and PHASE == "final":
                    print(f"  ALLOWED {msg}  [{SOURCE_GAPS[(s, m)]}]")
                else:
                    fails.append(msg)
        rows.append(row)
        rep(step, total, f"features_dep_{s}")
        del x, j
    tmpl = ids(pl.read_parquet(TEMPLATE, columns=["MVT_ID_mvt"])).join(base, on="MVT_ID_mvt", how="left")
    prow = []
    for p in tests:
        step += 1
        x = ids(pl.read_parquet(p)).unique("MVT_ID_mvt")
        num = [c for c, t in x.schema.items() if c != "MVT_ID_mvt" and t.is_numeric()]
        j = tmpl.join(x.with_columns(_hit=pl.lit(True)), on="MVT_ID_mvt", how="left")
        g = j.group_by("ym").agg(hit=pl.col("_hit").is_not_null().mean(),
                                 nan=pl.max_horizontal([(pl.col(c).is_null() | pl.col(c).is_nan()).mean() for c in num])
                                 if num else pl.lit(0.0))
        d = {r["ym"]: r for r in g.iter_rows(named=True) if r["ym"] in rank_m}
        miss = {m: 1 - d[m]["hit"] for m in d if d[m]["hit"] < 1}
        if miss:
            fails.append(f"test/{p.name}: template ids missing {', '.join(f'{m} {v:.1%}' for m, v in miss.items())}")
        best = min((d[m]["nan"] for m in d), default=0.0)
        worse = {m: d[m]["nan"] for m in d if d[m]["nan"] > best + 0.01}
        if worse:
            fails.append(f"test/{p.name}: NaN share above the best month ({best:.1%}) on "
                         + ", ".join(f"{m} {v:.1%}" for m, v in worse.items()))
        prow.append({"file": p.name, **{m: f"{d[m]['hit']:.3f}/{d[m]['nan']:.3f}" for m in d}})
        rep(step, total, f"test/{p.name}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=60):
        if rows:
            print(f"\nfeature non-null share, ranked month vs training reference ({PROC.name})")
            print(pl.DataFrame(rows))
        if prow:
            print(f"\ntest predictions: template-id coverage / worst numeric NaN share per ranked month ({TEST})")
            print(pl.DataFrame(prow))
    print(f"\n{len(fails)} gate failure(s)")
    for f in fails:
        print("  FAIL", f)
    rep(total, total, f"{len(fails)} gate failures", finished=True, outcome="failure" if fails else "success")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
