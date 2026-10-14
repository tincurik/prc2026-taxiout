# SPDX-License-Identifier: GPL-3.0-or-later
"""Rehearsal of the blind final: build a 'final' split from labelled 2025 months.

The real final (final_ranking.parquet) is Jan/Jul 2026 plus two further months, departures' block time and taxi time masked,
take-off times jittered. This builds the same shape from files we hold labels for, so the whole pipeline can be run and
scored end to end before the real months are known (EC_PHASE=pseudo, see overnight/ocommon.py):

  data/pseudo/raw/training_*.parquet       symlinks to the training months NOT held out
  data/pseudo/raw/final_ranking.parquet    ranking.parquet (Jan/Jul 2026) + the held-out months with every DEP's BLOCK_TIME_UTC_mvt
                                           and TAXITIME_SEC_mvt nulled, exactly the masking ranking.parquet has
  data/pseudo/raw/final_submitting.parquet MVT_ID_mvt + null TAXITIME_SEC_mvt for every departure of final_ranking
  data/pseudo/truth.parquet                the held-out months' true taxi times (for scoring the rehearsal)

  python scripts/make_pseudo_final.py --months 2025-04,2025-10
  python scripts/make_pseudo_final.py --phase pseudo2 --months 2025-02,2025-06   # the real final months, one year earlier
"""
import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="2025-04,2025-10", help="two held-out 2025 months, YYYY-MM")
    ap.add_argument("--phase", default="pseudo", help="EC_PHASE name; files go to data/<phase>/ (pseudo2 = Feb+Jun 2025)")
    a = ap.parse_args()
    months = a.months.split(",")
    OUT = ROOT / f"data/{a.phase}/raw"
    OUT.mkdir(parents=True, exist_ok=True)
    held = []
    for p in sorted(RAW.glob("training_*.parquet")):
        month = p.name.split("_")[1][:7]
        link = OUT / p.name
        if link.is_symlink() or link.exists():
            link.unlink()
        if month in months:
            held.append(p)
        else:
            os.symlink(os.path.relpath(p, OUT), link)
    assert len(held) == len(months), f"months {months} not all found in {RAW}"
    ranking = pq.read_table(RAW / "ranking.parquet")
    parts, truth = [ranking], []
    for p in held:
        t = pq.read_table(p)
        assert t.schema.equals(ranking.schema), p.name
        dep = pc.equal(t["PHASE_mvt"], "DEP")
        truth.append(t.filter(dep).select(["MVT_ID_mvt", "TAXITIME_SEC_mvt"]))
        for col in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"):
            i = t.schema.get_field_index(col)
            t = t.set_column(i, t.schema.field(col), pc.if_else(dep, pa.scalar(None, t.schema.field(col).type), t[col]))
        parts.append(t)
    final = pa.concat_tables(parts)
    pq.write_table(final, OUT / "final_ranking.parquet")
    dep_ids = final.filter(pc.equal(final["PHASE_mvt"], "DEP"))["MVT_ID_mvt"]
    tmpl = pa.table({"MVT_ID_mvt": dep_ids, "TAXITIME_SEC_mvt": pa.nulls(len(dep_ids), pa.int32())})
    pq.write_table(tmpl, OUT / "final_submitting.parquet")
    pq.write_table(pa.concat_tables(truth), OUT.parent / "truth.parquet")
    (OUT.parent / "pseudo.json").write_text(json.dumps({"held_out_months": months, "rows": final.num_rows,
                                                        "template_rows": tmpl.num_rows}, indent=1))
    print(f"held out {months}: final_ranking {final.num_rows:,} rows, template {tmpl.num_rows:,} departures, "
          f"{sum(len(t) for t in truth):,} truth rows")


if __name__ == "__main__":
    main()
