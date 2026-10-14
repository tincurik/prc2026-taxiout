# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-airport 2025 caps for the density cap (overnight/density_cap.py): p99 of a2_npts, p01 / p99 of a2_maxgap over the 2025 rows with an ADS-B track
(features_dep_adsb2m, split == train).  Writes outputs/density_cap/caps.json.   EC_PHASE=final python scripts/density_cap/caps.py [processed dir]
Optional: the density cap is an A/B switch (overnight/density_cap.py), not part of the shipped chain.  The 2025 rows are the same in every phase."""
import json
import os
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ("data/processed_" + os.environ["EC_PHASE"] if os.environ.get("EC_PHASE") else "data/processed")
f = pl.read_parquet(PROC / "features_dep_adsb2m.parquet", columns=["MVT_ID_mvt", "a2_npts", "a2_maxgap"])
b = pl.read_parquet(PROC / "features_dep.parquet", columns=["MVT_ID_mvt", "apt", "split"])
d = b.join(f, on="MVT_ID_mvt").filter((pl.col("split") == "train") & pl.col("a2_npts").is_not_null()).to_pandas()
g = d.groupby("apt")
caps = {"npts_hi": g.a2_npts.quantile(.99).to_dict(), "gap_lo": g.a2_maxgap.quantile(.01).to_dict(), "gap_hi": g.a2_maxgap.quantile(.99).to_dict(),
        "n_rows_2025": g.size().to_dict(), "source": str(PROC)}
out = ROOT / "outputs/density_cap/caps.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(caps, indent=1))
print(json.dumps({k: {a: round(v, 1) for a, v in caps[k].items()} for k in ("npts_hi", "gap_lo", "gap_hi")}, indent=1))
