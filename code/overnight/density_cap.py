# SPDX-License-Identifier: GPL-3.0-or-later
"""Density cap : the Jun/Jul 2026 ADS-B tracks are 2-4x denser than any 2025 month, so the
stackers see a2_npts / a2_maxgap values outside their training range.  The cap clips those two features, per airport, to the 2025 range:
a2_npts <= 2025 p99, a2_maxgap >= 2025 p01 (denser tracks have SMALLER gaps) and <= 2025 p99.  Only 2026 rows are touched (2025 rows keep their values).
caps json = {"npts_hi": {apt: v}, "gap_lo": {apt: v}, "gap_hi": {apt: v}}, written by scripts/density_cap/caps.py from the 2025 matched rows.  An optional A/B switch: the shipped chain runs without it (uncapped features).
  features_adsb2.py   ADSB_CAP=<json> ... --assemble_only 1 --out adsb2mcap   (stage 3 of the cached stage-1 days)
  adsb_direct.py      --cap <json>                                          (test rows of the month-complete ADS-B model)
No label is read anywhere."""
import json

import numpy as np

T2026 = 1767225600  # 2026-01-01 00:00 UTC


def load(path):
    return json.load(open(path))


def clip(npts, gap, apt, mvt, caps):
    """npts, gap: float arrays; apt: str array; mvt: epoch seconds.  Returns clipped copies and the number of rows changed."""
    npts, gap = np.asarray(npts, float).copy(), np.asarray(gap, float).copy()
    apt = np.asarray(apt).astype(str)
    new = np.asarray(mvt, float) >= T2026
    hi = np.array([caps["npts_hi"].get(a, np.inf) for a in apt])
    lo = np.array([caps["gap_lo"].get(a, -np.inf) for a in apt])
    gh = np.array([caps["gap_hi"].get(a, np.inf) for a in apt])
    n2 = np.where(new, np.minimum(npts, hi), npts)
    g2 = np.where(new, np.minimum(np.maximum(gap, lo), gh), gap)
    changed = int(((n2 != npts) | (g2 != gap)).sum())
    return n2, g2, changed
