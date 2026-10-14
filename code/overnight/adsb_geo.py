# SPDX-License-Identifier: GPL-3.0-or-later
"""First-heard position of each ADS-B-matched departure (the 'taxi map' feature): where on the airport the aircraft was
when first heard, in metres east/north of the airport reference and of its learned stand position, plus bearing.
Label-free (positions only). Source: stage-1 cache outputs/overnight/cache/adsb2/<day>.parquet (lat0/lon0), stand
coordinates data/processed/adsb_stand_coords_adsb2full.parquet.
Output: data/processed/features_dep_adsbgeo.parquet (MVT_ID_mvt, g_x0, g_y0, g_dxs, g_dys, g_bear, g_sx, g_sy)
  python adsb_geo.py
"""
import glob

import numpy as np
import pandas as pd

from ocommon import OUTD, PROC

REF = {"EDDF": (50.033333, 8.570556), "EDDM": (48.353783, 11.786086), "EGLL": (51.4775, -0.461389), "EHAM": (52.308613, 4.763889),
       "LEBL": (41.297078, 2.078464), "LEMD": (40.471926, -3.56264), "LFPG": (49.008960, 2.554117), "LIRF": (41.804532, 12.251998),
       "LSZH": (47.458056, 8.548056), "LTFM": (41.274874, 28.732136)}


def xy(lat, lon, apt):
    la0 = apt.map(lambda a: REF[a][0]).to_numpy(float); lo0 = apt.map(lambda a: REF[a][1]).to_numpy(float)
    return (lon - lo0) * 111320.0 * np.cos(np.radians(la0)), (lat - la0) * 110540.0


def main():
    C = ["MVT_ID_mvt", "phase", "apt", "stand", "lat0", "lon0"]
    parts = [x[C] for x in (pd.read_parquet(f) for f in sorted(glob.glob(str(OUTD / "cache/adsb2/*.parquet")))) if "lat0" in x.columns]  # empty days have no columns
    d = pd.concat(parts, ignore_index=True)
    d = d[(d.phase == "DEP") & d.lat0.notna()].drop_duplicates("MVT_ID_mvt")
    s = pd.read_parquet(PROC / "adsb_stand_coords_adsb2full.parquet")
    d = d.merge(s[["apt", "stand", "s_lat", "s_lon"]], on=["apt", "stand"], how="left")
    d["g_x0"], d["g_y0"] = xy(d.lat0.to_numpy(), d.lon0.to_numpy(), d.apt)
    d["g_sx"], d["g_sy"] = xy(d.s_lat.to_numpy(), d.s_lon.to_numpy(), d.apt)
    d["g_dxs"], d["g_dys"] = d.g_x0 - d.g_sx, d.g_y0 - d.g_sy
    d["g_bear"] = np.degrees(np.arctan2(d.g_dxs, d.g_dys))
    out = d[["MVT_ID_mvt", "g_x0", "g_y0", "g_dxs", "g_dys", "g_bear", "g_sx", "g_sy"]]
    out.to_parquet(PROC / "features_dep_adsbgeo.parquet", index=False)
    print(len(out), out.describe().round(0).to_string())


if __name__ == "__main__":
    main()
