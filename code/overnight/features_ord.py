# SPDX-License-Identifier: GPL-3.0-or-later
"""Label-free departure features -> data/processed/features_dep_ext_ord.parquet.

  o_dev_mvt / o_dev_flt / o_n_line  record ordering: per (split, month,
                                    airport) departures sorted by MVT, MVT_ID (FLIGHT_ID) minus the rolling median of
                                    its +-250 neighbours, over the rolling p90 - p10. FLIGHT_ID is ordered by LOBT, so
                                    this is a neighbour-normalised "how late against plan".
  gb_apt_aobt / gb_apt_eobt / gb_rwy_aobt   gate backlog: flights j at the airport
                                    (runway) with EOBT_1_j <= t < AOBT_3_j (capped 2 h), at t = own AOBT_3 / own EOBT_1.
  gb_ratio                          backlog at AOBT_3 over departures in the previous hour.
  rc_air_m1 / rc_air_m3 / rc_mvt_arvt1   route-centred NM durations: ARVT_1 - EOBT_1,
                                    ARVT_3 - AOBT_3 and MVT - ARVT_1 minus the train-split median per (ADEP, ADES, type),
                                    falling back to (ADEP, ADES).
  lfpg_works                        LFPG departure on a day with > 10% of departures on 09L/27R (runway works
                                    16 Jul - ~10 Dec 2025; 0 in both test months).
"""
import numpy as np
import pandas as pd
import polars as pl

from ocommon import PROC, report

RUN = "features-ord"


def roll_dev(ids, w=250):
    s = pd.Series(ids, dtype=float)
    r = s.rolling(2 * w + 1, center=True, min_periods=50)
    med, p90, p10 = r.median(), r.quantile(0.9), r.quantile(0.1)
    return ((s - med) / np.maximum(p90 - p10, 1)).to_numpy(), r.count().to_numpy()


def backlog(eobt, aobt, t):
    """count of j with eobt_j <= t < aobt_j (only eobt < aobt <= eobt + 2 h)."""
    ok = np.isfinite(eobt) & np.isfinite(aobt) & (aobt > eobt)
    e = np.sort(eobt[ok])
    a = np.sort(np.minimum(aobt[ok], eobt[ok] + 7200))
    out = np.full(len(t), np.nan)
    f = np.isfinite(t)
    out[f] = np.searchsorted(e, t[f], side="right") - np.searchsorted(a, t[f], side="right")
    return out


def main():
    report(RUN, 0, 4, "loading", purpose="Build record-ordering, gate-backlog, route-centred and LFPG-works features")
    cols = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_1_flt", "ARVT_3_flt",
            "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "split", "apt", "month"]
    d = pl.read_parquet(PROC / "features_dep.parquet", columns=cols).to_pandas()
    for c in ["MVT_TIME_UTC_mvt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_1_flt", "ARVT_3_flt", "FLIGHT_ID_mvt", "MVT_ID_mvt"]:
        d[c] = d[c].astype(float)
    out = pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt.astype("int64")})
    for c in ["o_dev_mvt", "o_dev_flt", "o_n_line", "gb_apt_aobt", "gb_apt_eobt", "gb_rwy_aobt"]:
        out[c] = np.nan

    # record ordering
    for (sp, mo, apt), g in d.groupby(["split", "month", "apt"]):
        g = g.sort_values(["MVT_TIME_UTC_mvt", "MVT_ID_mvt"])
        dm, n = roll_dev(g.MVT_ID_mvt.to_numpy())
        out.loc[g.index, "o_dev_mvt"] = dm
        out.loc[g.index, "o_n_line"] = n
        gf = g[g.FLIGHT_ID_mvt.notna()]
        if len(gf) > 100:
            df_, _ = roll_dev(gf.FLIGHT_ID_mvt.to_numpy())
            out.loc[gf.index, "o_dev_flt"] = df_
    report(RUN, 1, 4, "record ordering done")

    # gate backlog (all departures of both splits: EOBT/AOBT are not masked)
    for apt, g in d.groupby("apt"):
        e, a = g.EOBT_1_flt.to_numpy(), g.AOBT_3_flt.to_numpy()
        out.loc[g.index, "gb_apt_aobt"] = backlog(e, a, a)
        self_in = (np.isfinite(e) & np.isfinite(a) & (a > e)).astype(float)
        out.loc[g.index, "gb_apt_eobt"] = backlog(e, a, e) - self_in
        for rwy, h in g.groupby("RUNWAY_mvt"):
            out.loc[h.index, "gb_rwy_aobt"] = backlog(h.EOBT_1_flt.to_numpy(), h.AOBT_3_flt.to_numpy(), h.AOBT_3_flt.to_numpy())
    dep = pl.read_parquet(PROC / "features_dep.parquet", columns=["dep_prev_60"]).to_pandas().dep_prev_60.to_numpy(float)
    out["gb_ratio"] = out.gb_apt_aobt / np.maximum(dep, 1)
    report(RUN, 2, 4, "gate backlog done")

    # route-centred NM durations (medians from the train split only)
    d["air_m1"] = d.ARVT_1_flt - d.EOBT_1_flt
    d["air_m3"] = d.ARVT_3_flt - d.AOBT_3_flt
    d["mvt_arvt1"] = d.MVT_TIME_UTC_mvt - d.ARVT_1_flt
    tr = d[d.split == "train"]
    for c, name in [("air_m1", "rc_air_m1"), ("air_m3", "rc_air_m3"), ("mvt_arvt1", "rc_mvt_arvt1")]:
        k3 = ["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt"]
        m3 = tr.groupby(k3)[c].agg(["median", "count"]).reset_index()
        m3 = m3[m3["count"] >= 5].drop(columns="count").rename(columns={"median": "m3"})
        m2 = tr.groupby(["ADEP_mvt", "ADES_mvt"])[c].median().rename("m2").reset_index()
        x = d[k3 + [c]].merge(m3, on=k3, how="left").merge(m2, on=["ADEP_mvt", "ADES_mvt"], how="left")
        out[name] = (x[c] - x.m3.fillna(x.m2)).clip(-3600, 3600).to_numpy()
    report(RUN, 3, 4, "route-centred durations done")

    # LFPG works configuration
    lf = d[d.apt == "LFPG"].copy()
    lf["day"] = (lf.MVT_TIME_UTC_mvt // 86400).astype(int)
    share = lf.groupby("day").RUNWAY_mvt.apply(lambda r: r.isin(["09L", "27R"]).mean())
    out["lfpg_works"] = 0.0
    out.loc[lf.index, "lfpg_works"] = (lf.day.map(share) > 0.10).astype(float).to_numpy()

    out.to_parquet(PROC / "features_dep_ext_ord.parquet", index=False)
    desc = out.drop(columns="MVT_ID_mvt").describe().T.round(3)
    print(desc.to_string())
    print("lfpg_works by split/month:", out.assign(sp=d.split, mo=d.month, apt=d.apt)[d.apt == "LFPG"].groupby(["sp", "mo"]).lfpg_works.mean().round(2).to_dict())
    report(RUN, 4, 4, f"wrote features_dep_ext_ord.parquet ({len(out):,} rows)", finished=True)


if __name__ == "__main__":
    main()
