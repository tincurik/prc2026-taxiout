# SPDX-License-Identifier: GPL-3.0-or-later
"""fs3: label-free stand/runway "distance" features from rolling windows -> data/processed/features_dep_fs3.parquet.

Written to a separate file (joined by `mix.py --fs3 1`) so runs already queued keep a fixed feature set.
- arrivals' taxi-in at the same stand (ranking arrivals keep BLOCK and taxi-in), windows +-3 d and +-30 d
- other departures' MVT - AOBT_3 (label-free taxi proxy) at the same stand and same stand x runway, excluding self
- local-midnight crossing between SCHED and MVT, local scheduled hour
"""
import numpy as np
import polars as pl

from features import TZ
from ocommon import PROC, report

RUN = "overnight-features3"
DAY = 86400


def keyed_window_mean(key_src, t_src, v_src, key_q, t_q, lo, hi, self_v=None):
    """Mean of v_src over rows with the same key and t in [t_q+lo, t_q+hi]; optionally subtract the query row itself."""
    o = np.lexsort((t_src, key_src))
    ks, ts, vs = key_src[o], t_src[o], v_src[o]
    has = ~np.isnan(vs)
    comp = ks.astype(np.int64) * (1 << 34) + ts.astype(np.int64)
    cs = np.concatenate([[0.0], np.cumsum(np.where(has, vs, 0.0))])
    ch = np.concatenate([[0.0], np.cumsum(has)])
    q = key_q.astype(np.int64) * (1 << 34)
    a = np.searchsorted(comp, q + t_q.astype(np.int64) + lo, "left")
    b = np.searchsorted(comp, q + t_q.astype(np.int64) + hi, "right")
    s, n = cs[b] - cs[a], ch[b] - ch[a]
    if self_v is not None:
        sh = ~np.isnan(self_v)
        s = s - np.where(sh, self_v, 0.0)
        n = n - sh
    return np.where(n > 0, s / np.maximum(n, 1), np.nan), n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="none", choices=["none", "trunc_val", "monthlocal"],
                    help="restrict rolling windows to month blocks: trunc_val = only the CV validation months (Jan/Jul) and ranking rows "
                         "see their own calendar month (mimics the test-time truncation); monthlocal = every row sees only its own month")
    args = ap.parse_args()
    report(RUN, 0, 4, "loading", purpose="Stand/runway taxi-distance proxies from arrivals' taxi-in and neighbours' MVT-AOBT")
    mv = pl.read_parquet(PROC / "movements.parquet").with_columns(
        apt=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")))
    mv = mv.with_columns(stand_key=(pl.col("apt") + "|" + pl.col("STAND_mvt").fill_null("NA")),
                         rwy_key=(pl.col("apt") + "|" + pl.col("RUNWAY_mvt").fill_null("NA")))
    mv = mv.with_columns(stand_code=pl.col("stand_key").cast(pl.Categorical).to_physical().cast(pl.Int64),
                         sr_code=(pl.col("stand_key") + "#" + pl.col("rwy_key")).cast(pl.Categorical).to_physical().cast(pl.Int64))
    if args.block != "none":
        mon = pl.from_epoch("MVT_TIME_UTC_mvt", time_unit="s").dt.month().cast(pl.Int64)
        blk = mon + pl.when(pl.col("split") == "train").then(0).otherwise(20)
        if args.block == "trunc_val":
            blk = pl.when((pl.col("split") != "train") | mon.is_in([1, 7])).then(blk).otherwise(0)
        mv = mv.with_columns(blk=blk).with_columns(stand_code=pl.col("stand_code") * 64 + pl.col("blk"),
                                                    sr_code=pl.col("sr_code") * 64 + pl.col("blk"))
    dep = mv.filter(pl.col("PHASE_mvt") == "DEP")
    arr = mv.filter(pl.col("PHASE_mvt") == "ARR")
    out = {"MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy()}
    tq = dep["MVT_TIME_UTC_mvt"].to_numpy().astype(np.int64)
    sq = dep["stand_code"].to_numpy().astype(np.int64)
    srq = dep["sr_code"].to_numpy().astype(np.int64)

    # arrivals' taxi-in at the stand
    ta = arr["MVT_TIME_UTC_mvt"].to_numpy().astype(np.int64)
    va = np.clip(arr["TAXITIME_SEC_mvt"].to_numpy().astype(float), 0, 1800)
    sa = arr["stand_code"].to_numpy().astype(np.int64)
    for name, w in (("3d", 3 * DAY), ("30d", 30 * DAY)):
        m, n = keyed_window_mean(sa, ta, va, sq, tq, -w, w)
        out[f"f3_arrstand_taxiin_{name}"] = m
        out[f"f3_arrstand_n_{name}"] = n
    report(RUN, 1, 4, "arrival stand taxi-in done")

    # departures' label-free MVT - AOBT_3 at the stand and stand x runway, excluding self
    d = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).to_numpy().astype(float)
    d = np.where((d >= 0) & (d <= 3600), d, np.nan)
    for name, w in (("3d", 3 * DAY), ("30d", 30 * DAY)):
        m, n = keyed_window_mean(sq, tq, d, sq, tq, -w, w, self_v=d)
        out[f"f3_depstand_daobt_{name}"] = m
    for name, w in (("7d", 7 * DAY), ("60d", 60 * DAY)):
        m, n = keyed_window_mean(srq, tq, d, srq, tq, -w, w, self_v=d)
        out[f"f3_depsr_daobt_{name}"] = m
        out[f"f3_depsr_n_{name}"] = n
    report(RUN, 2, 4, "departure stand/runway proxies done")

    # stand x runway: arrivals at stand minus departures at stand x runway (difference in proxies)
    out["f3_sr_minus_arrstand_30d"] = out["f3_depsr_daobt_60d"] - out["f3_arrstand_taxiin_30d"]

    # local-midnight crossing + local scheduled hour
    apt = dep["apt"].to_numpy()
    cross = np.zeros(len(dep), dtype=np.int8)
    sh = np.zeros(len(dep), dtype=np.int8)
    for a, tz in TZ.items():
        m = apt == a
        sub = dep.filter(pl.col("apt") == a).select(
            md=pl.from_epoch("MVT_TIME_UTC_mvt", time_unit="s").dt.replace_time_zone("UTC").dt.convert_time_zone(tz).dt.date(),
            sd=pl.from_epoch("SCHED_TIME_UTC_mvt", time_unit="s").dt.replace_time_zone("UTC").dt.convert_time_zone(tz),
        )
        cross[m] = (sub["md"] - sub["sd"].dt.date()).dt.total_days().clip(-1, 2).to_numpy()
        sh[m] = sub["sd"].dt.hour().to_numpy()
    out["f3_cross_mid"] = cross
    out["f3_sched_local_hour"] = sh
    res = pl.DataFrame(out)
    res.write_parquet(PROC / ("features_dep_fs3.parquet" if args.block == "none" else f"features_dep_fs3_{args.block}.parquet"))
    report(RUN, 4, 4, f"wrote features_dep_fs3.parquet {res.shape}", finished=True)
    print(res.describe())


if __name__ == "__main__":
    main()
