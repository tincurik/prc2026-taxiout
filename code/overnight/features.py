# SPDX-License-Identifier: GPL-3.0-or-later
"""Label-free feature builder for all departures (train 2025 + ranking 2026).

Every feature uses only columns present in the ranking inputs: own timestamps, categorical attributes,
and other movements' timestamps / arrival taxi-in times. Departure labels and DEP block times are never
used (they are masked in ranking), so the same code applies to both splits.
"""
import numpy as np
import polars as pl

from ocommon import PROC, report

TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam",
      "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LIRF": "Europe/Rome",
      "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}
RUN = "overnight-features"


def win_count(sorted_t, q, lo, hi):
    """Number of sorted_t in [q+lo, q+hi)."""
    return np.searchsorted(sorted_t, q + hi, "left") - np.searchsorted(sorted_t, q + lo, "left")


def win_sum(sorted_t, csum, q, lo, hi):
    a = np.searchsorted(sorted_t, q + lo, "left")
    b = np.searchsorted(sorted_t, q + hi, "left")
    return csum[b] - csum[a], b - a


def group_features(t, v_daobt, ob_est, is_self_group=True):
    """Within one airport's departures sorted by takeoff time t."""
    out = {}
    for w in (5, 15, 30, 60):
        s = w * 60
        out[f"dep_prev_{w}"] = win_count(t, t, -s, 0)           # takeoffs in [t-w, t) (excl self: self at t not < t)
        out[f"dep_next_{w}"] = win_count(t, t, 1, s + 1)         # takeoffs in (t, t+w]
    # pushed back but not yet airborne at my takeoff / at my off-block
    ob_sorted = np.sort(ob_est)
    out["queue_at_mvt"] = np.searchsorted(ob_sorted, t, "right") - np.searchsorted(t, t, "right")
    out["queue_at_ob"] = np.searchsorted(ob_sorted, ob_est, "right") - np.searchsorted(t, ob_est, "right")
    # takeoffs between my off-block estimate and my takeoff
    out["dep_during_taxi"] = np.searchsorted(t, t, "left") - np.searchsorted(t, ob_est, "left")
    # neighbours' MVT-AOBT (label-free taxi proxy), excluding self
    vv = np.nan_to_num(v_daobt, nan=0.0)
    has = (~np.isnan(v_daobt)).astype(float)
    cs = np.concatenate([[0.0], np.cumsum(vv)])
    ch = np.concatenate([[0.0], np.cumsum(has)])
    idx = np.arange(len(t))
    for w in (15, 30, 60):
        s = w * 60
        a = np.searchsorted(t, t - s, "left")
        b = np.searchsorted(t, t + s, "right")
        sm = cs[b] - cs[a] - vv
        n = ch[b] - ch[a] - has
        out[f"nb_daobt_pm{w}"] = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
        # past-only window
        sm = cs[idx] - cs[a]
        n = ch[idx] - ch[a]
        out[f"nb_daobt_prev{w}"] = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
    for k in (5, 20):
        a = np.maximum(idx - k, 0)
        b = np.minimum(idx + k + 1, len(t))
        sm = cs[b] - cs[a] - vv
        n = ch[b] - ch[a] - has
        out[f"nb_daobt_k{k}"] = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
    return out


def main():
    report(RUN, 0, 6, "loading movements", purpose="Build label-free departure features (timestamps, traffic, queues)")
    mv = pl.read_parquet(PROC / "movements.parquet").with_columns(
        apt=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")))
    dep = mv.filter(pl.col("PHASE_mvt") == "DEP")
    arr = mv.filter(pl.col("PHASE_mvt") == "ARR")

    T = pl.col("MVT_TIME_UTC_mvt")
    dep = dep.with_columns(
        d_aobt=T - pl.col("AOBT_3_flt"), d_eobt=T - pl.col("EOBT_1_flt"), d_sched=T - pl.col("SCHED_TIME_UTC_mvt"),
        d_lobt=T - pl.col("LOBT_flt"), d_iobt=T - pl.col("IOBT_flt"),
        aobt_m_sched=pl.col("AOBT_3_flt") - pl.col("SCHED_TIME_UTC_mvt"),
        eobt_m_sched=pl.col("EOBT_1_flt") - pl.col("SCHED_TIME_UTC_mvt"),
        aobt_m_eobt=pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt"),
        eobt_m_iobt=pl.col("EOBT_1_flt") - pl.col("IOBT_flt"),
        air_m1=pl.col("ARVT_1_flt") - pl.col("EOBT_1_flt"),
        air_m3=pl.col("ARVT_3_flt") - pl.col("AOBT_3_flt"),
        arvt3_m_mvt=pl.col("ARVT_3_flt") - T,
        arvt3_m_arvt1=pl.col("ARVT_3_flt") - pl.col("ARVT_1_flt"),
        mvt_sec=T % 60, sched_sec=pl.col("SCHED_TIME_UTC_mvt") % 60,
        has_flt=pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Int8),
        month=pl.from_epoch(T).dt.month(),
        dow=pl.from_epoch(T).dt.weekday(),
        airline=pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{3})", 1),
        stand_pre=pl.col("STAND_mvt").str.extract(r"^([A-Za-z]*\d?)", 1),
        apt_rwy=pl.col("apt") + "_" + pl.col("RUNWAY_mvt").fill_null("NA"),
        apt_stand=pl.col("apt") + "_" + pl.col("STAND_mvt").fill_null("NA"),
        callsign_mismatch=(pl.col("CALLSIGN_flt").is_not_null() & (pl.col("AIRCRAFT_TYPE_mvt") != pl.col("AIRCRAFT_TYPE_flt"))).cast(pl.Int8),
        ades_mismatch=(pl.col("ADES_flt").is_not_null() & (pl.col("ADES_mvt") != pl.col("ADES_flt"))).cast(pl.Int8),
    )
    # local time of day at the airport
    parts = []
    for apt, tz in TZ.items():
        sub = dep.filter(pl.col("apt") == apt).with_columns(lt=pl.from_epoch(T).dt.replace_time_zone("UTC").dt.convert_time_zone(tz))
        parts.append(sub.with_columns(local_hour=pl.col("lt").dt.hour(),
                                      local_min=pl.col("lt").dt.hour().cast(pl.Int32) * 60 + pl.col("lt").dt.minute().cast(pl.Int32)).drop("lt"))
    dep = pl.concat(parts)
    report(RUN, 1, 6, "base features done")

    # ---- traffic features per airport
    dep = dep.sort("apt", "MVT_TIME_UTC_mvt")
    arr = arr.sort("apt", "MVT_TIME_UTC_mvt")
    offset = dep.group_by("apt").agg(off=pl.col("d_aobt").median())
    dep = dep.join(offset, on="apt", how="left").with_columns(
        ob_est=pl.coalesce(pl.col("AOBT_3_flt"), T - pl.col("off")).cast(pl.Int64))
    feats = {}
    for apt in TZ:
        m = (dep["apt"] == apt).to_numpy()
        sub = dep.filter(pl.col("apt") == apt)
        t = sub["MVT_TIME_UTC_mvt"].to_numpy().astype(np.int64)
        v = sub["d_aobt"].cast(pl.Float64).to_numpy().copy()
        v = np.where((v < 0) | (v > 3600), np.nan, v)
        f = group_features(t, v, sub["ob_est"].to_numpy())
        # arrivals at same airport
        a = arr.filter(pl.col("apt") == apt)
        at = a["MVT_TIME_UTC_mvt"].to_numpy().astype(np.int64)
        ab = np.sort(a["BLOCK_TIME_UTC_mvt"].fill_null(a["MVT_TIME_UTC_mvt"] + 420).to_numpy().astype(np.int64))
        ti = a["TAXITIME_SEC_mvt"].to_numpy().astype(float)
        ti = np.where((ti < 0) | (ti > 3600) | np.isnan(ti), np.nan, ti)
        for w in (15, 30, 60):
            f[f"arr_pm{w}"] = win_count(at, t, -w * 60, w * 60)
        f["arr_taxiing_at_mvt"] = np.searchsorted(at, t, "right") - np.searchsorted(ab, t, "right")
        ob = sub["ob_est"].to_numpy()
        f["arr_taxiing_at_ob"] = np.searchsorted(at, ob, "right") - np.searchsorted(ab, ob, "right")
        f["arr_during_taxi"] = np.searchsorted(at, t, "left") - np.searchsorted(at, ob, "left")
        cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(ti))])
        ch = np.concatenate([[0.0], np.cumsum(~np.isnan(ti))])
        for w in (30, 60):
            lo = np.searchsorted(at, t - w * 60, "left"); hi = np.searchsorted(at, t + w * 60, "left")
            n = ch[hi] - ch[lo]
            f[f"arr_taxiin_pm{w}"] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
        # daily volume (UTC day) at the airport
        day = t // 86400
        _, inv, cnt = np.unique(day, return_inverse=True, return_counts=True)
        f["dep_day_cnt"] = cnt[inv]
        for k, val in f.items():
            feats.setdefault(k, []).append(val)
    dep = dep.with_columns([pl.Series(k, np.concatenate(v)) for k, v in feats.items()])
    report(RUN, 3, 6, "airport traffic features done")

    # ---- same-runway features
    dep = dep.sort("apt_rwy", "MVT_TIME_UTC_mvt")
    rf = {}
    for key, sub in dep.group_by("apt_rwy", maintain_order=True):
        t = sub["MVT_TIME_UTC_mvt"].to_numpy().astype(np.int64)
        v = sub["d_aobt"].cast(pl.Float64).to_numpy().copy()
        v = np.where((v < 0) | (v > 3600), np.nan, v)
        f = {}
        f["rwy_prev_10"] = win_count(t, t, -600, 0)
        f["rwy_next_10"] = win_count(t, t, 1, 601)
        f["rwy_prev_30"] = win_count(t, t, -1800, 0)
        gap_prev = np.diff(t, prepend=t[0] - 99999); gap_next = np.diff(t, append=t[-1] + 99999)
        f["rwy_gap_prev"] = np.minimum(gap_prev, 7200); f["rwy_gap_next"] = np.minimum(gap_next, 7200)
        ob = sub["ob_est"].to_numpy()
        f["rwy_queue_at_mvt"] = np.searchsorted(np.sort(ob), t, "right") - np.searchsorted(t, t, "right")
        f["rwy_dep_during_taxi"] = np.searchsorted(t, t, "left") - np.searchsorted(t, ob, "left")
        vv = np.nan_to_num(v); has = (~np.isnan(v)).astype(float)
        cs = np.concatenate([[0.0], np.cumsum(vv)]); ch = np.concatenate([[0.0], np.cumsum(has)])
        a = np.searchsorted(t, t - 1800, "left"); b = np.searchsorted(t, t + 1800, "right")
        n = ch[b] - ch[a] - has
        f["rwy_nb_daobt_pm30"] = np.where(n > 0, (cs[b] - cs[a] - vv) / np.maximum(n, 1), np.nan)
        for k, val in f.items():
            rf.setdefault(k, []).append(val)
    dep = dep.with_columns([pl.Series(k, np.concatenate(v)) for k, v in rf.items()])
    # share of the airport's takeoffs in +-60 min using this runway (runway configuration proxy)
    dep = dep.with_columns(rwy_share=(pl.col("rwy_prev_30") + 1) / (pl.col("dep_prev_30") + 1))
    report(RUN, 5, 6, "runway features done")

    keep_drop = ["BLOCK_TIME_UTC_mvt", "off", "PHASE_mvt"]
    dep = dep.drop([c for c in keep_drop if c in dep.columns]).sort("MVT_TIME_UTC_mvt")
    dep.write_parquet(PROC / "features_dep.parquet")
    report(RUN, 6, 6, f"wrote features_dep.parquet {dep.shape}", finished=True)


if __name__ == "__main__":
    main()
