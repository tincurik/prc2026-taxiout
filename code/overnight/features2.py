# SPDX-License-Identifier: GPL-3.0-or-later
"""Feature set 2: label-free signals from arrivals (their block times are NOT masked in ranking).

- arrival block==schedule "imputation" rates by (airport, airline) and airport in time windows
- the previous arrival at the same stand (likely the inbound aircraft): imputed flag, turnaround gap,
  its taxi-in, arrival delay, same-airline flag
- departure-side: same-stand next departure gap
Adds columns to data/processed/features_dep.parquet in place (idempotent).
"""
import numpy as np
import polars as pl

from ocommon import PROC, report

RUN = "overnight-features2"
SHIFT = np.int64(1 << 34)


def composite(codes, t):
    return codes.astype(np.int64) * SHIFT + t.astype(np.int64)


def main():
    report(RUN, 0, 4, "loading", purpose="Label-free arrival-side imputation and turnaround features")
    dep = pl.read_parquet(PROC / "features_dep.parquet")
    new_cols = [c for c in dep.columns if c.startswith(("fa_",))]
    dep = dep.drop(new_cols)
    arr = pl.read_parquet(PROC / "movements.parquet").filter(pl.col("PHASE_mvt") == "ARR").with_columns(BLOCK_TIME_UTC_mvt=pl.coalesce(pl.col("BLOCK_TIME_UTC_mvt"), pl.col("MVT_TIME_UTC_mvt") + 420)).with_columns(
        apt=pl.col("ADES_mvt"),
        airline=pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{3})", 1),
        imp=((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).abs() <= 30).cast(pl.Float64),
        a_delay=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).cast(pl.Float64),
    ).with_columns(apt_stand=pl.col("apt") + "_" + pl.col("STAND_mvt").fill_null("NA"),
                   apt_airline=pl.col("apt") + "_" + pl.col("airline").fill_null("NA"))
    dep = dep.with_columns(apt_airline=pl.col("apt") + "_" + pl.col("airline").fill_null("NA"))
    out = {}
    t_dep = dep["MVT_TIME_UTC_mvt"].to_numpy().astype(np.int64)
    ob_dep = dep["ob_est"].to_numpy().astype(np.int64)
    sched_dep = dep["SCHED_TIME_UTC_mvt"].to_numpy().astype(np.int64)

    # ---- windowed imputation rates keyed by (apt, airline) and by apt
    for key, wins in (("apt_airline", (6, 24, 72)), ("apt", (1, 3, 12))):
        cats = pl.concat([arr[key], dep[key]]).unique().sort()
        mp = dict(zip(cats.to_list(), range(len(cats))))
        ca = np.array([mp[v] for v in arr[key].to_list()])
        cd = np.array([mp[v] for v in dep[key].to_list()])
        ta = arr["BLOCK_TIME_UTC_mvt"].to_numpy().astype(np.int64)
        comp = composite(ca, ta)
        order = np.argsort(comp)
        comp = comp[order]
        imp = arr["imp"].to_numpy()[order]
        cs = np.concatenate([[0.0], np.cumsum(imp)])
        q = composite(cd, sched_dep)
        for w in wins:
            lo = np.searchsorted(comp, q - w * 3600, "left")
            hi = np.searchsorted(comp, q + w * 3600, "right")
            n = hi - lo
            out[f"fa_arrimp_{key}_{w}h"] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
            out[f"fa_arrn_{key}_{w}h"] = n
    report(RUN, 1, 4, "imputation-rate windows done")

    # ---- previous arrival at same stand before departure's estimated off-block
    cats = pl.concat([arr["apt_stand"], dep["apt_stand"]]).unique().sort()
    mp = dict(zip(cats.to_list(), range(len(cats))))
    ca = np.array([mp[v] for v in arr["apt_stand"].to_list()])
    cd = np.array([mp[v] for v in dep["apt_stand"].to_list()])
    ta = arr["BLOCK_TIME_UTC_mvt"].to_numpy().astype(np.int64)
    comp = composite(ca, ta)
    order = np.argsort(comp)
    comp = comp[order]
    q = composite(cd, np.minimum(ob_dep, sched_dep + 3600))
    j = np.searchsorted(comp, q, "right") - 1
    valid = (j >= 0) & ((comp[np.maximum(j, 0)] // SHIFT) == cd)
    jj = np.maximum(j, 0)
    gap = np.where(valid, q - comp[jj], np.nan).astype(float)
    valid &= gap < 18 * 3600
    gap = np.where(valid, gap, np.nan)
    A = {c: arr[c].to_numpy()[order] for c in ["imp", "a_delay", "TAXITIME_SEC_mvt"]}
    out["fa_prevarr_gap"] = gap
    out["fa_prevarr_imp"] = np.where(valid, A["imp"][jj], np.nan)
    out["fa_prevarr_delay"] = np.where(valid, A["a_delay"][jj], np.nan)
    out["fa_prevarr_taxiin"] = np.where(valid, A["TAXITIME_SEC_mvt"][jj], np.nan)
    al_arr = arr["airline"].fill_null("NA").to_numpy()[order]
    out["fa_prevarr_same_airline"] = np.where(valid, (al_arr[jj] == dep["airline"].fill_null("NA").to_numpy()).astype(float), np.nan)
    # previous-arrival block time relative to my schedule / takeoff (turnaround)
    out["fa_prevarr_blk_m_sched"] = np.where(valid, comp[jj] % SHIFT - sched_dep, np.nan).astype(float)
    report(RUN, 2, 4, "previous same-stand arrival done")

    # ---- same-stand departures: gap to previous/next departure off-block at the stand
    comp_d = composite(cd, ob_dep)
    od = np.argsort(comp_d); cds = comp_d[od]
    pos = np.searchsorted(cds, comp_d, "left")
    prev_ok = (pos > 0) & ((cds[np.maximum(pos - 1, 0)] // SHIFT) == cd)
    out["fa_stand_prevdep_gap"] = np.where(prev_ok, comp_d - cds[np.maximum(pos - 1, 0)], np.nan).astype(float)
    pos2 = np.searchsorted(cds, comp_d, "right")
    nxt_ok = (pos2 < len(cds)) & ((cds[np.minimum(pos2, len(cds) - 1)] // SHIFT) == cd)
    out["fa_stand_nextdep_gap"] = np.where(nxt_ok, cds[np.minimum(pos2, len(cds) - 1)] - comp_d, np.nan).astype(float)
    for k in list(out):
        if k.startswith("fa_stand_") :
            out[k] = np.where(out[k] > 48 * 3600, np.nan, out[k])
    report(RUN, 3, 4, "stand departure gaps done")
    dep = dep.drop("apt_airline").with_columns([pl.Series(k, v) for k, v in out.items()])
    dep.write_parquet(PROC / "features_dep.tmp.parquet"); import os; os.replace(PROC / "features_dep.tmp.parquet", PROC / "features_dep.parquet")
    report(RUN, 4, 4, f"features_dep.parquet now {dep.shape}", finished=True)


if __name__ == "__main__":
    main()


