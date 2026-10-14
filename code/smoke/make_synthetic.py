# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic stand-in for the challenge files, for the CPU smoke test only (it carries no information about the real data).

Same schema as the organisers' files (25 columns, MVT_ID / FLIGHT_ID doubles, UTC microsecond timestamps, int32 taxi time) and the same
masking: in the ranking / final_ranking files every departure's BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt are null.  A few days of each month,
the ten airports with their own mean taxi time, a runway and a stand effect, congestion, a lognormal tail and about 5 % of departures
whose off-block is a copy of the schedule, so every stage has something to learn.

  python smoke/make_synthetic.py --out data/raw [--days 8] [--per-day 18] [--seed 0] [--zip files_list.zip]
writes training_2025-MM-01_<next>.parquet x 12, ranking.parquet, submitting.parquet (Jan + Jul 2026),
final_ranking.parquet, final_submitting.parquet (Jan, Feb, Jun, Jul 2026); with --zip also the archive scripts/import_data.py reads.
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
BASE = dict(zip(APTS, [900, 800, 1000, 950, 700, 750, 900, 650, 600, 1100]))   # mean taxi-out seconds per airport (made up)
OTHER = ["EGKK", "LOWW", "EBBR", "EKCH", "LPPT", "LGAV", "EPWA", "EIDW", "KJFK", "OMDB"]
RUNWAYS = {a: [f"{r:02d}{s}" for r in (7, 9, 16, 25, 27) for s in ("L", "R")][:6] for a in APTS}
TYPES = [("A320", "M"), ("A21N", "M"), ("B738", "M"), ("E190", "M"), ("A333", "H"), ("B77W", "H"), ("A388", "H"), ("CRJ9", "M")]
AIRLINES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]   # made-up three-letter codes
SEG = ["Mainline", "Regional", "Low-cost"]
TS = pa.timestamp("us", tz="UTC")
TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]


def month_rows(rng, year, month, days, per_day, first_id):
    """Departures and arrivals of the first `days` days of one month; returns the 25-column frame (labels present)."""
    rows = []
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    for apt_i, apt in enumerate(APTS):
        n = days * per_day
        day = rng.integers(0, days, n)
        hour = np.clip(rng.normal(13, 4.5, n), 0, 23.99)
        tmin = start + pd.to_timedelta(day, "D") + pd.to_timedelta(hour * 3600, "s")
        wave = np.sin(hour / 24 * 2 * np.pi) * 0.5 + 1
        rwy = rng.integers(0, 6, n)
        stand = rng.integers(1, 80, n)
        ty = rng.integers(0, len(TYPES), n)
        delay = np.where(rng.random(n) < 0.25, rng.exponential(1500, n), 0.0)
        taxi = (BASE[apt] * (0.7 + 0.5 * wave) + rwy * 25 + (stand % 7) * 18 + np.array([TYPES[t][1] == "H" for t in ty]) * 120
                + rng.lognormal(4.5, 0.8, n)).round()
        far = (rng.random(n) < 0.15) if apt == "LIRF" else np.zeros(n, bool)   # one airport has late departures with no plan, many of them copies
        delay = np.where(far, rng.uniform(4000, 15000, n), delay)
        sched = tmin - pd.to_timedelta(delay, "s")
        block = tmin - pd.to_timedelta(taxi, "s")
        copy = (rng.random(n) < 0.05) | (far & (rng.random(n) < 0.6))                       # off-block recorded as the schedule time (the label's odd mechanism)
        block = pd.Series(np.where(copy, sched, block))
        taxi = ((tmin - block).dt.total_seconds()).to_numpy().round()
        dest = rng.choice(OTHER, n)
        fid = first_id + apt_i * 10**6 + np.arange(n)
        oper = np.array([f"{h:032x}" for h in rng.integers(0, 12, n) * 10**14])
        dep = pd.DataFrame({
            "MVT_ID_mvt": fid + 0.0, "FLIGHT_ID_mvt": fid + 7e8, "FLIGHT_mvt": [f"{AIRLINES[a]}{x}" for a, x in zip(rng.integers(0, len(AIRLINES), n), rng.integers(100, 999, n))],
            "FLIGHT_RULE_mvt": "I", "ADEP_mvt": apt, "ADES_mvt": dest, "PHASE_mvt": "DEP",
            "MVT_TIME_UTC_mvt": tmin, "BLOCK_TIME_UTC_mvt": block.values, "SCHED_TIME_UTC_mvt": sched,
            "AIRCRAFT_TYPE_mvt": [TYPES[t][0] for t in ty], "RUNWAY_mvt": [RUNWAYS[apt][r] for r in rwy], "STAND_mvt": [f"S{s}" for s in stand],
            "TAXITIME_SEC_mvt": taxi, "LOBT_flt": sched, "CALLSIGN_flt": [f"{AIRLINES[a]}{x}X" for a, x in zip(rng.integers(0, len(AIRLINES), n), rng.integers(100, 999, n))],
            "ADEP_flt": apt, "ADES_flt": dest, "ADES_FILED_flt": dest, "MARKET_SEGMENT_flt": rng.choice(SEG, n), "IOBT_flt": sched,
            "FLIGHT_RULE_flt": "I", "FLIGHT_TYPE_flt": "S", "AIRCRAFT_TYPE_flt": [TYPES[t][0] for t in ty], "WK_TBL_CAT_flt": [TYPES[t][1] for t in ty],
            "AIRCRAFT_OPERATOR_flt": oper, "EOBT_1_flt": sched - pd.to_timedelta(rng.integers(0, 600, n), "s"),
            "ARVT_1_flt": tmin + pd.to_timedelta(rng.integers(3600, 20000, n), "s"),
            "AOBT_3_flt": block.values + pd.to_timedelta(rng.integers(-120, 120, n), "s"), "ARVT_3_flt": tmin + pd.to_timedelta(rng.integers(3600, 20000, n), "s"),
        })
        dep.loc[far & (rng.random(n) < 0.7), "AOBT_3_flt"] = pd.NaT
        # a few departures have no flight record (every *_flt column empty); their arrival record at the destination carries the LOBT
        un = rng.random(n) < 0.08
        flt = [c for c in dep.columns if c.endswith("_flt")]
        wit = dep[un].copy()
        wit["PHASE_mvt"], wit["ADEP_mvt"], wit["ADES_mvt"] = "ARR", apt, dep.ADES_mvt[un].to_numpy()
        wit["MVT_ID_mvt"] = first_id + 9 * 10**8 + apt_i * 10**6 + np.arange(int(un.sum())) + 0.0
        wit["MVT_TIME_UTC_mvt"] = dep.MVT_TIME_UTC_mvt[un] + pd.to_timedelta(rng.integers(5400, 14400, int(un.sum())), "s")
        wit["BLOCK_TIME_UTC_mvt"] = wit.MVT_TIME_UTC_mvt + pd.to_timedelta(400, "s")
        wit["TAXITIME_SEC_mvt"] = 400.0
        dep.loc[un, flt] = None
        dep.loc[un, "FLIGHT_ID_mvt"] = np.nan
        # arrivals at the same airport: landing time, in-block time after the taxi-in
        m = n
        lt = start + pd.to_timedelta(rng.integers(0, days, m), "D") + pd.to_timedelta(np.clip(rng.normal(13, 4.5, m), 0, 23.99) * 3600, "s")
        tin = (300 + rng.lognormal(5, 0.5, m)).round()
        aid = first_id + 5 * 10**8 + apt_i * 10**6 + np.arange(m)
        arr = dep.sample(m, replace=True, random_state=int(rng.integers(0, 10**6))).reset_index(drop=True).copy()
        arr["MVT_ID_mvt"] = aid + 0.0
        arr["FLIGHT_ID_mvt"] = aid + 7e8
        arr["PHASE_mvt"] = "ARR"
        arr["ADEP_mvt"], arr["ADES_mvt"] = rng.choice(OTHER, m), apt
        arr["MVT_TIME_UTC_mvt"], arr["BLOCK_TIME_UTC_mvt"] = lt, lt + pd.to_timedelta(tin, "s")
        arr["SCHED_TIME_UTC_mvt"] = lt - pd.to_timedelta(rng.integers(-300, 900, m), "s")
        arr["TAXITIME_SEC_mvt"] = tin
        arr["ADEP_flt"], arr["ADES_flt"], arr["ADES_FILED_flt"] = arr["ADEP_mvt"], apt, apt
        for c in ("LOBT_flt", "IOBT_flt", "EOBT_1_flt"):
            arr[c] = lt - pd.to_timedelta(rng.integers(7200, 20000, m), "s")
        arr["AOBT_3_flt"] = arr["EOBT_1_flt"]
        arr["ARVT_1_flt"] = arr["ARVT_3_flt"] = lt
        rows += [dep, arr, wit]
    return pd.concat(rows, ignore_index=True)


def to_table(df, mask_dep=False):
    df = df.sort_values("MVT_TIME_UTC_mvt").reset_index(drop=True).copy()
    if mask_dep:
        d = df.PHASE_mvt.eq("DEP")
        df.loc[d, "BLOCK_TIME_UTC_mvt"] = pd.NaT
        df.loc[d, "TAXITIME_SEC_mvt"] = np.nan
    tab = {}
    for c in df.columns:
        if c in TIME_COLS:
            tab[c] = pa.array(pd.to_datetime(df[c], utc=True).astype("datetime64[us, UTC]"), type=TS, from_pandas=True)
        elif c == "TAXITIME_SEC_mvt":
            tab[c] = pa.array(df[c].astype("Int32") if mask_dep else df[c].round().astype("int32"), type=pa.int32(), from_pandas=True)
        elif c in ("MVT_ID_mvt", "FLIGHT_ID_mvt"):
            tab[c] = pa.array(df[c].astype("float64"))
        else:
            tab[c] = pa.array(df[c].astype(object).where(df[c].notna(), None), type=pa.string())
    return pa.table(tab)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--days", type=int, default=8, help="days kept per month")
    ap.add_argument("--per-day", type=int, default=18, help="departures per airport and day (and as many arrivals)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zip", type=Path, default=None)
    ap.add_argument("--truth", type=Path, default=Path("synthetic_truth.parquet"), help="true taxi times of the final months' departures (to score the smoke output)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    files = []
    next_id = 10**8
    for m in range(1, 13):
        df = month_rows(rng, 2025, m, a.days, a.per_day, next_id)
        next_id += 10**7
        nxt = "2026-01" if m == 12 else f"2025-{m + 1:02d}"
        f = a.out / f"training_2025-{m:02d}-01_{nxt}-01.parquet"
        pq.write_table(to_table(df), f)
        files.append(f)
    new = {(y, m): month_rows(rng, y, m, a.days, a.per_day, next_id + 10**7 * k) for k, (y, m) in enumerate([(2026, 1), (2026, 2), (2026, 6), (2026, 7)])}
    rank = pd.concat([new[(2026, 1)], new[(2026, 7)]], ignore_index=True)
    final = pd.concat(new.values(), ignore_index=True)
    for name, df in (("ranking.parquet", rank), ("final_ranking.parquet", final)):
        t = to_table(df, mask_dep=True)
        pq.write_table(t, a.out / name)
        if name == "ranking.parquet":
            files.append(a.out / name)
        dep = df[df.PHASE_mvt == "DEP"].sort_values("MVT_TIME_UTC_mvt")
        sub = pa.table({"MVT_ID_mvt": pa.array(dep.MVT_ID_mvt.astype("float64")), "TAXITIME_SEC_mvt": pa.array([None] * len(dep), type=pa.int32())})
        pq.write_table(sub, a.out / ("submitting.parquet" if name == "ranking.parquet" else "final_submitting.parquet"))
        if name == "ranking.parquet":
            files.append(a.out / "submitting.parquet")
    # the truth of the final months is kept beside, for scoring the smoke output only
    dep = final[final.PHASE_mvt == "DEP"][["MVT_ID_mvt", "TAXITIME_SEC_mvt"]]
    dep.to_parquet(a.truth, index=False)
    if a.zip:
        with zipfile.ZipFile(a.zip, "w") as z:
            for f in files:
                z.write(f, f.name)
    print(f"synthetic challenge files in {a.out}: {len(files)} in the archive set, final_ranking {len(final):,} rows")


if __name__ == "__main__":
    main()
