# SPDX-License-Identifier: GPL-3.0-or-later
"""Label-free features from the external open data in data/external/ (docs/data-sources.md).

Writes data/processed/features_dep_ext_<source>.parquet keyed on MVT_ID_mvt for every departure (train 2025 +
ranking 2026), usable as `mix.py --extra ext_metar,ext_prc,ext_cal,ext_osm`.

Information-availability assumptions (retrospective challenge; nothing after the movement time except day level):
  ext_metar  wxs_* = latest report at or before SCHED time, wxm_* = latest at or before MVT (take-off) time, with
             rolling windows ending at that report; wxday_* summarise the whole UTC day.
  ext_prc    EUROCONTROL airport x UTC-day traffic, arrival ATFM delay by cause and slot adherence for the UTC day of
             MVT (whole day, published after the fact).
             Monthly taxi-out additional time is deliberately NOT used: it is an aggregate of the target itself
             and the 2026 file stops at June.
  ext_cal    public holidays for the local date of SCHED.
  ext_osm    static layout: OSM taxi-network distance from stand to the departure runway threshold.
Usage: python features_ext.py [metar] [prc] [cal] [osm]   (default: all)
"""
import json
import math
import re
import sys

import numpy as np
import pandas as pd
import polars as pl

from features import TZ
from ocommon import PROC, ROOT, report

EXT = ROOT / "data/external"
RUN = "features-ext"
PURPOSE = "Build label-free weather, EUROCONTROL day-level, holiday and OSM taxi-distance features for all departures"
KEY = "MVT_ID_mvt"


def movements():
    return pl.read_parquet(PROC / "features_dep.parquet", columns=[
        KEY, "apt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "split"])


def n_rows():  # departures of the phase (2,429,888 in the 2025+ranking phase)
    return pl.scan_parquet(PROC / "features_dep.parquet").select(pl.len()).collect().item()


def save(df, name):
    assert df[KEY].n_unique() == df.height == n_rows(), (name, df.height)
    df.write_parquet(PROC / f"features_dep_{name}.parquet")
    nulls = {c: round(df[c].null_count() / df.height, 3) for c in df.columns if c != KEY}
    return {"file": f"data/processed/features_dep_{name}.parquet", "columns": df.width - 1, "null_share": nulls}


# ---------------------------------------------------------------- METAR
WX_FLAGS = {  # regex on the present-weather part of the raw METAR (after the time group, before RMK/TEMPO/BECMG)
    "ts": r"(?<![A-Z])[-+]?(VC)?[A-Z]*TS", "sn": r"(?<![A-Z])[-+]?(SH|BL|DR|FZ)?[A-Z]*(SN|SG|PL|GS|GR)",
    "fz": r"(?<![A-Z])[-+]?FZ(RA|DZ|FG)", "fg": r"(?<![A-Z])(MI|BC|PR|FZ|VC)?FG", "br": r"(?<![A-Z])(BR|HZ)(?![A-Z])",
    "ra": r"(?<![A-Z])[-+]?(SH|TS|FZ)?(RA|DZ)", "heavy": r"(?<![A-Z])\+[A-Z]{2,}", "cb": r"\d{3}(CB|TCU)",
}


def parse_metar():
    frames = []
    for f in sorted((EXT / "metar").glob("*.csv")):
        frames.append(pd.read_csv(f, na_values=["M"], low_memory=False,
                                  usecols=["station", "valid", "tmpf", "dwpf", "relh", "drct", "sknt", "gust", "vsby",
                                           "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4", "metar"]))
    w = pd.concat(frames, ignore_index=True)
    w["t"] = (pd.to_datetime(w["valid"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    w = w.sort_values(["station", "t"]).drop_duplicates(["station", "t"], keep="last")
    raw = w["metar"].fillna("")
    body = raw.str.split(r"\s(?:RMK|TEMPO|BECMG|NOSIG)\b", n=1, regex=True).str[0]
    out = pd.DataFrame({"apt": w["station"].to_numpy(), "t": w["t"].to_numpy()})
    out["temp_c"] = ((w["tmpf"] - 32) / 1.8).round(1).to_numpy()
    out["dewsprd_c"] = ((w["tmpf"] - w["dwpf"]) / 1.8).round(1).to_numpy()
    out["relh"] = w["relh"].to_numpy()
    out["wspd_kt"] = w["sknt"].to_numpy()
    out["gust_kt"] = w["gust"].fillna(w["sknt"]).to_numpy()
    vrb = body.str.contains(r"\sVRB\d\d", regex=True)
    drct = w["drct"].where(~vrb.to_numpy())
    out["wdir"] = drct.to_numpy()
    vis = w["vsby"] * 1609.34
    vis = vis.where(~body.str.contains("CAVOK"), 10000).clip(upper=10000)
    out["vis_m"] = vis.round().to_numpy()
    ceil = np.full(len(w), 20000.0)
    for i in range(1, 5):
        c, h = w[f"skyc{i}"].astype(str).str.strip(), w[f"skyl{i}"]
        hit = c.isin(["BKN", "OVC", "VV"]) & h.notna()
        ceil = np.where(hit.to_numpy() & (h.to_numpy() < ceil), h.to_numpy(), ceil)
    vv = body.str.extract(r"\sVV(\d{3})")[0].astype(float) * 100
    ceil = np.where(vv.notna() & (vv.to_numpy() < ceil), vv.to_numpy(), ceil)
    out["ceil_ft"] = ceil
    rvr = body.str.findall(r"\sR\d\d[LCR]?/[PM]?(\d{4})").apply(lambda xs: min(map(int, xs)) if xs else np.nan)
    out["rvr_min_m"] = rvr.to_numpy()
    wxpart = body.str.replace(r"^.*?\d{6}Z", "", regex=True)
    for k, rx in WX_FLAGS.items():
        out[k] = wxpart.str.contains(rx, regex=True).astype("int8").to_numpy()
    out["precip"] = ((out["ra"] + out["sn"] + out["fz"]) > 0).astype("int8")
    out["lvp"] = ((out["vis_m"] < 800) | (out["ceil_ft"] <= 200) | (out["rvr_min_m"] < 600)).astype("int8")
    moist = (out["precip"] == 1) | (out["fg"] == 1) | (out["br"] == 1) | (out["relh"] >= 95)
    out["deice"] = ((out["temp_c"] <= 3) & moist).astype("int8")
    return pl.from_pandas(out).sort("apt", "t")


def metar_features(mv):
    wx = parse_metar()
    wx = wx.with_columns(ts_dt=pl.from_epoch("t", time_unit="s"))
    roll = []
    for period in ("1h", "3h"):
        r = wx.rolling("ts_dt", period=period, group_by="apt").agg(
            pl.len().alias(f"nrep_{period}"), pl.col("vis_m").min().alias(f"vis_min_{period}"),
            pl.col("ceil_ft").min().alias(f"ceil_min_{period}"), pl.col("gust_kt").max().alias(f"gust_max_{period}"),
            *[pl.col(k).max().alias(f"{k}_{period}") for k in ("ts", "cb", "sn", "fz", "precip", "lvp", "deice", "heavy")])
        roll.append(r)
    base = wx
    for r in roll:
        base = base.join(r, on=["apt", "ts_dt"], how="left")
    base = base.drop("ts_dt").sort("t")
    day = wx.with_columns(day=pl.col("t") // 86400).group_by("apt", "day").agg(
        pl.col("ts").sum().alias("wxday_ts_n"), pl.col("sn").sum().alias("wxday_sn_n"),
        pl.col("fz").sum().alias("wxday_fz_n"), pl.col("lvp").sum().alias("wxday_lvp_n"),
        pl.col("deice").sum().alias("wxday_deice_n"), pl.col("vis_m").min().alias("wxday_vis_min"),
        pl.col("gust_kt").max().alias("wxday_gust_max"), pl.len().alias("wxday_nrep"))

    m = mv.select(KEY, "apt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt")
    out = m.select(KEY)
    for prefix, anchor in (("wxs_", "SCHED_TIME_UTC_mvt"), ("wxm_", "MVT_TIME_UTC_mvt")):
        j = m.select(KEY, "apt", anchor, "RUNWAY_mvt").sort(anchor).join_asof(
            base.rename({"t": "obs_t"}), left_on=anchor, right_on="obs_t", by="apt", strategy="backward",
            tolerance=3 * 3600)
        j = j.with_columns(age_min=((pl.col(anchor) - pl.col("obs_t")) / 60).cast(pl.Float32))
        if prefix == "wxm_":
            hdg = pl.col("RUNWAY_mvt").str.extract(r"^(\d\d)", 1).cast(pl.Float64, strict=False) * 10
            rel = (pl.col("wdir") - hdg) * math.pi / 180
            j = j.with_columns(head_kt=(pl.col("wspd_kt") * rel.cos()).round(1),
                               cross_kt=(pl.col("wspd_kt") * rel.sin()).abs().round(1))
        rad = pl.col("wdir") * math.pi / 180
        j = j.with_columns(wdir_sin=rad.sin().round(3), wdir_cos=rad.cos().round(3)).drop("wdir")
        keep = [c for c in j.columns if c not in (KEY, "apt", anchor, "RUNWAY_mvt", "obs_t")]
        out = out.join(j.select(KEY, *[pl.col(c).alias(prefix + c) for c in keep]), on=KEY, how="left")
    out = out.join(m.select(KEY, "apt", day=pl.col("MVT_TIME_UTC_mvt") // 86400).join(day, on=["apt", "day"], how="left")
                   .drop("apt", "day"), on=KEY, how="left")
    return out


# ---------------------------------------------------------------- EUROCONTROL PRC day level
def prc_features(mv):
    # Not used: ATC/All pre-departure delay XLSX (release 2026-08-14 has no July 2026, and ATC is ~50% missing even in
    # 2025), so they would be null for every July ranking row but filled for July training rows. The daily CSVs read
    # below reach 2026-08 and are used instead.
    def csvs(stem, **kw):
        d = pd.concat([pd.read_csv(p, encoding="latin-1", **kw) for p in sorted((EXT / "prc").glob(f"{stem}_202[456].csv*"))])
        d = d[d["APT_ICAO"].isin(TZ)].copy()
        d["FLT_DATE"] = d["FLT_DATE"].astype(str).str[:10]
        return d

    tr = csvs("airport_traffic")
    tr = tr.assign(prc_dep=tr["FLT_DEP_IFR_2"].fillna(tr["FLT_DEP_1"]), prc_arr=tr["FLT_ARR_IFR_2"].fillna(tr["FLT_ARR_1"]))
    tr = tr[["APT_ICAO", "FLT_DATE", "prc_dep", "prc_arr"]]
    dl = csvs("apt_dly")
    causes = {"wx": ["W"], "cap": ["C", "G"], "staff": ["S"], "strike": ["I", "N"], "deice": ["D"], "equip": ["T", "E"],
              "other": ["A", "M", "O", "P", "R", "V", "NA"]}
    dly = pd.DataFrame({"APT_ICAO": dl["APT_ICAO"], "FLT_DATE": dl["FLT_DATE"],
                        "prc_arr_atfm_min": dl["DLY_APT_ARR_1"],
                        "prc_arr_atfm_per_flt": dl["DLY_APT_ARR_1"] / dl["FLT_ARR_1"].replace(0, np.nan),
                        "prc_arr_dly15_share": dl["FLT_ARR_1_DLY_15"] / dl["FLT_ARR_1"].replace(0, np.nan)})
    for k, codes in causes.items():
        dly[f"prc_arr_atfm_{k}_min"] = sum(dl[f"DLY_APT_ARR_{c}_1"].fillna(0) for c in codes)
    sa = csvs("atfm_slot_adherence")
    slot = pd.DataFrame({"APT_ICAO": sa["APT_ICAO"], "FLT_DATE": sa["FLT_DATE"],
                         "prc_dep_regulated_share": sa["FLT_DEP_REG_1"] / sa["FLT_DEP_1"].replace(0, np.nan),
                         "prc_dep_regulated_n": sa["FLT_DEP_REG_1"],
                         "prc_dep_slot_late_share": sa["FLT_DEP_OUT_LATE_1"] / sa["FLT_DEP_REG_1"].replace(0, np.nan)})
    d = tr
    for x in (dly, slot):
        d = d.merge(x.drop_duplicates(["APT_ICAO", "FLT_DATE"]), on=["APT_ICAO", "FLT_DATE"], how="outer")
    med = d[d["FLT_DATE"].str.startswith("2025")].groupby("APT_ICAO")["prc_dep"].median()
    d["prc_dep_rel"] = d["prc_dep"] / d["APT_ICAO"].map(med)
    d["prc_strike_day"] = (d["prc_arr_atfm_strike_min"] > 0).astype("int8")
    d = pl.from_pandas(d.rename(columns={"APT_ICAO": "apt"}))
    m = mv.select(KEY, "apt", FLT_DATE=pl.from_epoch("MVT_TIME_UTC_mvt", time_unit="s").dt.strftime("%Y-%m-%d"))
    return m.join(d, on=["apt", "FLT_DATE"], how="left").drop("apt", "FLT_DATE")


# ---------------------------------------------------------------- calendars
def cal_features(mv):
    h = pd.read_csv(EXT / "calendars/public_holidays.csv")
    parts = []
    for apt, tz in TZ.items():
        sub = mv.filter(pl.col("apt") == apt).select(
            KEY, d=pl.from_epoch("SCHED_TIME_UTC_mvt", time_unit="s").dt.replace_time_zone("UTC")
            .dt.convert_time_zone(tz).dt.date())
        days = np.sort(pd.to_datetime(h.loc[h["airport"] == apt, "date"]).dt.date.unique()).astype("datetime64[D]")
        q = sub["d"].to_numpy().astype("datetime64[D]")
        i = np.searchsorted(days, q, "left")
        nxt = np.where(i < len(days), (days[np.minimum(i, len(days) - 1)] - q).astype(int), 99)
        prv = np.where(i > 0, (q - days[np.maximum(i - 1, 0)]).astype(int), 99)
        is_h = nxt == 0
        prv = np.where(is_h, 0, prv)
        parts.append(sub.select(KEY).with_columns(
            cal_holiday=pl.Series(is_h.astype("int8")),
            cal_days_to_holiday=pl.Series(np.minimum(nxt, 30).astype("int16")),
            cal_days_from_holiday=pl.Series(np.minimum(prv, 30).astype("int16")),
            cal_bridge=pl.Series(((~is_h) & ((nxt == 1) | (prv == 1))).astype("int8"))))
    return pl.concat(parts)


# ---------------------------------------------------------------- OSM taxi network
def _xy(lat, lon, lat0):
    return np.c_[np.asarray(lon) * 111320 * math.cos(math.radians(lat0)), np.asarray(lat) * 110540]


def _norm_ref(s):
    s = str(s).strip().upper().replace("PIER ", "").replace(" ", "")
    return re.sub(r"(?<=[A-Z])0+(?=\d)", "", s)


def _base(s):
    return re.sub(r"(?<=\d)[A-Z]$", "", s)


def osm_airport(apt, pairs):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.spatial import cKDTree

    els = json.loads((EXT / "osm" / f"{apt}.json").read_text(encoding="utf-8"))["elements"]
    ad = next(e for e in els if e.get("tags", {}).get("aeroway") == "aerodrome")
    lat0 = (ad["bounds"]["minlat"] + ad["bounds"]["maxlat"]) / 2
    pos, edges = {}, []
    rw_ways = []
    for e in els:
        a = e.get("tags", {}).get("aeroway")
        if e["type"] != "way" or a not in ("taxiway", "taxilane", "runway", "parking_position"):
            continue
        for nid, g in zip(e["nodes"], e["geometry"]):
            pos[nid] = (g["lat"], g["lon"])
        edges += list(zip(e["nodes"][:-1], e["nodes"][1:]))
        if a == "runway" and e["tags"].get("ref"):
            rw_ways.append(e)
    ids = np.array(list(pos))
    idx = {n: i for i, n in enumerate(ids)}
    xy = _xy([pos[n][0] for n in ids], [pos[n][1] for n in ids], lat0)
    a = np.array([idx[u] for u, v in edges]); b = np.array([idx[v] for u, v in edges])
    wgt = np.hypot(*(xy[a] - xy[b]).T)
    g = coo_matrix((np.r_[wgt, wgt], (np.r_[a, b], np.r_[b, a])), shape=(len(ids), len(ids))).tocsr()
    _, lab = connected_components(g, directed=False)
    main = np.bincount(lab).argmax()
    in_main = np.flatnonzero(lab == main)
    tree = cKDTree(xy[in_main])

    # stands: every parking_position / gate ref -> centroid
    stands = {}
    for e in els:
        t = e.get("tags", {})
        if t.get("aeroway") not in ("parking_position", "gate", "stand"):
            continue
        pts = [(e["lat"], e["lon"])] if e["type"] == "node" else [(p["lat"], p["lon"]) for p in e.get("geometry", [])]
        if not pts:
            continue
        c = _xy([p[0] for p in pts], [p[1] for p in pts], lat0).mean(0)
        for r in re.split(r"[/,;]", t.get("ref") or t.get("name") or ""):
            if r.strip():
                stands.setdefault(_norm_ref(r), []).append(c)
    stands = {k: np.mean(v, 0) for k, v in stands.items()}
    by_base = {}
    for k, v in stands.items():
        by_base.setdefault(_base(k), []).append(v)
    by_base = {k: np.mean(v, 0) for k, v in by_base.items()}

    def stand_xy(s):
        n = _norm_ref(s)
        if n in stands:
            return stands[n], 1
        if n in by_base:
            return by_base[n], 2
        if _base(n) in stands:
            return stands[_base(n)], 2
        mt = re.match(r"^([A-Z]*)(\d+)", n)
        if mt:
            cands = [(abs(int(re.match(r"^[A-Z]*(\d+)", k).group(1)) - int(mt.group(2))), v) for k, v in stands.items()
                     if re.match(rf"^{mt.group(1)}\d+", k) and re.match(r"^[A-Z]*(\d+)", k)]
            cands = [c for c in cands if c[0] <= 3]
            if cands:
                return min(cands, key=lambda c: c[0])[1], 3
        return None, 0

    # runway thresholds per designator
    thr = {}
    for e in rw_ways:
        p = _xy([q["lat"] for q in e["geometry"]], [q["lon"] for q in e["geometry"]], lat0)
        ends = [(e["nodes"][0], p[0]), (e["nodes"][-1], p[-1])]
        for des in e["tags"]["ref"].split("/"):
            mt = re.match(r"^(\d\d)", des.strip())
            if not mt:
                continue
            hdg = int(mt.group(1)) * 10
            for (n0, p0), (_, p1) in (ends, ends[::-1]):
                brg = math.degrees(math.atan2(p1[0] - p0[0], p1[1] - p0[1])) % 360
                if abs((brg - hdg + 180) % 360 - 180) < 45:
                    length = np.hypot(*(p1 - p0))
                    if des.strip() not in thr or length > thr[des.strip()][2]:
                        thr[des.strip()] = (idx[n0], p0, length, [idx[n] for n in e["nodes"]])
    rw_list = sorted(thr)
    src = []
    for r in rw_list:
        node = thr[r][0]
        if lab[node] != main:  # threshold node not connected: snap to nearest main-component node
            node = in_main[tree.query(thr[r][1])[1]]
        src.append(node)
    dist = dijkstra(g, directed=False, indices=src) if src else np.zeros((0, len(ids)))
    any_rw = {}
    for k, r in enumerate(rw_list):
        nodes = [n for n in thr[r][3] if lab[n] == main]
        any_rw[r] = dijkstra(g, directed=False, indices=nodes, min_only=True) if nodes else None

    rows = []
    for stand, rwy in pairs:
        sxy, how = stand_xy(stand) if stand is not None else (None, 0)
        rec = {"STAND_mvt": stand, "RUNWAY_mvt": rwy, "osm_stand_match": how}
        if sxy is not None:
            d0, j = tree.query(sxy)
            node = in_main[j]
            c = np.asarray(sxy)
            rec.update(osm_stand_x=float(c[0] - xy[in_main][:, 0].mean()), osm_stand_y=float(c[1] - xy[in_main][:, 1].mean()),
                       osm_snap_m=float(d0))
            r = (rwy or "").strip()
            if r in thr:
                k = rw_list.index(r)
                rec["osm_taxi_thr_m"] = float(dist[k, node] + d0)
                rec["osm_euclid_thr_m"] = float(np.hypot(*(c - thr[r][1])))
                rec["osm_taxi_rwy_any_m"] = float(any_rw[r][node] + d0) if any_rw[r] is not None else None
            rec["osm_taxi_nearest_thr_m"] = float(dist[:, node].min() + d0) if len(src) else None
        rows.append(rec)
    return pd.DataFrame(rows), {"graph_nodes": int(len(in_main)), "runways": rw_list, "osm_stands": len(stands)}


def osm_features(mv):
    parts, info = [], {}
    for apt in TZ:
        pairs = mv.filter(pl.col("apt") == apt).select("STAND_mvt", "RUNWAY_mvt").unique().rows()
        df, info[apt] = osm_airport(apt, pairs)
        df["apt"] = apt
        parts.append(df)
    lut = pl.from_pandas(pd.concat(parts, ignore_index=True)).with_columns(
        pl.col("osm_stand_match").cast(pl.Int8),
        osm_taxi_ratio=pl.col("osm_taxi_thr_m") / pl.col("osm_euclid_thr_m"))
    out = mv.select(KEY, "apt", "STAND_mvt", "RUNWAY_mvt").join(lut, on=["apt", "STAND_mvt", "RUNWAY_mvt"], how="left",
                                                               nulls_equal=True)
    for apt in TZ:
        s = out.filter(pl.col("apt") == apt)
        info[apt]["row_share_stand_matched"] = round(float((s["osm_stand_match"] > 0).mean()), 4)
        info[apt]["row_share_taxi_dist"] = round(float(s["osm_taxi_thr_m"].is_not_null().mean()), 4)
    return out.drop("apt", "STAND_mvt", "RUNWAY_mvt"), info


def main():
    todo = sys.argv[1:] or ["metar", "prc", "cal", "osm"]
    mv = movements()
    summary_path = ROOT / "reports/external_features.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    report(RUN, 0, len(todo), "loaded movements", purpose=PURPOSE)
    for k, name in enumerate(todo, 1):
        if name == "metar":
            summary["ext_metar"] = save(metar_features(mv), "ext_metar")
        elif name == "prc":
            summary["ext_prc"] = save(prc_features(mv), "ext_prc")
        elif name == "cal":
            summary["ext_cal"] = save(cal_features(mv), "ext_cal")
        elif name == "osm":
            df, info = osm_features(mv)
            summary["ext_osm"] = save(df, "ext_osm") | {"airports": info}
        summary_path.write_text(json.dumps(summary, indent=2))
        report(RUN, k, len(todo), f"{name} written", finished=k == len(todo))


if __name__ == "__main__":
    main()
