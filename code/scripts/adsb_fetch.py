# SPDX-License-Identifier: GPL-3.0-or-later
"""Stream adsb.lol globe_history day archives and keep the ten airports' surface, terminal area and aircraft days.

Each day (2-4 GB split tar) is streamed straight from GitHub releases through tarfile 'r|' -- nothing big touches
disk. Per trace_full_<hex>.json, three zones (column `zone`):
  0 near   within NEAR_KM of an airport reference point, on the ground or below NEAR_FT -- every point
  1 tma    within TMA_KM, ground or below TMA_FT (arrival flows, holds, runway config) -- every point
  2 track  rest of the day of any aircraft with a zone-0 point (previous leg, turnaround) -- one point / TRACK_SEC
Output: data/external/adsblol/points/<day>.parquet (one row per point) and aircraft/<day>.parquet (every aircraft
of the day: icao, reg, type, owner, n_points, first/last time, near flag). Resumable: finished days are skipped.
Filter v1 (7 km / 3000 ft only) outputs are kept in points_narrow_v1/. Source of the day->release map: PREFERRED_RELEASES.txt in the roots of the adsblol/globe_history_<year> repositories.
Terms of use: docs/data-sources.md.

  python scripts/adsb_fetch.py                  # all Jan+Jul days of 2025 and 2026
  python scripts/adsb_fetch.py --final          # only the days of final_ranking.parquet that are not on disk yet (final phase)
  python scripts/adsb_fetch.py --months 2026-03 2026-09   # whole calendar months
  python scripts/adsb_fetch.py --days 2025-01-15 --max-traces 3000   # smoke test
"""
import argparse
import datetime as dt
import gzip
import io
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:  # 3-5x faster trace parsing where available (Kaggle: pip install orjson)
    from orjson import loads as _loads
except ImportError:
    _loads = json.loads

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ADSB_OUT", ROOT / "data/external/adsblol/points"))  # Kaggle kernels override
LOG = OUT.parent / "days.jsonl"
RUN = "adsb_fetch"

APTS = {  # ourairports reference points
    "EDDF": (50.026706, 8.558350), "EDDM": (48.353802, 11.786100), "EGLL": (51.470748, -0.459909),
    "EHAM": (52.308601, 4.763890), "LEBL": (41.297100, 2.078460), "LEMD": (40.493407, -3.572249),
    "LFPG": (49.008960, 2.554117), "LIRF": (41.804532, 12.251998), "LSZH": (47.458056, 8.548056),
    "LTFM": (41.274874, 28.732136),
}
NEAR_KM, NEAR_FT = 7.0, 3000.0
TMA_KM, TMA_FT = 100.0, 15000.0
TRACK_SEC = 30
FILTER = {"v": 2, "near": [NEAR_KM, NEAR_FT], "tma": [TMA_KM, TMA_FT], "track_sec": TRACK_SEC}
ACS = OUT.parent / "aircraft"
_A = np.array(list(APTS.values()))
_NAMES = np.array(list(APTS))
_BOX = (_A[:, 0].min() - 1.0, _A[:, 0].max() + 1.0, _A[:, 1].min() - 1.7, _A[:, 1].max() + 1.7)

_progress = None


def report(done, total, detail, finished=False):
    print(f"[{time.strftime('%H:%M:%S')}] [{done}/{total}] {detail}", flush=True)
    if _progress:
        _progress.report(RUN, "EUROCONTROL adsb.lol fetch", done, total, detail=detail, finished=finished,
                         purpose="Pull adsb.lol surface, terminal-area and aircraft-day tracks for the 10 airports, 2025 + Jan/Jul 2026")


RANK = ("prod-0", "staging-0", "prod-0tmp", "staging-0tmp")  # full releases before tmp; mlatonly and the rest last


def preferred_urls():
    """Day -> split-tar URLs. Late-Dec 2025 days are listed by both repos (the 2026 list names MLAT-only releases
    for 2025-12-14..17 while the 2025 list names prod/staging), so keep the best-ranked release per day."""
    urls = {}
    rank = lambda tag: RANK.index(tag.split("-readsb-")[1]) if tag.split("-readsb-")[1] in RANK else len(RANK)
    for yr in (2025, 2026):
        txt = requests.get(f"https://raw.githubusercontent.com/adsblol/globe_history_{yr}/main/PREFERRED_RELEASES.txt",
                           timeout=60).text
        for line in txt.split():
            parts = line.strip().split(",")
            tag = parts[0].split("/download/")[1].split("/")[0]  # v2025.01.15-planes-readsb-prod-0
            day = tag[1:11].replace(".", "-")
            if day not in urls or rank(tag) < urls[day][0]:
                urls[day] = (rank(tag), parts)
    return {d: p for d, (_, p) in urls.items()}


def fallback_urls(day):
    """Other replicas of the same day (used when the preferred one fails)."""
    yr = day[:4]
    out = []
    for suffix in ("prod-0", "staging-0", "prod-0tmp", "staging-0tmp"):
        tag = f"v{day.replace('-', '.')}-planes-readsb-{suffix}"
        r = requests.get(f"https://api.github.com/repos/adsblol/globe_history_{yr}/releases/tags/{tag}", timeout=60)
        if r.ok:
            assets = sorted(a["browser_download_url"] for a in r.json().get("assets", []) if ".tar." in a["name"])
            if assets:
                out.append(assets)
    return out


class Chain(io.RawIOBase):
    """Sequential read over the concatenated HTTP bodies of the split tar parts.

    A broken connection (GitHub's CDN resets long downloads: IncompleteRead after 1-1.8 GB of a 2 GB part) resumes the
    same part with a Range request at the byte already delivered instead of failing the whole day (the parts are plain
    split tar, so the byte stream continues exactly)."""

    RETRIES = 25

    def __init__(self, urls):
        self.urls, self.cur, self.nbytes = list(urls), None, 0
        self.url, self.pos, self.total = None, 0, None

    def readable(self):
        return True

    def _open(self):
        hdr = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        r = requests.get(self.url, stream=True, timeout=120, headers=hdr)
        r.raise_for_status()
        if self.pos and r.status_code != 206:
            raise OSError(f"server ignored Range at {self.pos}")
        n = r.headers.get("Content-Length")
        self.total = self.pos + int(n) if n is not None else None
        self.cur = r.raw

    def readinto(self, b):
        fails = 0
        while True:
            try:
                if self.cur is None:
                    if self.url is None or (self.total is not None and self.pos >= self.total):
                        if not self.urls:
                            return 0
                        self.url, self.pos, self.total = self.urls.pop(0), 0, None
                    self._open()
                n = self.cur.readinto(b)
                if n:
                    self.nbytes += n
                    self.pos += n
                    return n
                if self.total is not None and self.pos < self.total:
                    raise OSError(f"short body {self.pos}/{self.total}")
                self.cur, self.url = None, None
            except Exception as e:  # noqa: BLE001 - reconnect at the byte offset
                if self.url is None or (isinstance(e, requests.HTTPError) and e.response.status_code < 500):
                    raise
                fails += 1
                self.cur = None
                if fails > self.RETRIES:
                    raise
                time.sleep(min(2 * fails, 30))


def near(lat, lon):
    """Index of nearest airport and distance (km) for arrays of points."""
    d = np.full(lat.shape, np.inf)
    idx = np.zeros(lat.shape, int)
    for i, (alat, alon) in enumerate(_A):
        dy = (lat - alat) * 111.2
        dx = (lon - alon) * 111.2 * np.cos(np.radians(alat))
        di = np.hypot(dx, dy)
        m = di < d
        d[m], idx[m] = di[m], i
    return idx, d


def parse_trace(raw):
    """-> (points DataFrame or None, aircraft record or None)."""
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    j = _loads(raw)
    tr = j.get("trace") or []
    if not tr:
        return None, None
    ts = j["timestamp"]
    ac = {"icao": j.get("icao"), "reg": j.get("r"), "type": j.get("t"), "desc": j.get("desc"),
          "owner": j.get("ownOp"), "year": j.get("year"), "db_flags": j.get("dbFlags"), "n_points": len(tr),
          "t0": ts + tr[0][0], "t1": ts + tr[-1][0], "near": False}
    lat = np.fromiter((p[1] for p in tr), float, len(tr))
    lon = np.fromiter((p[2] for p in tr), float, len(tr))
    inbox = (lat > _BOX[0]) & (lat < _BOX[1]) & (lon > _BOX[2]) & (lon < _BOX[3])
    if not inbox.any():
        return None, ac
    idx, dist = near(lat, lon)
    alt_raw = [p[3] for p in tr]
    ground = np.array([a == "ground" for a in alt_raw])
    alt = np.array([np.nan if (a == "ground" or a is None) else float(a) for a in alt_raw])
    near_m = (dist < NEAR_KM) & (ground | (alt < NEAR_FT))
    tma_m = (dist < TMA_KM) & (ground | (alt < TMA_FT))
    t = ts + np.fromiter((p[0] for p in tr), float, len(tr))
    keep = tma_m.copy()
    if near_m.any():
        ac["near"] = True
        keep[np.unique(np.floor(t / TRACK_SEC), return_index=True)[1]] = True
    if not keep.any():
        return None, ac
    # callsign / squawk come in the per-point details dict (element 8) when they change; forward-fill them
    cs, sq, c, q = [], [], None, None
    for p in tr:
        if len(p) > 8 and isinstance(p[8], dict):
            c = (p[8].get("flight") or "").strip() or c
            q = p[8].get("squawk") or q
        cs.append(c)
        sq.append(q)
    k = np.flatnonzero(keep)
    f = lambda i, default=np.nan: np.array([tr[x][i] if len(tr[x]) > i and tr[x][i] is not None else default
                                            for x in k], dtype="float32")
    return pd.DataFrame({
        "icao": ac["icao"], "reg": ac["reg"], "type": ac["type"], "t": t[k],
        "lat": lat[k].astype("float32"), "lon": lon[k].astype("float32"), "alt": alt[k].astype("float32"),
        "ground": ground[k], "gs": f(4), "track": f(5), "flags": f(6, 0).astype("int8"), "vrate": f(7),
        "alt_geom": f(10), "ias": f(12),
        "src": [tr[x][9] if len(tr[x]) > 9 else None for x in k],
        "callsign": [cs[x] for x in k], "squawk": [sq[x] for x in k],
        "apt": _NAMES[idx[k]], "dist_km": dist[k].astype("float32"),
        "zone": np.where(near_m[k], 0, np.where(tma_m[k], 1, 2)).astype("int8"),
    }), ac


def fetch_day(day, url_sets, max_traces=None):
    OUT.mkdir(parents=True, exist_ok=True)
    last_err = None
    for urls in url_sets:
        t0, frames, acs, n_tr, n_keep = time.time(), [], [], 0, 0
        try:
            stream = io.BufferedReader(Chain(urls), buffer_size=1 << 20)
            with tarfile.open(fileobj=stream, mode="r|") as tf:
                for m in tf:
                    if "trace_full" not in m.name or not m.isfile():
                        continue
                    n_tr += 1
                    df, ac = parse_trace(tf.extractfile(m).read())
                    if ac is not None:
                        acs.append(ac)
                    if df is not None:
                        frames.append(df)
                        n_keep += 1
                    if max_traces and n_tr >= max_traces:
                        break
            if n_tr == 0:
                raise RuntimeError("no traces in archive")
            pts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            dest = OUT / f"{day}{'_smoke' if max_traces else ''}.parquet"
            ACS.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(acs).to_parquet(ACS / dest.name, index=False)
            pts.to_parquet(dest.with_suffix(".tmp"), index=False)
            dest.with_suffix(".tmp").replace(dest)
            rec = {"day": day, "urls": urls, "traces": n_tr, "aircraft_kept": n_keep, "points": len(pts),
                   "gb": round(stream.raw.nbytes / 1e9, 2), "sec": round(time.time() - t0),
                   "mb": round(dest.stat().st_size / 1e6, 1), "filter": FILTER,
                   "per_zone": pts["zone"].value_counts().sort_index().to_dict() if len(pts) else {},
                   "per_apt": pts.loc[pts["zone"] == 0, "apt"].value_counts().to_dict() if len(pts) else {}}
            return rec
        except Exception as e:  # noqa: BLE001 - try the next replica
            last_err = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"{day}: all replicas failed; last: {last_err}")


def all_days():
    """All of 2025 plus Jan and Jul 2026 (the challenge months)."""
    d = list(pd.date_range("2025-01-01", "2025-12-31").date) + list(pd.date_range("2026-01-01", "2026-01-31").date)         + list(pd.date_range("2026-07-01", "2026-07-31").date)
    return [str(x) for x in d]


def final_days(path=None):
    """Every calendar day of every month present in final_ranking.parquet (UTC; whole months, the jitter and
    cross-midnight turnarounds need the neighbouring days)."""
    import pyarrow.parquet as pq
    path = path or ROOT / "data/raw/final_ranking.parquet"
    t = pq.read_table(path, columns=["MVT_TIME_UTC_mvt"])["MVT_TIME_UTC_mvt"].to_pandas()
    ts = pd.DatetimeIndex(t.dropna()).tz_localize(None)
    cnt = pd.Series(1, index=ts.to_period("M")).groupby(level=0).sum()
    # months holding >= 2 % of the rows; a few stragglers (a Sep 30 arrival in the October file) do not pull in a month
    months = [str(m) for m, n in cnt.items() if n >= 0.02 * cnt.sum()]
    return month_days(months)


def month_days(months):
    out = []
    for m in months:
        out += [str(x.date()) for x in pd.date_range(f"{m}-01", periods=pd.Period(m).days_in_month)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*")
    ap.add_argument("--months", nargs="*", help="YYYY-MM calendar months instead of the default 2025 + Jan/Jul 2026")
    ap.add_argument("--final", action="store_true", help="the days present in data/raw/final_ranking.parquet")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-traces", type=int)
    a = ap.parse_args()
    days = a.days or (month_days(a.months) if a.months else final_days() if a.final else all_days())
    pref = preferred_urls()
    todo = [d for d in days if a.max_traces or not (OUT / f"{d}.parquet").exists()]
    done = len(days) - len(todo)
    report(done, len(days), f"{len(todo)} days to fetch, {done} already on disk")
    fails = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {}
        for d in todo:
            sets = [pref[d]] if d in pref else []
            futs[ex.submit(fetch_day_with_fallback, d, sets, a.max_traces)] = d
        for fu in as_completed(futs):
            d = futs[fu]
            try:
                rec = fu.result()
                with open(LOG, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                done += 1
                report(done, len(days), f"{d}: {rec['points']:,} pts ({rec['mb']} MB), {rec['aircraft_kept']:,}/{rec['traces']:,} "
                                        f"aircraft, {rec['gb']} GB in {rec['sec']} s")
            except Exception as e:  # noqa: BLE001
                fails.append(d)
                print(f"FAILED {d}: {e}", flush=True)
    report(done, len(days), f"finished; {len(fails)} failed days: {fails}", finished=True)
    return 1 if fails else 0


def fetch_day_with_fallback(day, sets, max_traces):
    try:
        return fetch_day(day, sets or fallback_urls(day), max_traces)
    except RuntimeError:
        if not sets:
            raise
        return fetch_day(day, fallback_urls(day), max_traces)


if __name__ == "__main__":
    sys.exit(main())
