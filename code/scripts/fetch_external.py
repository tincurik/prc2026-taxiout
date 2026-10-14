# SPDX-License-Identifier: GPL-3.0-or-later
"""Download the external open data candidates into data/external/<source>/.

Resumable: a file already on disk with the server's size is skipped; partial downloads go to
.part and are renamed only when complete. Each source dir gets SOURCE.md (URL, query, retrieval
date) and files.jsonl (url, path, bytes, sha256, retrieved_utc). Label-free; no features built here.
"""
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone

from common import ROOT, _progress

RUN = "eurocontrol-fetch_external"
EXT = ROOT / "data/external"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
UA = {"User-Agent": "prc-data-challenge-2026 research"}
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")

PRC = "https://www.eurocontrol.int/performance/data/download"
PRC_CSV = [f"{PRC}/csv/{name}_{y}.{ext}" for y in (2024, 2025, 2026) for name, ext in (
    ("airport_traffic", "csv"), ("apt_dly", "csv.bz2"), ("taxi_out_additional_time", "csv"),
    ("taxi_in_additional_time", "csv"), ("atfm_slot_adherence", "csv"), ("all_pre_departure_delays", "csv"))]
PRC_XLS = [f"{PRC}/xls/{n}.xlsx" for n in (
    "ATC_Pre-Departure_Delay", "All_Pre-Departure_Delay", "ASMA_Additional_Time",
    "Taxi-Out_Additional_Time", "Taxi_times_Planning_Data_S14_S22")]

OURAIRPORTS = [f"https://davidmegginson.github.io/ourairports-data/{n}.csv"
               for n in ("airports", "runways", "navaids")]

IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_START, IEM_END = date(2024, 12, 31), date(2026, 8, 1)  # end exclusive


def final_months():
    """Calendar months with >= 2 % of final_ranking.parquet's rows (final phase), else []. Jan/Jul 2026 included."""
    f = ROOT / "data/raw/final_ranking.parquet"
    if not f.exists():
        return []
    import pandas as pd
    import pyarrow.parquet as pq
    t = pq.read_table(f, columns=["MVT_TIME_UTC_mvt"])["MVT_TIME_UTC_mvt"].to_pandas().dropna()
    cnt = pd.Series(1, index=pd.DatetimeIndex(t).tz_localize(None).to_period("M")).groupby(level=0).sum()
    return sorted(str(m) for m, n in cnt.items() if n >= 0.02 * cnt.sum())


FINAL_MONTHS = final_months()  # empty until `scripts/fetch_final.py` has downloaded the final inputs
if FINAL_MONTHS:  # extend the weather window to the end of the latest final month (end exclusive)
    _last = max(FINAL_MONTHS)
    _y, _m = int(_last[:4]), int(_last[5:])
    IEM_END = max(IEM_END, date(_y + (_m == 12), _m % 12 + 1, 1))

OVERPASS = "https://overpass-api.de/api/interpreter"
OSM_QUERY = """[out:json][timeout:300];
nwr["icao"="{icao}"]["aeroway"="aerodrome"]->.apt;
.apt out geom;
.apt map_to_area->.a;
(
  nwr(area.a)["aeroway"~"^(runway|taxiway|taxilane|apron|parking_position|holding_position|stand|gate|hangar|terminal|runway_holding_position|deicing_pad)$"];
);
out geom;"""


def _next_month(m):
    y, mo = int(m[:4]), int(m[5:])
    return f"{y + (mo == 12)}-{mo % 12 + 1:02d}"


def opdi_urls():
    """OPDI files overlapping 2025 (+ a few days before) and Jan/Jul 2026, from the published link lists."""
    urls = set()
    for page in ("flight-event-data", "measurement-data", "flight-list-data"):
        html = get_bytes(f"https://www.opdi.aero/{page}").decode("utf-8", "replace")
        urls |= set(re.findall(r"https://www\.eurocontrol\.int/performance/data/download/OPDI/v002/"
                               r"[a-z_]+/[a-z_]+_[0-9_]+\.parquet", html))
    keep = []
    for u in sorted(urls):
        d = re.findall(r"_(\d{6,8})", u.rsplit("/", 1)[1])
        if len(d[0]) == 6:
            ok = 202412 <= int(d[0]) <= 202512 or d[0] in ("202601", "202607") or d[0] in {m.replace("-", "") for m in FINAL_MONTHS}
        else:
            a, b = int(d[0]), int(d[1])
            ok = (b > 20241225 and a < 20260201) or (b > 20260701 and a < 20260801) or any(
                b > int(m.replace("-", "") + "01") and a < int(_next_month(m).replace("-", "") + "01") for m in FINAL_MONTHS)
        if ok:
            keep.append(u)
    # flight lists first (small), then events, then measurements
    order = {"flight_list": 0, "flight_events": 1, "measurements": 2}
    return sorted(keep, key=lambda u: (order[u.split("/")[-2]], u))


def get_bytes(url, data=None, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - network retry
            code = getattr(e, "code", None)
            if i == tries - 1 or code in (400, 403, 404):
                raise
            wait = 60 if code in (429, 504) else 10 * (i + 1)
            print(f"  retry {i + 1} in {wait}s: {url} ({e})", flush=True)
            time.sleep(wait)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def keep_old(path):
    """A refresh replaces a file already on disk: keep the previous vintage in data/external/_superseded/ first."""
    if path.exists():
        import shutil
        dest = EXT / "_superseded" / path.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / f"{path.name}.{datetime.fromtimestamp(path.stat().st_mtime):%Y%m%d}")


def record(src, url, path):
    with open(EXT / src / "files.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"url": url, "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "bytes": path.stat().st_size, "sha256": sha256(path),
                            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")


def stream(src, url, name=None, tries=6):
    """Download url to data/external/<src>/<name>, resuming across restarts by size check."""
    out = EXT / src / (name or url.rsplit("/", 1)[1])
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=600) as r:
                size = int(r.headers.get("Content-Length") or -1)
                if out.exists() and out.stat().st_size == size:
                    return out, "skip"
                part = out.with_suffix(out.suffix + ".part")
                n = 0
                with open(part, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        n += len(chunk)
                if size >= 0 and n != size:
                    raise IOError(f"short read {n}/{size}")
                keep_old(out)
                part.replace(out)
            record(src, url, out)
            return out, "ok"
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None)
            if i == tries - 1 or code in (403, 404):
                raise
            print(f"  retry {i + 1}: {url} ({e})", flush=True)
            time.sleep(15 * (i + 1))


def write_source(src, text):
    (EXT / src).mkdir(parents=True, exist_ok=True)
    p = EXT / src / "SOURCE.md"
    if not p.exists():
        p.write_text(text.strip() + f"\n\nFirst retrieved: {NOW} (per-file retrieval time and sha256 in `files.jsonl`).\n"
                     "Downloaded by `scripts/fetch_external.py`.\n", encoding="utf-8")


def main():
    only = set(sys.argv[1:])
    jobs = []  # (source, label, callable)

    def want(s):
        return not only or s in only

    if want("ourairports"):
        write_source("ourairports", """
# OurAirports
URL: https://ourairports.com/data/ (mirror https://davidmegginson.github.io/ourairports-data/).
Files: airports.csv, runways.csv (thresholds, headings, lengths), navaids.csv. Terms: see docs/data-sources.md.
Use: runway threshold coordinates as a fallback to OSM for taxi-distance features.""")
        for u in OURAIRPORTS:
            jobs.append(("ourairports", u.rsplit("/", 1)[1], lambda u=u: stream("ourairports", u)))

    if want("prc"):
        write_source("prc", """
# EUROCONTROL Performance Review (ansperformance.eu) open data
Pages: https://ansperformance.eu/data/ and https://ansperformance.eu/csv/ .
Yearly CSVs 2024-2026: airport_traffic, apt_dly (airport arrival ATFM delay with delay causes incl.
industrial action, weather, capacity), taxi_out_additional_time, taxi_in_additional_time,
atfm_slot_adherence. XLSX: ATC_Pre-Departure_Delay, All_Pre-Departure_Delay, ASMA_Additional_Time,
Taxi-Out_Additional_Time, Taxi_times_Planning_Data_S14_S22.
Granularity: mostly airport x day (taxi/ASMA monthly). Note 2026 files are the vintage on retrieval date.
Terms: see docs/data-sources.md.""")
        for u in PRC_CSV + PRC_XLS:
            jobs.append(("prc", u.rsplit("/", 1)[1], lambda u=u: stream("prc", u)))

    if want("calendars"):
        write_source("calendars", """
# Public holidays
Generated locally with the Python `holidays` package (pinned in requirements.txt) for
the airports' countries and relevant subdivisions, 2024-2026. No network download.
Output: public_holidays.csv (date, airport, country, subdiv, name).
Not covered: school holidays and ATC strike days (no openly licensed structured source found;
industrial-action ATFM delay causes in prc/apt_dly_*.csv are the proxy).""")
        jobs.append(("calendars", "public_holidays.csv", make_holidays))

    if want("metar"):
        write_source("metar", f"""
# METAR / SPECI weather (Iowa Environmental Mesonet ASOS/AWOS archive)
URL: {IEM}
Query per station: station=<ICAO>&data=all&tz=Etc/UTC&format=onlycomma&latlon=yes&elev=yes&missing=M
&trace=T&report_type=3&report_type=4, one request per station per calendar year, covering
{IEM_START} to {IEM_END} (exclusive). report_type 3 = routine METAR, 4 = SPECI.
Includes the raw METAR string (wx codes, RVR, CB/TCU) plus parsed fields (tmpf F, sknt kt, vsby mi,
sky layers ft). Stations: {', '.join(AIRPORTS)}. IEM data is freely available with no account.""")
        for icao in AIRPORTS:
            for y in range(IEM_START.year, IEM_END.year + 1):
                a, b = max(IEM_START, date(y, 1, 1)), min(IEM_END, date(y + 1, 1, 1))
                if a < b:
                    jobs.append(("metar", f"{icao}_{y}", lambda i=icao, a=a, b=b, y=y: fetch_metar(i, a, b, y)))

    if want("osm"):
        write_source("osm", f"""
# OpenStreetMap airport layout (Overpass API)
URL: {OVERPASS}
Query per airport (ICAO substituted), output with full geometry:
```
{OSM_QUERY}
```
Terms: see docs/data-sources.md. Snapshot of OSM at retrieval date.""")
        for icao in AIRPORTS:
            jobs.append(("osm", icao, lambda i=icao: fetch_osm(i)))

    if want("opdi"):
        write_source("opdi", """
# OPDI - Open Performance Data Initiative (EUROCONTROL PRC + OpenSky Network), v0.0.2
Pages: https://www.opdi.aero/flight-list-data , /flight-event-data , /measurement-data
Pattern: https://www.eurocontrol.int/performance/data/download/OPDI/v002/{flight_list/flight_list_YYYYMM,
flight_events/flight_events_YYYYMMDD_YYYYMMDD+10, measurements/measurements_YYYYMMDD_YYYYMMDD+10}.parquet
Selected: every file overlapping 2024-12-25..2026-02-01 or 2026-07-01..2026-08-01, plus the months of the final inputs that `scripts/fetch_final.py` has downloaded (the
February and June 2026 files are fetched that way; 10-day files cover up to but excluding the end date). ADS-B derived flights, events (first/last seen, take-off, landing, level
crossings, entry/exit of OSM runway, taxiway, apron, parking_position, threshold, hangar, deicing_pad;
`info` JSON carries osm_ref e.g. stand or runway designator and time in use) and measurements, all airports (not filtered to our ten).
CAVEATS: (1) terms of use: see docs/data-sources.md; (2) surface events can partly reconstruct the masked
block time - check the data's terms of use before using them as features.""")
        for u in opdi_urls():
            jobs.append(("opdi", u.rsplit("/", 1)[1], lambda u=u: stream("opdi", u)))

    total, failed = len(jobs), []
    _report(0, total, f"{total} files queued")
    for k, (src, label, fn) in enumerate(jobs, 1):
        try:
            res = fn()
            status = res[1] if isinstance(res, tuple) else "ok"
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            status = f"FAILED {e}"
            failed.append(f"{src}/{label}: {e}")
        _report(k, total, f"{src}/{label} {status}")
    summary = f"done: {total - len(failed)}/{total} ok" + (f"; failed: {failed}" if failed else "")
    _report(total, total, summary, finished=True)
    print(summary, flush=True)
    sys.exit(1 if failed else 0)


def _report(done, total, detail, finished=False):
    print(f"[{done}/{total}] {detail}", flush=True)
    if _progress:
        _progress.report(RUN, "EUROCONTROL external open data download", done, total, detail=detail,
                         finished=finished, purpose="Fetch all external open data candidates "
                         "(METAR, OPDI ADS-B, PRC airport series, OSM layout, OurAirports, holidays)")


def _metar_covers(path, b):
    """True if the file's last report is within 2 days of the (exclusive) end date b: a stale year file is refetched."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4000))
        last = f.read().decode("utf-8", "replace").strip().splitlines()[-1].split(",")
    try:
        ts = datetime.strptime(last[1], "%Y-%m-%d %H:%M")
    except (IndexError, ValueError):
        return True
    return (datetime(b.year, b.month, b.day) - ts).days <= 2


def fetch_metar(icao, a, b, y):
    out = EXT / "metar" / f"{icao}_{y}.csv"
    if out.exists() and out.stat().st_size > 1000 and _metar_covers(out, b):
        return out, "skip"
    b1 = b  # asos.py end date is exclusive at 00:00
    q = [("station", icao), ("data", "all"), ("year1", a.year), ("month1", a.month), ("day1", a.day),
         ("year2", b1.year), ("month2", b1.month), ("day2", b1.day), ("tz", "Etc/UTC"),
         ("format", "onlycomma"), ("latlon", "yes"), ("elev", "yes"), ("missing", "M"), ("trace", "T"),
         ("direct", "no"), ("report_type", 3), ("report_type", 4)]
    url = IEM + "?" + urllib.parse.urlencode(q)
    body = get_bytes(url)
    if body.count(b"\n") < 20 * (b - a).days or not body.startswith(b"station,"):  # ~48 reports/day expected
        raise IOError(f"unexpected IEM response ({len(body)} bytes): {body[:120]!r}")
    part = out.with_suffix(".csv.part")
    part.write_bytes(body)
    keep_old(out)
    part.replace(out)
    record("metar", url, out)
    time.sleep(2)  # IEM asks for sequential, polite use
    return out, f"ok {body.count(b'\n') - 1} rows"


def fetch_osm(icao):
    out = EXT / "osm" / f"{icao}.json"
    if out.exists() and out.stat().st_size > 1000:
        return out, "skip"
    query = OSM_QUERY.replace("{icao}", icao)
    body = get_bytes(OVERPASS, data=urllib.parse.urlencode({"data": query}).encode())
    els = json.loads(body).get("elements", [])
    kinds = {}
    for e in els:
        t = e.get("tags", {}).get("aeroway", "?")
        kinds[t] = kinds.get(t, 0) + 1
    if not kinds.get("aerodrome") or not kinds.get("taxiway"):
        raise IOError(f"incomplete OSM result for {icao}: {kinds}")
    part = out.with_suffix(".json.part")
    part.write_bytes(body)
    part.replace(out)
    record("osm", f"{OVERPASS} (POST data=<query for {icao}>)", out)
    time.sleep(10)  # overpass-api.de fair use
    return out, "ok " + ",".join(f"{k}={v}" for k, v in sorted(kinds.items()))


def make_holidays():
    import csv
    import holidays
    subdiv = {"EDDF": ("DE", "HE"), "EDDM": ("DE", "BY"), "EGLL": ("GB", "ENG"), "EHAM": ("NL", None),
              "LEBL": ("ES", "CT"), "LEMD": ("ES", "MD"), "LFPG": ("FR", None), "LIRF": ("IT", None),
              "LSZH": ("CH", "ZH"), "LTFM": ("TR", None)}
    out = EXT / "calendars" / "public_holidays.csv"
    rows = []
    for apt, (c, s) in subdiv.items():
        cal = holidays.country_holidays(c, subdiv=s, years=range(2024, 2027))
        rows += [(d.isoformat(), apt, c, s or "", n) for d, n in sorted(cal.items())]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "airport", "country", "subdiv", "name"])
        w.writerows(rows)
    record("calendars", f"python holidays=={holidays.__version__}", out)
    return out, f"ok {len(rows)} rows"


if __name__ == "__main__":
    main()
