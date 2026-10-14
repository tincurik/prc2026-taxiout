#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fetch the adsb.lol ground points for the days you name: resumable, checksummed, with the cost stated before it starts.

  make adsb DAYS=2026-01-05:2026-01-11          a range of days (end included)
  make adsb DAYS=2026-01                        a calendar month
  make adsb DAYS=2026-01-05,2026-02-10,2026-03  any mix, comma separated
  make adsb DAYS=2026-01 PLAN=1                 print the plan and the cost; fetch nothing
  make adsb DAYS=2026-01 VERIFY=1               re-hash the files on disk against the manifest; fetch nothing

What it does, per day: `scripts/adsb_fetch.py` streams the day's archive from the adsb.lol GitHub releases (nothing large touches the disk) and writes
`data/external/adsblol/points/<day>.parquet` (the ground and terminal-area points of the ten airports, a few hundred MB at most).
This wrapper adds three things around it:
  * resumable: a day whose file exists and matches its recorded sha256 is skipped; a file with a different hash, or one that cannot be read, is deleted and fetched again;
  * checksums: `data/external/adsblol/points.sha256` holds `<sha256>  <day>.parquet` for every day fetched (the same format as `sha256sum -c`);
  * cost: the plan lists days to fetch, GB streamed, disk and time, from the 96 days the recorded run fetched (median 3.5 GB read, 212 MB written, 572 s per day on one worker).
The checksums guard the files on your disk (a cut-off download, a changed filter); they cannot say that the upstream archive is unchanged, because the archive itself
publishes no checksum. A day with no archive upstream is reported and left out; the features then see no track for it.
"""
import argparse
import datetime as dt
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "data/external/adsblol/points"
MANIFEST = POINTS.parent / "points.sha256"
GB_READ, MB_WRITTEN, SEC_PER_DAY = 3.5, 212, 572     # medians of the 96 recorded days (one worker); the maxima were 4.3 GB, 242 MB and 1,352 s


def days_of(spec):
    out = []
    for part in [p for p in spec.split(",") if p]:
        if ":" in part:
            a, b = (dt.date.fromisoformat(x) for x in part.split(":"))
            out += [(a + dt.timedelta(n)).isoformat() for n in range((b - a).days + 1)]
        elif len(part) == 7:
            d = dt.date.fromisoformat(part + "-01")
            while d.month == dt.date.fromisoformat(part + "-01").month:
                out.append(d.isoformat())
                d += dt.timedelta(1)
        else:
            out.append(dt.date.fromisoformat(part).isoformat())
    return sorted(set(out))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest():
    if not MANIFEST.exists():
        return {}
    return {l.split()[1].removesuffix(".parquet"): l.split()[0] for l in MANIFEST.read_text().splitlines() if len(l.split()) == 2}


def write_manifest(m):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("".join(f"{h}  {n}.parquet\n" for n, h in sorted(m.items())))


def readable(p):
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(p).metadata.num_rows >= 0
    except Exception:  # noqa: BLE001  (a cut-off file)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("days", help="YYYY-MM-DD, YYYY-MM-DD:YYYY-MM-DD, YYYY-MM or a comma separated mix")
    ap.add_argument("--workers", type=int, default=3, help="days fetched at once; each reads a 2-4 GB archive (default 3)")
    ap.add_argument("--plan", action="store_true", help="print the plan and the cost, fetch nothing")
    ap.add_argument("--verify", action="store_true", help="re-hash the files of these days against the manifest, fetch nothing")
    a = ap.parse_args()
    days = days_of(a.days)
    man = read_manifest()
    have, redo = [], []
    for d in days:
        p = POINTS / f"{d}.parquet"
        if not p.exists():
            redo.append(d)
        elif d in man and sha256(p) != man[d]:
            redo.append(d)
            print(f"{d}: file differs from the manifest")
        elif not readable(p):
            redo.append(d)
            print(f"{d}: file cannot be read")
        else:
            have.append(d)
    n = len(redo)
    print(f"{len(days)} days ({days[0]} .. {days[-1]}): {len(have)} on disk and verified, {n} to fetch")
    print(f"cost of the {n} days to fetch: about {n * GB_READ:,.0f} GB read from the network (streamed, not stored), about {n * MB_WRITTEN / 1024:,.1f} GB written to "
          f"{POINTS.relative_to(ROOT)}, about {n * SEC_PER_DAY / 3600 / max(a.workers, 1):,.1f} h with {a.workers} workers (about {SEC_PER_DAY / 60:.0f} min a day each; "
          f"this depends on the network)")
    if a.verify:
        bad = [d for d in redo if (POINTS / f"{d}.parquet").exists()]
        print("verify:", "all present days match" if not bad else f"mismatch: {bad}")
        for d in have:
            man.setdefault(d, sha256(POINTS / f"{d}.parquet"))
        write_manifest(man)
        return 1 if bad else 0
    if a.plan or not n:
        return 0
    for d in redo:
        (POINTS / f"{d}.parquet").unlink(missing_ok=True)
    rc = subprocess.call([sys.executable, str(ROOT / "scripts/adsb_fetch.py"), "--days", *redo, "--workers", str(a.workers)])
    got = [d for d in redo if (POINTS / f"{d}.parquet").exists()]
    for d in got:
        man[d] = sha256(POINTS / f"{d}.parquet")
    for d in have:
        man.setdefault(d, sha256(POINTS / f"{d}.parquet"))
    write_manifest(man)
    missing = [d for d in redo if d not in got]
    print(f"fetched {len(got)} of {n} days; manifest {MANIFEST.relative_to(ROOT)} holds {len(man)} days" + (f"; no file for {missing}" if missing else ""))
    return 1 if missing or rc else 0


if __name__ == "__main__":
    sys.exit(main())
