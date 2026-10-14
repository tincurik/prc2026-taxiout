# SPDX-License-Identifier: GPL-3.0-or-later
"""Save the ansperformance.eu airport dashboards (10 airports) and extract their embedded download tables.

Each chart's "download CSV" button carries its data inline as `data:text/csv;charset=utf-8,<csv>`; that is the
only machine-readable source of the per-runway taxi-out / taxi-in / ASMA tables (monthly, source APDF).
Output: data/external/prc_dashboard/<APT>.html (raw) and one CSV per table id (all airports stacked).

  python scripts/fetch_prc_dashboard.py
"""
import datetime as dt
import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/prc_dashboard"
APTS = "EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM".split()


def tables(html):
    for m in re.finditer(r"data:text/csv;charset=utf-8,(.*?)'", html):
        s = m.group(1).replace('\\\\"', '"').replace('\\"', '"').replace("\\\\\\\\n", "\n").replace("\\\\n", "\n")
        try:
            d = pd.read_csv(io.StringIO(s))
        except Exception:  # noqa: BLE001
            continue
        if len(d):
            yield d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs, found = [], {}
    for a in APTS:
        url = f"https://ansperformance.eu/dashboard/stakeholder/airport/db/{a}.html"
        h = requests.get(url, timeout=120).text
        (OUT / f"{a}.html").write_text(h, encoding="utf-8")
        recs.append({"url": url, "path": f"data/external/prc_dashboard/{a}.html", "bytes": len(h.encode()),
                     "sha256": hashlib.sha256(h.encode()).hexdigest(),
                     "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        for d in tables(h):
            key = "_".join(c for c in d.columns if c not in ("AIRPORT", "APT_ICAO", "YEAR", "MONTH", "MONTH_NUM"))[:70]
            if "AIRPORT" not in d.columns and "APT_ICAO" not in d.columns:
                d.insert(0, "AIRPORT", a)
            found.setdefault(key, []).append(d)
    (OUT / "files.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    index = []
    for key, frames in found.items():
        d = pd.concat(frames, ignore_index=True).drop_duplicates()
        fn = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_").lower() + ".csv"
        d.to_csv(OUT / fn, index=False)
        index.append({"file": fn, "rows": len(d), "columns": list(d.columns)})
        print(f"{fn:60s} {len(d):6d} rows  {list(d.columns)}")
    (OUT / "tables.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
