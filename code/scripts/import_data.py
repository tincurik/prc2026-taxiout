# SPDX-License-Identifier: GPL-3.0-or-later
"""Import the challenge files into data/raw/ from the organisers' archive (a ZIP) or from a directory that already holds them, without overwriting
changed data.

The required files are the twelve training months, ranking.parquet and submitting.parquet (14 in all).  The archive may hold more (they are
skipped and listed) and may keep the files in a sub-folder (matched by file name); if a required file is missing the script stops and names it.

  python scripts/import_data.py <files_list.zip | directory>
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"ranking.parquet", "submitting.parquet"} | {
    f"training_2025-{m:02d}-01_{'2026-01' if m == 12 else f'2025-{m+1:02d}'}-01.parquet"
    for m in range(1, 13)
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources(archive):
    """-> ({required name: (opener, size)}, [names skipped]) for a ZIP file or a directory."""
    if archive.is_dir():
        found = {p.name: p for p in archive.rglob("*.parquet")}
        have = {n: (lambda p=found[n]: p.open("rb"), found[n].stat().st_size) for n in found if n in EXPECTED}
        return have, sorted(set(found) - EXPECTED)
    zf = zipfile.ZipFile(archive)
    entries = {Path(i.filename).name: i for i in zf.infolist() if not i.is_dir()}
    have = {n: ((lambda i=entries[n]: zf.open(i)), entries[n].file_size) for n in entries if n in EXPECTED}
    return have, sorted(set(entries) - EXPECTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, help="the challenge ZIP, or a directory holding the parquet files")
    args = parser.parse_args()
    if not args.archive.exists():
        raise SystemExit(f"import_data: {args.archive} does not exist")
    have, skipped = sources(args.archive)
    missing = sorted(EXPECTED - set(have))
    if missing:
        raise SystemExit(f"import_data: {args.archive} lacks {len(missing)} of the {len(EXPECTED)} required files: {', '.join(missing)}")
    if skipped:
        print(f"skipping {len(skipped)} file(s) that are not part of the challenge set: {', '.join(skipped)}")
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for name in sorted(have):
        opener, size = have[name]
        target = raw / name
        temporary = target.with_suffix(".parquet.part")
        try:
            with opener() as source, temporary.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            digest = sha256(temporary)
            if target.exists():
                if sha256(target) != digest:
                    raise ValueError(f"Refusing to overwrite changed data: {target}")
            else:
                temporary.replace(target)
            records.append({"file": target.relative_to(ROOT).as_posix(), "bytes": size, "sha256": digest})
        finally:
            temporary.unlink(missing_ok=True)
    manifest = {"imported_utc": datetime.now(timezone.utc).isoformat(), "archive_name": args.archive.name,
                "archive_sha256": sha256(args.archive) if args.archive.is_file() else None, "files": records}
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(records)} files in {raw}")


if __name__ == "__main__":
    main()
