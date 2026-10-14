# SPDX-License-Identifier: GPL-3.0-or-later
"""Check a prediction file before it ships, read-only.

  1. template: exactly the two template columns in order, the template's ids in its order (unique), finite non-negative seconds
     (scripts/baseline.py validate_submission; EC_PHASE=final reads data/raw/final_submitting.parquet);
  2. recipe chain (--id): every step the file is built from has its recorded configuration in recipes/ (the walk follows every
     parent link of each recipe: base, src, regs, chain, blend, ...), so each number in the file traces to a recorded step;
  3. hash (--sha256_file): the file's sha256 equals the line for its name in that file (final/SHA256).

  EC_PHASE=final python scripts/check_final.py outputs/overnight_final/test/ar_134naclh_tww_wit_zf3.parquet --id ar_134naclh_tww_wit_zf3
  python scripts/check_final.py final/<file>.parquet --sha256_file final/SHA256          # the shipped file against its recorded hash
Exit 1 on any failure; prints what it checked.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes"
PARENTS = ("base", "p_from", "src", "regs", "chain", "blend", "a", "b", "q_cache", "reg_from", "aux", "rank", "ids", "parts", "test_raw")


def recipe(i):
    # the month-complete stackers are named <id>_<x|c|l|w>_m12 in the chain, their recipes <id>_<x|c|l|w>.json
    names = [i] + ([i[:-4]] if i.endswith("_m12") else [])
    return next((d / f"{n}.json" for n in names for d in (RECIPES, RECIPES / "other_chains") if (d / f"{n}.json").exists()), None)


def walk(cv_id):
    """Every step reachable from cv_id through parent links; returns (steps with a recipe, steps without one)."""
    seen, missing, todo = set(), [], [cv_id]
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        p = recipe(i)
        if p is None:
            missing.append(i)
            continue
        d = json.loads(p.read_text())
        if not isinstance(d, dict):
            continue
        for k in PARENTS:
            v = d.get(k)
            vals = v if isinstance(v, list) else (v.split(",") if isinstance(v, str) else [])
            todo += [x for x in vals if isinstance(x, str) and x and x not in seen]
    return sorted(seen - set(missing)), missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--id", help="chain id of the file (its recipe in recipes/)")
    ap.add_argument("--sha256_file", help="file of '<sha256>  <name>' lines (final/SHA256)")
    ap.add_argument("--no_template", action="store_true", help="skip check 1 (no challenge data on this machine)")
    a = ap.parse_args()
    src = Path(a.file).resolve()
    bad = []
    if not a.no_template:
        sys.path.insert(0, str(ROOT / "scripts"))
        from baseline import validate_submission
        try:
            validate_submission(src)
        except (ValueError, FileNotFoundError) as e:
            bad.append(f"template: {e}")
    if a.id:
        ok, missing = walk(a.id)
        print(f"recipe chain {a.id}: {len(ok)} steps with a recorded configuration")
        bad += [f"recipe chain: step {m} has no recipe in recipes/" for m in missing]
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"sha256 {sha}  {src.name}")
    if a.sha256_file:
        rec = {l.split()[1]: l.split()[0] for l in Path(a.sha256_file).read_text().splitlines() if len(l.split()) == 2}
        if rec.get(src.name) != sha:
            bad.append(f"sha256: {src.name} is {sha}, {a.sha256_file} records {rec.get(src.name)}")
    for b in bad:
        print("FAIL", b)
    print("OK" if not bad else f"{len(bad)} failure(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
