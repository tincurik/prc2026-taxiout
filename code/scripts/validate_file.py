# SPDX-License-Identifier: GPL-3.0-or-later
"""Check a prediction file against the template of its phase: exactly the two template columns in order, the template's ids in the
template's order (unique), finite non-negative seconds.

  python scripts/validate_file.py <predictions.parquet>        # EC_PHASE=final (or pseudoN) picks final_submitting.parquet of that phase
"""
import sys

from baseline import validate_submission

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    validate_submission(sys.argv[1])
