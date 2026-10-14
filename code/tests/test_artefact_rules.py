# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the three rules that read the plan-attachment window (|BLOCK - LOBT| <= 3606 s), on a few synthetic rows each:
the twin window (overnight/twin_window.py), the duplicated-record pair (overnight/dup_pair.py) and the plan window / witness push
(scripts/witness_post.py). They need no data and no GPU and run in seconds:  make test   (or  python -m unittest discover -s tests)
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "overnight"), str(ROOT / "scripts")]

import dup_pair  # noqa: E402
import twin_window  # noqa: E402
import witness_post  # noqa: E402


class TwinWindow(unittest.TestCase):
    def tw(self, apt, yl, ya, d):
        return pd.DataFrame({"MVT_ID_mvt": [1], "apt": [apt], "yl": [yl], "ya": [ya], "d": [d]})

    def run_step(self, apt, yl, ya, d, pred, reg=900.0, q=1.0):
        p = pd.DataFrame({"MVT_ID_mvt": [1], "pred": [pred]})
        g = pd.DataFrame({"MVT_ID_mvt": [1], "reg": [reg]})
        return twin_window.step(p, g, self.tw(apt, yl, ya, d), q).iloc[0]

    def test_copy_cell_moves_to_schedule_copy(self):
        # actual off-block inside the plan window (|ya - yl| <= 3606), d beyond the window edge: the schedule copy
        r = self.run_step("LIRF", yl=1000, ya=1200, d=5000, pred=1500)
        self.assertEqual(r.cell, "copy")
        self.assertAlmostEqual(r.new, 5000)

    def test_copy_share_q_blends(self):
        r = self.run_step("LIRF", yl=1000, ya=1200, d=5000, pred=1500, q=0.5)
        self.assertAlmostEqual(r.new, 0.5 * 5000 + 0.5 * 1500)

    def test_normal_cell_moves_to_regular_value(self):
        # the actual off-block is outside the window: that alone explains the failed join, so no copy
        r = self.run_step("LIRF", yl=1000, ya=1000 + 4000, d=5000, pred=2500, reg=900)
        self.assertEqual(r.cell, "normal")
        self.assertAlmostEqual(r.new, 900)

    def test_other_airports_untouched(self):
        r = self.run_step("EGLL", yl=1000, ya=1200, d=5000, pred=1500)
        self.assertEqual(r.cell, "")
        self.assertAlmostEqual(r.new, 1500)

    def test_prediction_already_at_the_copy_is_left(self):
        r = self.run_step("LIRF", yl=1000, ya=1200, d=5000, pred=5000 - 100)   # within AT_COPY (600 s)
        self.assertEqual(r.cell, "")

    def test_copy_beyond_the_cap_is_left(self):
        r = self.run_step("LIRF", yl=1000, ya=1200, d=40000, pred=1500)         # d > DMAX (30,000 s)
        self.assertEqual(r.cell, "")

    def test_no_twin_no_change(self):
        p = pd.DataFrame({"MVT_ID_mvt": [7], "pred": [1500.0]})
        g = pd.DataFrame({"MVT_ID_mvt": [7], "reg": [900.0]})
        r = twin_window.step(p, g, self.tw("LIRF", 1000, 1200, 5000), 1.0).iloc[0]
        self.assertAlmostEqual(r.new, 1500)

    def test_twins_pairing_rules(self):
        # a no-NM departure and its arrival twin (same flight, ADEP, ADES; arrival 10 min to 8 h later) with an NM flight
        base = dict(FLIGHT_mvt="AZ1", ADEP_mvt="LIRF", ADES_mvt="LFPG")
        cols = twin_window.COLS
        rows = [
            dict(MVT_ID_mvt=1, FLIGHT_ID_mvt=np.nan, PHASE_mvt="DEP", MVT_TIME_UTC_mvt=10_000.0, SCHED_TIME_UTC_mvt=6_000.0,
                 LOBT_flt=np.nan, AOBT_3_flt=np.nan, **base),
            dict(MVT_ID_mvt=2, FLIGHT_ID_mvt=5.0, PHASE_mvt="ARR", MVT_TIME_UTC_mvt=10_000.0 + 7_200, SCHED_TIME_UTC_mvt=0.0,
                 LOBT_flt=8_000.0, AOBT_3_flt=9_000.0, **base),
            # the destination must itself be a departure airport of the file (a challenge airport)
            dict(MVT_ID_mvt=4, FLIGHT_ID_mvt=9.0, PHASE_mvt="DEP", MVT_TIME_UTC_mvt=50_000.0, SCHED_TIME_UTC_mvt=49_000.0,
                 LOBT_flt=49_000.0, AOBT_3_flt=49_100.0, FLIGHT_mvt="AF9", ADEP_mvt="LFPG", ADES_mvt="EGLL"),
            # a second arrival 5 min later: too close to be this flight's twin
            dict(MVT_ID_mvt=3, FLIGHT_ID_mvt=6.0, PHASE_mvt="ARR", MVT_TIME_UTC_mvt=10_000.0 + 300, SCHED_TIME_UTC_mvt=0.0,
                 LOBT_flt=1.0, AOBT_3_flt=2.0, **base),
        ]
        df = pd.DataFrame(rows)[cols]
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m.parquet"
            df.to_parquet(p)
            out = twin_window.twins([p])
        self.assertEqual(len(out), 1)
        r = out.iloc[0]
        self.assertEqual((r.MVT_ID_mvt, r.apt), (1, "LIRF"))
        self.assertAlmostEqual(r.yl, 2_000.0)       # MVT - LOBT of the twin
        self.assertAlmostEqual(r.ya, 1_000.0)       # MVT - AOBT_3 of the twin
        self.assertAlmostEqual(r.d, 4_000.0)        # MVT - SCHED of the departure


class DupPair(unittest.TestCase):
    def movements(self, rows):
        cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt",
                "TAXITIME_SEC_mvt", "split"]
        return pd.DataFrame(rows, columns=cols)

    def test_pairs_finds_matched_half_of_a_close_pair(self):
        t0 = 1.7e9
        rows = [
            [1, "DEP", "LFPG", "AF1", None, t0, t0 - 3000, t0 - 2000, 2500.0, "train"],       # the NM-matched half
            [2, "DEP", "LFPG", "AF1", None, t0 + 20, np.nan, np.nan, 1100.0, "train"],        # its twin record, no NM
            [3, "DEP", "LFPG", "AF2", None, t0, t0 - 1000, t0 - 900, 900.0, "train"],         # a lone departure
            [4, "DEP", "LFPG", "AF3", None, t0, t0 - 3000, t0 - 2000, 800.0, "train"],        # same flight number, hours apart
            [5, "DEP", "LFPG", "AF3", None, t0 + 5 * 3600, t0 + 4 * 3600, t0 + 4 * 3600, 800.0, "train"],
            [6, "ARR", "LFPG", "AF1", None, t0, t0 - 3000, t0 - 2000, np.nan, "train"],       # arrivals are not pairs
        ]
        with tempfile.TemporaryDirectory() as t:
            self.movements(rows).to_parquet(Path(t) / "movements.parquet")
            old, dup_pair.PROC = dup_pair.PROC, Path(t)
            try:
                p = dup_pair.pairs(300)
            finally:
                dup_pair.PROC = old
        self.assertEqual(p.MVT_ID_mvt.tolist(), [1])
        self.assertAlmostEqual(p.dlo.iloc[0], 3000.0)

    def test_fit_recovers_a_line(self):
        t = pd.DataFrame({"dlo": [0.0, 1000.0, 2000.0, 3000.0]})
        t["TAXITIME_SEC_mvt"] = 300 + 0.8 * t.dlo
        c = dup_pair.fit(t)
        np.testing.assert_allclose(c, [300.0, 0.8], atol=1e-6)


class PlanWindow(unittest.TestCase):
    def ctx(self, **kw):
        base = dict(apt="LIRF", matched=False, reg="irregular", ds=np.nan, al="AZA", off=0.0, w_lobt=np.nan, mvt=100_000.0, lobt=np.nan, y=np.nan)
        base.update(kw)
        return pd.DataFrame([base], index=pd.Index([1], name="MVT_ID_mvt"))

    PAR = {"m": 500.0, "n_zero_floor": 10, "pooled": 0.5, "c_al": {}}

    def run_rule(self, ctx, pred, ideas, win=3606):
        new, moved = witness_post.apply(pd.Series([pred], index=[1]), ctx, self.PAR, ideas, win)
        return float(new.iloc[0]), moved

    def test_plan_window_clip_upper(self):
        # matched row: pred must lie in [MVT - LOBT - W, MVT - LOBT + W]
        c = self.ctx(matched=True, lobt=100_000.0 - 2000)             # MVT - LOBT = 2000
        v, m = self.run_rule(c, 9000.0, "3")
        self.assertAlmostEqual(v, 2000 + 3606)
        self.assertEqual(m["3"], 1)

    def test_plan_window_clip_lower_floor_zero(self):
        c = self.ctx(matched=True, lobt=100_000.0 - 1000)             # lower edge 1000 - 3606 < 0 -> 0
        v, _ = self.run_rule(c, -50.0, "3")
        self.assertAlmostEqual(v, 0.0)

    def test_plan_window_inside_is_unchanged(self):
        c = self.ctx(matched=True, lobt=100_000.0 - 2000)
        v, m = self.run_rule(c, 2500.0, "3")
        self.assertAlmostEqual(v, 2500.0)
        self.assertEqual(m["3"], 0)

    def test_plan_window_clip_uses_the_final_width(self):
        c = self.ctx(matched=True, lobt=100_000.0 - 2000)
        v, _ = self.run_rule(c, 9000.0, "3", win=3612)
        self.assertAlmostEqual(v, 2000 + 3612)

    def test_unmatched_rows_not_clipped(self):
        c = self.ctx(matched=False, lobt=100_000.0 - 2000)
        v, _ = self.run_rule(c, 9000.0, "3")
        self.assertAlmostEqual(v, 9000.0)

    def test_zero_floor_lifts_lirf_to_above_the_window(self):
        # witness LOBT leaves the window's lower edge at <= 60 s: the label lies above hi; lift to hi + m
        c = self.ctx(w_lobt=100_000.0 - 1000)                          # dl = 1000, lo = -2600, hi = 4600
        v, m = self.run_rule(c, 1500.0, "1")
        self.assertAlmostEqual(v, 1000 + witness_post.WINDOW + 500.0)
        self.assertEqual(m["1_zero_floor"], 1)

    def test_zero_floor_leaves_other_airports(self):
        c = self.ctx(apt="EGLL", w_lobt=100_000.0 - 1000)
        v, _ = self.run_rule(c, 1500.0, "1")
        self.assertAlmostEqual(v, 1500.0)

    def test_far_side_moves_to_the_nearer_edge(self):
        c = self.ctx(apt="EGLL", w_lobt=100_000.0 - 8000)              # dl = 8000, window [4400, 11600]
        v, m = self.run_rule(c, 5000.0, "1")
        self.assertAlmostEqual(v, 4400.0)
        self.assertEqual(m["1_far_side"], 1)
        v2, _ = self.run_rule(c, 11000.0, "1")
        self.assertAlmostEqual(v2, 11600.0)

    def test_far_side_outside_window_is_left(self):
        c = self.ctx(apt="EGLL", w_lobt=100_000.0 - 8000)
        v, _ = self.run_rule(c, 2000.0, "1")
        self.assertAlmostEqual(v, 2000.0)

    def test_no_ideas_no_change(self):
        c = self.ctx(matched=True, lobt=100_000.0 - 2000, w_lobt=100_000.0 - 1000)
        v, m = self.run_rule(c, 9000.0, "")
        self.assertAlmostEqual(v, 9000.0)
        self.assertEqual(m, {})


if __name__ == "__main__":
    unittest.main()
