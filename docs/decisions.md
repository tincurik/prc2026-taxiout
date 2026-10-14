# Decision register: every hand-set constant and what backs it

The model has a few dozen constants that were chosen, not learned by the main models. This page lists each with its status, so a reader can see which ones rest on a measured fact,
which were fitted, and which were set once and never varied. It is a register of the constants named in [model](model.md); it adds no new measurements.

**Statuses.**

- **fact**: a recorded property of the 2025 data that the constant only states (an edge, a label value); there was no free choice to validate.
- **fitted**: a parameter estimated on 2025 rows with the scored months left out (out of fold), by least squares or by a fixed rule.
- **set on the folds**: chosen on the January and July folds, then scored on the five rehearsals (twelve months in all) it was not set on.
- **set by hand**: chosen once from the 2025 rows and never varied; no run compares it with another value.

"Alternatives tried" lists only runs that exist in the [results](results.md); "none" means none was run.

| constant or step | where | status | checked off the data it was set on? | alternatives tried |
|---|---|---|---|---|
| copy tolerance, `\|BLOCK − SCHED\| ≤ 30 s` defines a copy | `overnight/mix.py`, [model](model.md#the-mixture) | fact (the recorded copies sit at 0 to a few seconds) | the calibration of `p` against the observed copy share, every bin shown | none |
| plan cliff 3,600 s: a copy cannot happen once the plan's `LOBT` is more than that after the schedule | `overnight/rules_chain.py --cliff` | fact: 0 copies in 66,025 rows of 2025 above it, 69 in the bin just below | no | none |
| plan window 3,606 s | `scripts/witness_post.py`, `overnight/twin_window.py` | fact: a flight plan is attached only where `\|BLOCK − LOBT\| ≤ 3,606 s`, true for all 2,062,577 planned departures | on 2025 only. `BLOCK` is masked in the test months, so the window cannot be checked there; the step assumes the 2026 files were joined the same way | the clip edge of the first witness rule is 3,600 s: two constants fixed when each rule was built, kept as built |
| date-shift threshold: `MVT − AOBT_3` above 20,000 s is a record a day early, predicted as `MVT − AOBT_3` | `overnight/clockfix.py` | set by hand | no | none |
| clock-gap hold: gap 4,800 s, 4,000 s, weight 0.5 (moves the prediction halfway to `MVT − LOBT`) | `overnight/clockfix.py` | retuned by hand on the 2025 out-of-fold rows | no | none |
| LIRF long-copy band 20,000 to 50,000 s after schedule, no plan: predict `MVT − SCHED` | `overnight/clockfix.py` | fact: 48 of 50 such rows in 2025 are copies | no | none |
| delayed-row gate weight 0.75 at LIRF, 0 elsewhere | `overnight/gate2.py` (grid of 0, 0.25, 0.5, 0.75, 1 for LIRF and for the others) | set on the folds | partly: the rehearsals | the best weight is 0.5 for January and 1.0 for July; 0.75 sits between them |
| LIRF no-plan copy classifier, weight 0.5 | `overnight/lirf_far.py` | set on the folds | weak: over six held-out pairs −0.01 RMSE, June 2025 +1,492 MSE (interval includes zero); the fold gain also drops from −1.23 to −0.47 when refitted with buffer months ([model](model.md#four-post-steps-on-the-chain-output)) | weight not swept |
| LIRF near-schedule group mean | `overnight/lirf_bel.py` | fitted on 20 rows | no: the folds, then twelve months (−0.27 [−0.49, −0.08]) | none |
| far-LIRF record constant 87,500 s | `overnight/hg_far50.py`, [model](model.md#four-post-steps-on-the-chain-output) | fact: one day (86,400 s) plus a typical taxi of about 1,100 s; every such 2025 row is the copy or this value | no: 3 of 3 above a copy probability of 0.5 are copies, 2 of 13 below | none |
| duplicated-record rule, `a + b·(MVT − LOBT)`, `a = 120 s`, `b = 0.90` | `overnight/dup_pair.py` | fitted on 118 pairs of 2025, scored month left out | yes: out of month | none |
| daily-delay calibration: factor 0.6; hedge thresholds 4,000 s and 1,200 s; z clip ±2; coefficient `b` | `overnight/dly_cal.py` | factor, thresholds and clip set by hand on 2025; `b` fitted on 2025 out-of-fold rows | partly: twelve months, worse without it in all twelve ([daily delay file](daily-delay-file.md)) | none beyond leaving the step out |
| residual clips ±5,000 s (training), ±3,600 s (early stopping), ±1,800 s (stacker target); 1.15 × iterations for refits | [model](model.md#the-mixture) | set on the folds | no | none |
| runway-calibration step: a quarter of the correction, a 150 s cap (left out of the final chain) | `overnight/prc_cal2.py` | set on the folds | six held-out pairs: gains in all six, −0.03 RMSE | a half instead of a quarter outside winter moves CV by +0.01 |
| plan-window step: lift margin 1,230 s for LIRF unmatched rows | `scripts/witness_post.py` | fitted: the median distance above the window in the other 2025 months | no: all 30 such LIRF rows of 2025 had their label above it | none |
| LIRF copy re-mix: acts where the plan copy and the rules-chain value differ by more than 600 s, schedule delay below 20,000 s and `0.02 < p_eff < 0.98`; schedule-delay bands at 1,800, 3,600, 7,200 and 20,000 s, plan / no plan; at least 200 rows per cell | `overnight/r16_remix.py` | cut-offs and bands set by hand; the slope and intercept of each cell fitted on the twelve-month pool, leaving out the folds that hold the row's own months | the step's twelve-month score, −0.16 [−0.27, −0.06]; the cut-offs were not varied | none |
| runway-neighbour level: windows ±300, ±900, ±1,800 s on the runway and ±900 s on the airport; shrinkage of each mean towards 0 with 3 pseudo-rows; target `clip(pred − y, ±1,800 s)`; LightGBM with 15 leaves, 300 rounds, learning rate 0.05, at least 2,000 rows per leaf | `overnight/r17_rwnb.py` | windows and settings set by hand; the model fitted on the twelve-month pool, own months' folds left out | the step's twelve-month score, −0.23 [−0.28, −0.19] | a linear fit on the same columns, worse than LightGBM in every fold |
| far-LIRF step cut-offs: plan copy above 50,000 s, prediction more than 600 s from the copy; copy probability split at 0.5, each bin's copy share smoothed as (k + 0.5) / (n + 1) | `overnight/hg_far50.py` | set by hand; the two bins' shares counted on the pool, own months' folds left out (16 rows in 2025) | twelve months −0.04 [−0.17, +0.10] (the interval includes zero) | none |
| arrival-twin window: arrival 10 min to 8 h after the departure; copy cell where the twin's two clocks agree within 3,606 s and the schedule delay is at most 30,000 s; normal cell moves rows more than 300 s from the rules-chain value; weight `q = 1` | `overnight/twin_window.py` | set by hand; `q = 1` against a leave-one-month-out copy share of 0.82–0.90 | twelve months −0.14 [−0.23, −0.05] | none |
| ADS-B match: a departure is *matched* when an ADS-B take-off lies within 300 s of its recorded take-off | `overnight/adsb_stack.py` | set by hand | no | none |
| ADS-B events: callsign search ±90 min, type-and-time match within 120 s, airborne at 90 kt or more from 10 min before take-off, segments cut at 15-min gaps, movement at 1 and 3 kt, displacement 15 and 30 m, roll below 30 kt, stops of 20 s or more, parked below 1 kt within 60 m of the stand, stand positions from at least 3 sightings | `overnight/features_adsb2.py` | set by hand from a design study on two days of 2025 (15 January, 15 July) at six airports | no; the features are label-free inputs, and the models learn how much to trust them | none |
| stacker training rows: non-copy rows with label up to 20,000 s | `overnight/adsb_stack.py` | set by hand | no | none |
| density cap: clip two ADS-B track-density features of 2026 rows to their 2025 range per airport (1st and 99th percentile) | `overnight/density_cap.py` | fitted on 2025 inputs (no label); **not in the shipped chain**, an optional switch | no | the shipped chain runs uncapped |

**Reading it.** Of the 23 rows, five are facts of the 2025 data (copy tolerance, plan cliff, plan window, long-copy band, far-LIRF constant), four are fitted out of month (group mean on 20 rows, duplicate rule, lift margin, density cap), and fourteen are set by hand or on the folds. The four post-steps fit their coefficients out of fold, but their cut-offs, windows and bands were set by hand and never varied. Most hand-set constants were chosen on the same 2025 months that later score them; the rehearsals and the twelve-month score are the only checks on months a constant was not set on. The weakest are the LIRF no-plan classifier, the far-LIRF step and the hand-set clocks of the delayed-row rules; all act on LIRF rows, and [limits](limits.md) says how far the Rome rows can be trusted.

See also: [model](model.md), [validation](validation.md), [limits](limits.md).

## Why the runway-level step stays

The runway-neighbour step is worse on both validation folds and is still in the shipped chain. The recipe `recipes/ar_134naclh_rwnb.json` records the folds: January 320.776 → 320.840 (+0.064), July 247.418 → 247.515 (+0.097), both months together 282.52 → 282.60 (+0.08, 0.03 % of the RMSE). No interval was computed for the step on the two folds.

It stays because the chain judges post-steps on the twelve-month score, not on two months ([validation](validation.md#where-the-numbers-are-optimistic)): −0.23 [−0.28, −0.19], better in 12 of 12 months, measured on an earlier version of the chain. That score is optimistic, because the pool the step fits on keeps the scored month's neighbours, and the two folds show no gain. So the step's effect on the final RMSE lies between +0.08 and −0.23, under 0.1 % of the score either way.

## Data checks

Each check is a rule or a feature in the code ([`code/`](../code/README.md)); none reads the label of a test-month flight.

| Check | Rule | Code |
|---|---|---|
| Date shift | take-off minus the Network Manager off-block clock above 20,000 s (5.5 h): the records are a day early; predict the clock difference | `overnight/clockfix.py` |
| Plan window | a planned departure's off-block lies within 3,606 s of the plan's last filed time (all 2,062,577 plan rows of 2025). A LIRF departure with no plan whose arrival twin (same flight number and airport pair, landing 10 min to 8 h later) has one failed that join; the twin's clocks decide whether its record is a copy or the real off-block | `overnight/twin_window.py` |
| Duplicated take-offs | take-offs seconds apart for one flight number at one airport are one take-off recorded twice, mostly at LFPG; the half with a Network Manager off-block follows the plan, and the model replaces its prediction | `overnight/dup_pair.py` |
| ADS-B track | needs an airborne point (at least 90 kt) from 10 min before to 15 min after the recorded take-off; 15-min gaps cut segments; parked = within 60 m of the stand at under 1 kt | `overnight/features_adsb2.py` |
| LIRF copies | no plan and 20,000 to 50,000 s after schedule: a copy in 48 of 50 cases in 2025 | `overnight/clockfix.py` |
| Coverage | a feature file or member covering a test month at under 0.8 times its training share stops the build | `scripts/final_coverage.py` |
| Prediction file | template ids in order, finite non-negative seconds, recipe chain, sha256 | `scripts/check_final.py` |
