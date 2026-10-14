# Results

The tables by step, airport and month use the *reference chain* (CV 285.44), whose every step was measured on its own. It is
the final model without its last additions (context features, month-complete stackers, stream-transformer inputs, the four
post-steps) and with the LIRF plan-row classifier the final model drops. The last rows of the progress table and the sections
[The final chain's last additions](#the-final-chains-last-additions) and [Final chain by month](#final-chain-by-month) cover the
final model's own steps.

Unless a section says otherwise, numbers are RMSE in seconds on the January and July 2025 folds (153,706 + 190,713 = 344,419
departures, each predicted by models that did not train on its month); lower is better.

## From the baseline to the final

![From the airport-mean baseline to the final model, one bar per stage](../writeup/figures/fig_progress.png)
*The three largest cuts (blue) come from understanding the label; open data (green) gives the largest later gain.*

Gains are differences of unrounded scores, so a gain can differ by 0.01 from the difference of the two rounded values.

| stage | CV before → after | gain |
|---|---|---:|
| airport mean | 660.89 | |
| residual on the NM off-block clock (`MVT − AOBT_3`) | 660.89 → 511.51 | 149.38 |
| mixture: P(BLOCK copied from SCHED) and copy branch | 511.51 → 437.17 | 74.33 |
| LIRF recording-regime post-processing | 437.17 → 351.80 | 85.37 |
| target clip, deeper trees, stand and runway history | 351.80 → 330.12 | 21.68 |
| CatBoost and DAE members, cross-fitted NNLS blend | 330.12 → 320.47 | 9.65 |
| clock rules: date shift, gate hold, LIRF copy | 320.47 → 311.46 | 9.01 |
| OPDI own-flight taxi events in the members | 311.46 → 309.86 | 1.60 |
| plan-window rules, LIRF delayed-row gate | 309.86 → 306.96 | 2.90 |
| ADS-B ground-track residual stacker | 306.96 → 294.51 | 12.45 |
| MLP and LOBT members, second stacker, LIRF rules | 294.51 → 288.33 | 6.18 |
| record rules, EUROCONTROL calibrations, a second ADS-B feed | 288.33 → 286.29 | 2.04 |
| member-disagreement stacker inputs | 286.29 → 285.44 | 0.84 |
| departure-stream context features | 285.44 → 284.32 | 1.12 |
| month-complete stackers | 284.32 → 283.64 | 0.68 |
| LIRF plan-row classifier left out (does not hold on held-out months) | 283.64 → 284.11 | −0.47 |
| stream-transformer inputs to the stackers (two members' columns) | 284.11 → 282.69 | 1.42 |
| neighbour-label head of the stream transformer | 282.69 → 282.48 | 0.21 |
| monthly runway calibration left out (tested; the final chain uses one published aggregate of the scored months, not two) | 282.48 → 282.53 | −0.05 |
| four chain post-steps (LIRF copy re-mix, runway-neighbour level, far-LIRF record mix, arrival-twin window rule) | 282.53 → 279.27 | 3.26 |
| second ADS-B feed left out ([data sources](data-sources.md)) | 279.27 → 279.61 | −0.34 |

The final file is the last row's chain plus the plan-window post-step, which the cross-validated score leaves out: it moves 84
of the 670,790 test rows and changed the score by only −0.04 on the earlier bundle chain.

The three largest steps concern the *label*: reading take-off minus NM's off-block clock (149), modelling the schedule copies as
a mixture (74) and the Rome recording regimes (85). The ADS-B residual stacker, with its two refinements (events from positions,
day-block cross-fit), is the largest gain from open data (12.45). In the chain table below the stackers are worth only 3.48,
because there five of the regressors already read ADS-B features.

## The chain, step by step

Each step against the one before, on the reference chain (CV 285.44 at the end). Intervals are 95 %, from a paired bootstrap over
whole days.

![Each tail step](../writeup/figures/fig_tail.png)
*Every interval lies below zero or ends at it (monthly runway step); the ADS-B stackers are worth more than the other steps together.*

| step | CV | trim | clean | Jan | Jul | Δ RMSE [95 % CI] |
|---|---:|---:|---:|---:|---:|---|
| rules chain on the 11-member blend | 292.33 | 230.30 | 217.47 | 326.80 | 261.27 | |
| + ADS-B residual stackers (average of 4) | 288.85 | 225.85 | 212.72 | 323.25 | 257.81 | −3.48 [−4.61, −2.69] |
| + LIRF no-plan copy classifier | 287.70 | 224.38 | 212.72 | 323.25 | 255.47 | −1.15 [−2.62, −0.10] |
| + LIRF near-schedule rule | 287.25 | 223.80 | 212.77 | 323.32 | 254.49 | −0.45 [−0.98, −0.07] |
| + LIRF plan-row copy classifier | 286.88 | 223.32 | 212.26 | 322.98 | 254.07 | −0.37 [−0.69, −0.11] |
| + daily delay calibration | 285.75 | 221.83 | 210.77 | 322.49 | 252.27 | −1.13 [−1.95, −0.53] |
| + duplicated-record rule | 285.47 | 221.47 | 210.59 | 322.12 | 252.08 | −0.28 [−0.60, −0.07] |
| + monthly runway calibration | 285.44 | 221.44 | 210.54 | 322.11 | 252.03 | −0.03 [−0.06, 0.00] |

The final chain stops at the duplicated-record rule. We tested the monthly runway calibration and left it out, at +0.028 RMSE
[+0.010, +0.046] on twelve months ([model](model.md#calibration-to-published-aggregates)).

## The final chain's last additions

Judged on twelve months weighted like the test months ([validation](validation.md#twelve-months-weighted-like-the-test-set)), with a
paired day-block bootstrap and 95 % intervals, on earlier versions of the chain that still read the second ADS-B feed; the progress table above gives the Jan/Jul CV.

| change | RMSE change [95 % CI] | months better |
|---|---|---|
| month-complete stackers | −0.71 [−0.86, −0.56] | 11 of 12 (February loses, a snow month) |
| stream-transformer inputs and month-complete stackers together, against the context chain | −1.67 [−1.88, −1.45] | |
| neighbour-label head, against the bundle without it | −0.13 [−0.20, −0.06] | 11 of 12 |
| the four post-steps together (on the version that still had the monthly runway step) | −0.98 [−1.34, −0.62] | 12 of 12 |
| second ADS-B feed left out, against the shipped chain's twin with it | +0.30 [+0.21, +0.38] | 1 of 12 |

Half of the post-steps' gain is one far LIRF row; without the 20 rows that carry most of it the four steps still gain −0.35 ([model](model.md#four-post-steps-on-the-chain-output)).

## What did not help, and why

Methods we measured and did not keep. *RMSE change* is the change in RMSE (seconds) of the chain or member the method was
added to; **positive is worse**. *Measured on* names the sample, since the comparisons use different ones: *Jan/Jul* is the two
cross-validation folds, a *rehearsal* a held-out pair of 2025 months ([validation](validation.md)), and a *screening run* a quick
test on a small sample of rows that decides whether a method earns a full measurement. A *dark row* is a departure with no ADS-B
track. The *gain bar*, the minimum improvement a method must show to enter the model, was set before the test: −0.8 RMSE for the
stream-network variants.

| method | RMSE change | measured on | why it did not help |
|---|---|---|---|
| LIRF plan-row copy classifier | −0.37 on Jan/Jul; +0.27 in June 2025; −0.09 [−0.23, +0.04] over six rehearsal pairs | Jan/Jul, June 2025, six pairs | gains on the folds it was tuned on, loses in June with an interval above zero, neutral over the pairs |
| Monthly runway taxi-out calibration | −0.03 on Jan/Jul; −0.028 [−0.046, −0.010] over twelve months | Jan/Jul, twelve months | a small gain that needs a second published aggregate of the scored months; left out so the model uses one. Without a guard on the invalid-flight share it loses 150 s² of MSE at LTFM in the February 2025 snow month |
| Dropping the five most year-drifting inputs of each stacker | +0.59 | February and June 2025 rehearsal | worse in both months (February +1.28, June +0.18) against the same stackers with the inputs kept [+0.28, +1.07] |
| Open tabular foundation models (TabPFN, TabDPT) as members | +20 to +28 at LFPG, LIRF, LTFM | 2,000-row screening run | worse than the tuned gradient-boosted members, and their errors correlate 0.87–0.91 with the blend, so they add little diversity |
| Take-off order as a constraint (first-in-first-out, learning-to-rank, joint sequence prediction) | +10.9 to +149.7 | rehearsal | every way of projecting the predictions onto an order raised the error; real take-offs overtake each other at about the rate the labels imply, so the constraint is wrong |
| Sequence stacker over the departure stream | +0.14 | rehearsal | worse than the existing stackers (z = +2.96) |
| Neural network stacker | about 0 (+2.5 MSE) | rehearsal | no gain over the existing stackers |
| MLP stacker with ADS-B dropout | +0.31 | four test-month rehearsals, 0 of 4 months better | loses even where the track is hidden (dark rows): the stackers already sit below the no-ADS-B level there, so the members' gain on those rows adds nothing and their cost on ordinary rows passes through |
| Relation bias in the stream transformer's attention | +0.32 | rehearsal (3 seeds) | worse on the January fold (196.18 against 194.54) and equal on July; it failed the rule that a method must gain at least 0.8 RMSE to be kept |
| LightGBM on the rows flagged as large-error | +0.01 to +0.12 | twelve-month leave-one-month-out | hurts, most on snow rows; the 741 flagged rows hold 48.5 % of the squared error and no mechanism we tested recovers more than 0.2 of it |
| Classifiers for the far LIRF cluster | +0.15 to +19 | rehearsal | the record's marker for the cluster is already priced into the mixture |
| Airport-hour traffic-state features | −0.08 to −0.17 at best | rehearsal | fragile: 91 % of one variant's gain is a single LTFM snow day and two thirds of another's is 20 rows |
| GRU or convolutional trunk in place of attention in the stream network, added beside it | −0.04 [−0.13, +0.05] (GRU), −0.04 (convolutional) | rehearsal estimate for the four test months | interval includes zero and half the gain is two rows (GRU) or one LIRF airport-day (convolutional); their residuals correlate 0.97 with the attention network's, so the trunk is not the limit |
| Copy classifier over the airport-day sequence | none (not scored: LIRF AUC 0.853 against 0.859 for the row classifier) | LIRF, 2025 | no better than the row classifier at finding the copies; the hardest rows carry no clock a sequence model could anchor on |

## By airport

![Share of squared error and RMSE by airport](../writeup/figures/fig_airports.png)
*Paris and Rome hold two thirds of the squared error; on rows up to 2 h every airport but Rome sits at 80–260 s.*

The reference chain (CV 285.44) on the two folds. *Share of SSE* is the airport's share of the squared error of both folds together. RMSE ≤ 2 h
excludes the rows with labels above two hours.

| airport | Jan rows | Jan RMSE | Jan ≤ 2 h | Jul rows | Jul RMSE | Jul ≤ 2 h | ADS-B match Jan / Jul | share of SSE |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| EDDF | 15,496 | 121.4 | 121.4 | 21,334 | 128.3 | 128.3 | 0.91 / 0.94 | 2.1 % |
| EDDM | 11,266 | 79.3 | 79.3 | 15,825 | 143.5 | 143.5 | 0.97 / 0.37 | 1.4 % |
| EGLL | 19,420 | 193.7 | 193.1 | 20,790 | 271.4 | 244.7 | 0.68 / 0.29 | 8.1 % |
| EHAM | 18,460 | 141.1 | 115.9 | 22,489 | 104.3 | 98.9 | 0.99 / 0.98 | 2.2 % |
| LEBL | 12,234 | 90.5 | 90.5 | 16,752 | 225.5 | 225.5 | 0.98 / 0.94 | 3.4 % |
| LEMD | 16,510 | 154.0 | 154.0 | 18,818 | 169.5 | 169.5 | 0.00 / 0.30 | 3.3 % |
| LFPG | 17,900 | 795.9 | 244.7 | 21,712 | 236.3 | 232.4 | 0.02 / 0.11 | 44.7 % |
| LIRF | 11,317 | 372.8 | 293.3 | 15,211 | 582.0 | 505.3 | 0.01 / 0.94 | 24.0 % |
| LSZH | 9,735 | 136.5 | 113.7 | 12,611 | 130.3 | 106.4 | 0.97 / 0.96 | 1.4 % |
| LTFM | 21,368 | 213.0 | 208.7 | 25,171 | 259.2 | 259.2 | 0.00 / 0.00 | 9.5 % |

- LFPG's January share is almost entirely the two outlier records of [the label](task-and-data.md#the-label-is-two-processes).
- LIRF's July error comes from its delayed departures with no flight plan.
- On rows up to 2 h the airports range from 79 s (EDDM, January) to 505 s (LIRF, July).

### With and without an ADS-B match

| rows | count | after the rules chain: RMSE / ≤ 2 h | after the stackers | reference chain |
|---|---:|---|---|---|
| matched | 182,684 | 214.6 / 196.7 | 208.9 / 190.4 | 203.0 / 188.3 |
| not matched | 161,735 | 360.5 / 238.8 | 358.3 / 235.4 | 356.3 / 233.1 |

The stackers help both groups. On rows up to 2 h the error is about 45 s lower where the aircraft's track is known; part of that
gap is the airport mix, since LTFM and LFPG are mostly unmatched.

## By month

![Leave-one-month-out error by month](../writeup/figures/fig_months.png)
*July is the hardest month and October the easiest. Missed schedule copies drive the summer rise (June to September); rows over 2 h
are heavy but erratic (January as large as July, November above June).*

From the twelve-month leave-one-month-out run of the base mixture (RMSE 316.42 overall). MSE shares are in thousands of s².

| month | RMSE | RMSE ≤ 2 h | MSE from ordinary rows | from copies | from 1–2 h | from > 2 h |
|---|---:|---:|---:|---:|---:|---:|
| Jan | 388.6 | 209.3 | 36.9k | 6.3k | 3.4k | 104.4k |
| Feb | 287.6 | 245.8 | 32.5k | 20.3k | 22.3k | 7.6k |
| Mar | 232.5 | 179.5 | 26.0k | 8.3k | 2.2k | 17.7k |
| Apr | 229.3 | 184.3 | 27.8k | 22.7k | 1.8k | 0.3k |
| May | 289.5 | 182.4 | 28.0k | 7.7k | 1.6k | 46.6k |
| Jun | 355.3 | 212.4 | 32.8k | 30.7k | 3.8k | 58.9k |
| Jul | 461.7 | 253.4 | 47.0k | 56.6k | 7.1k | 102.5k |
| Aug | 333.3 | 219.0 | 38.5k | 66.4k | 2.1k | 4.2k |
| Sep | 333.9 | 207.7 | 34.4k | 41.5k | 2.9k | 32.7k |
| Oct | 210.9 | 198.3 | 30.8k | 10.9k | 2.2k | 0.6k |
| Nov | 323.5 | 185.1 | 27.4k | 13.5k | 2.9k | 60.9k |
| Dec | 236.5 | 194.7 | 30.6k | 12.1k | 3.0k | 10.2k |

## Final chain by month

The final chain's validation run, scored on the twelve held-out months of 2025: January and July out of fold, the other ten
from the rehearsal that leaves each month out ([validation](validation.md)). It runs the shipped chain's steps:
the bundle chain, the LIRF copy-direction re-mix, the runway-neighbour level corrector, the far-LIRF step and the LIRF
twin-window step, plus the LIRF plan-row classifier, which the final chain dropped ([below](#the-four-post-steps-by-month)). It stops before the witness plan-window step and the seed bag. The shipped file is the
final-phase rebuild of this chain, and the two differ by draw noise (rms 33 s on the test months with the release draws,
[reproduction](reproduction.md#what-has-been-replayed)), so these tables describe the chain's design, not the shipped bytes. The same draw noise is why their January and July differ
slightly from the [README](../README.md)'s, which are the rebuild's own fold scores (its recipe in `code/recipes/`).

[`writeup/final_chain_tables.py`](../writeup/final_chain_tables.py) scores the stored predictions read-only; it needs the full
run's stored predictions, which this repository does not hold. The tables are also in [`writeup/data/`](../writeup/data/)
(`final_chain_month.csv`, `final_chain_airport.csv`, `final_chain_month_airport.csv`). *RMSE ≤ 2 h* leaves out labels above
7,200 s; *bias* is the mean of prediction minus label on those rows (positive = predicted too long). Months are UTC take-off
months, so a count can differ by one row from the other tables.

| month of 2025 | rows | RMSE | RMSE ≤ 2 h | bias ≤ 2 h |
|---|---:|---:|---:|---:|
| Jan | 153,706 | 315.2 | 175.2 | +1.3 |
| Feb | 143,731 | 200.7 | 186.4 | +0.6 |
| Mar | 164,449 | 157.0 | 154.4 | -1.4 |
| Apr | 175,288 | 166.9 | 164.9 | -5.4 |
| May | 185,202 | 267.3 | 160.7 | -1.0 |
| Jun | 183,142 | 264.4 | 188.9 | +0.3 |
| Jul | 190,713 | 246.8 | 229.9 | +0.8 |
| Aug | 191,181 | 211.1 | 199.0 | +0.7 |
| Sep | 183,950 | 196.1 | 184.9 | +0.3 |
| Oct | 185,674 | 181.6 | 177.4 | -0.6 |
| Nov | 162,332 | 159.2 | 156.4 | +1.8 |
| Dec | 165,677 | 166.9 | 163.6 | +1.0 |
| all twelve, pooled | 2,085,045 | 216.9 | 180.5 | -0.2 |

By airport, same rows:

| airport | rows | RMSE | RMSE ≤ 2 h | bias ≤ 2 h | share of squared error |
|---|---:|---:|---:|---:|---:|
| EDDF | 230,141 | 97.8 | 97.8 | -0.04 | 2.2 % |
| EDDM | 167,334 | 87.4 | 87.4 | -0.18 | 1.3 % |
| EGLL | 239,546 | 199.8 | 194.0 | -0.14 | 9.8 % |
| EHAM | 247,951 | 95.9 | 92.8 | -0.79 | 2.3 % |
| LEBL | 179,705 | 144.1 | 144.1 | +0.11 | 3.8 % |
| LEMD | 212,241 | 133.5 | 133.5 | +0.05 | 3.9 % |
| LFPG | 239,552 | 297.9 | 211.9 | -0.57 | 21.7 % |
| LIRF | 160,704 | 446.9 | 364.8 | +1.23 | 32.7 % |
| LSZH | 134,907 | 259.6 | 90.7 | +0.07 | 9.3 % |
| LTFM | 272,964 | 216.6 | 214.4 | -0.71 | 13.1 % |

The January/July table ([by airport](#by-airport)) and the rehearsal table ([rehearsal](#rehearsal-on-held-out-feb-and-jun-2025))
are of the reference chain, whose every step is measured on its own; the two tables above are of the final chain's validation
run. The next table shows what the four post-steps do to that run's earlier stage, the bundle chain.

### The four post-steps by month

RMSE in seconds of the final model's *[validation twin](glossary.md)*, the final chain as it stood when validated, run in the same held-out
phases: first the bundle chain (the chain before the four post-steps, [names](reproduction.md#names-used-in-this-repository)),
then the same chain with the four post-steps. Each month is scored in the one of the six held-out phases that leaves it out. Both
columns still carry the LIRF plan-row step that the final chain drops.

| month of 2025 | rows | bundle chain | with the four post-steps | change |
|---|---:|---:|---:|---:|
| Jan | 153,706 | 320.5 | 315.2 | −5.2 |
| Feb | 143,731 | 205.5 | 200.7 | −4.8 |
| Mar | 164,449 | 159.1 | 157.0 | −2.1 |
| Apr | 175,288 | 174.3 | 166.9 | −7.4 |
| May | 185,202 | 268.8 | 267.3 | −1.4 |
| Jun | 183,142 | 272.3 | 264.4 | −8.0 |
| Jul | 190,713 | 248.0 | 246.8 | −1.2 |
| Aug | 191,181 | 212.7 | 211.1 | −1.6 |
| Sep | 183,950 | 234.3 | 196.1 | −38.2 |
| Oct | 185,674 | 182.5 | 181.6 | −1.0 |
| Nov | 162,332 | 163.7 | 159.2 | −4.5 |
| Dec | 165,677 | 168.2 | 166.9 | −1.3 |
| all twelve, pooled | 2,085,045 | 223.3 | 216.9 | −6.4 |

Every month improves. September is the outlier: 53 % of the pooled gain is one LIRF airport-day (28 September 2025), and
about half of the squared error removed there is one row, a "+1 day" record of 87,543 s. Outside September the change is −1.0
to −8.0 per month. The pooled change differs from the weighted gain of the previous section, which counts only the four months
the test covers.

## Rehearsal on held-out Feb and Jun 2025

The reference chain: **247.48** (Feb 208.21, Jun 274.39); trim 201.81, clean 191.54. The airport-mean baseline scores 558.74 on the
same rows (Feb 606.14, Jun 518.51). By airport:

![Rehearsal on held-out February and June 2025, by airport](../writeup/figures/fig_rehearsal.png)
*Rows up to 2 h by airport on the two held-out months; Istanbul in February is the snow month.*

| airport | Feb rows | Feb RMSE | Feb ≤ 2 h | Feb mean error | Jun rows | Jun RMSE | Jun ≤ 2 h | Jun mean error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EDDF | 15,099 | 93.8 | 93.8 | −1.3 | 20,706 | 106.6 | 106.6 | −2.3 |
| EDDM | 10,381 | 71.1 | 71.1 | +1.7 | 15,661 | 100.0 | 100.0 | −0.5 |
| EGLL | 17,956 | 168.3 | 162.7 | −9.9 | 20,063 | 215.6 | 211.5 | −11.4 |
| EHAM | 17,852 | 85.3 | 85.3 | +4.2 | 21,257 | 85.4 | 85.4 | +2.0 |
| LEBL | 12,169 | 85.3 | 85.3 | +3.3 | 16,315 | 132.3 | 132.3 | +6.2 |
| LEMD | 15,390 | 142.3 | 142.3 | −4.9 | 17,992 | 120.6 | 120.6 | −7.7 |
| LFPG | 16,806 | 193.1 | 193.0 | −8.3 | 20,887 | 234.2 | 228.2 | −5.5 |
| LIRF | 10,703 | 357.2 | 290.2 | +1.4 | 14,696 | 804.6 | 421.9 | −0.3 |
| LSZH | 9,048 | 204.0 | 85.4 | +9.2 | 12,083 | 96.0 | 96.0 | −0.3 |
| LTFM | 18,327 | 379.4 | 363.5 | −22.5 | 23,482 | 205.5 | 205.5 | −13.5 |

LTFM February is the snow month; LIRF June carries the long no-plan rows; LSZH February has a few long taxis. The mean error on
rows up to 2 h is slightly negative (−4.0 s overall), largest at LTFM, EGLL, LEMD and LFPG.

## Calibration

![Mean error by prediction and by true value](../writeup/figures/fig_calibration.png)
*Right on average up to 30 min of prediction; long true taxis are predicted too short, as a mean must when their cause is invisible.*

- Grouped by prediction, the mean error is small up to 30 min: −6.6 s for predictions under 5 min, +1.8 s at 5–10 min, −2.5, −4.1
  and −6.9 s at 10–15, 15–20 and 20–30 min; −31 s at 30–60 min and +99 s for the 534 rows predicted at 1–2 h.
- Grouped by true value, it shows the usual regression to the mean: +418 s for the 79 rows with `y ≤ 0`, −251 s at 30–60 min,
  −992 s at 1–2 h and −5,571 s above 2 h. A conditional-mean model does this when the inputs cannot see why a taxi is long.
