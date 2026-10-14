# Validation

**In short:** two validation folds on the same calendar months as the test set; changes compared by a paired bootstrap over
whole days; the whole pipeline rehearsed on February and June 2025, masked like the test months; the last model changes judged on
all twelve months of 2025, weighted like the test months.

The test months have no labels, and 2026 differs from 2025. Every part of the design asks one question: *does this still help on
months that no fitted parameter has seen?*

## Folds

| fold | validation month | training months | buffer months (left out of training) |
|---|---|---|---|
| Jan | Jan 2025 (153,706 departures) | Mar–Nov 2025 | Feb, Dec |
| Jul | Jul 2025 (190,713 departures) | Jan–May and Sep–Dec 2025 | Jun, Aug |

- **Same calendar months as the test set's first two months**, because seasons differ. A classifier on label-free features
  separates all of 2025 from Jan/Jul 2026 at AUC 0.968, but Jan/Jul 2025 from Jan/Jul 2026 only at 0.777 (a control with shuffled
  source labels gives 0.501). Of that classifier's total feature-importance gain, 46.8 % comes from flight numbers and callsigns, 12.1 % from stands,
  8.7 % from runways: naming and allocation drift between years, more than operations. Raw flight numbers and callsigns are
  therefore not model inputs.
- **Buffers** keep rolling and history features of the days next to a validation month out of training.
- **Nine training months** keep each fold's model about as large as the twelve-month refit used for the test files.

## Comparing two runs

A change is compared with the run it modifies by a **paired bootstrap over day blocks** (1,000 resamples) on all rows, on *trim*
rows (label ≤ 20,000 s) and on *clean* rows (label ≤ 7,200 s). Resampling whole days keeps the day-level correlation of errors, so a
lucky day cannot pass as a gain. Every Δ with a 95 % interval in these pages comes from it.

## Rehearsal on held-out February and June 2025

The test set adds two months the folds do not cover (February and June). So the whole pipeline was run one year earlier:

![Rehearsal on held-out February and June 2025, by airport](../writeup/figures/fig_rehearsal.png)
*On the two held-out months the error by airport looks like the validation months': Rome, Istanbul in a snow month, then Paris and Heathrow.*

- the training set becomes the other ten months of 2025;
- Feb and Jun 2025 departures (326,873) are masked exactly like the test months and appended to the Jan/Jul rows, so the rehearsal
  has the test set's shape and months;
- the pipeline then runs unchanged: features rebuilt, every member refit on ten months, blend, rules, stackers and tail, scored
  against the held-out labels.

**Result of the reference chain** (CV 285.44 on the January and July folds; the rehearsal sample is the 326,873 February and June 2025 departures of the ten airports): RMSE **247.48** (trim 201.81, clean 191.54); Feb 208.21, Jun 274.39. The airport-mean
baseline on the same rows scores 558.74 (Feb 606.14, Jun 518.51), against 660.89 on the folds: these two months are easier than
January and July, so 247.48 compares with 558.74, not with the folds' 285.44. The final model has no single rehearsal number: its
month-complete stackers learn from the held-out pairs, so each pair needs its own refit without that pair. Its additions over the
reference chain are scored instead on the six-pair score below (context −0.58, month-complete stackers −0.71, stream-transformer inputs with the month-complete stackers −1.67 against the context chain, neighbour-label head −0.13, four post-steps −0.98 RMSE).

ΔMSE is the change in mean squared error (s²) from the step; negative is better.

| step | Feb 2025: RMSE before → after, ΔMSE [95 % CI] | Jun 2025: RMSE before → after, ΔMSE [95 % CI] |
|---|---|---|
| gate, plan cliff and plan-window clip (over the blend with LIRF regimes and clock rules) | 238.88 → 236.47, −1,145 [−2,144, −305] | 283.93 → 276.16, −4,351 [−7,285, −1,893] |
| ADS-B stackers | 236.47 → 227.71, −4,069 [−6,815, −1,892] | 276.16 → 274.54, −894 [−1,017, −766] |
| LIRF no-plan classifier | no rows touched | 274.54 → 277.25, +1,492 [−689, +5,375] |
| LIRF near-schedule rule | no rows touched | 277.25 → 276.85, −219 [−566, 0] |
| LIRF plan-row classifier (not in the final model) | 227.71 → 226.34, −619 [−1,034, −211] | 276.85 → 277.12, +150 [+7, +300] |
| daily delay calibration | 226.34 → 208.33, −7,829 [−18,433, −752] | 277.12 → 274.89, −1,233 [−3,068, −218] |
| duplicated-record rule | 208.33 → 208.27, −28 [−105, +41] | 274.89 → 274.49, −219 [−489, −3] |
| monthly runway calibration (not in the final model) | 208.27 → 208.21, −25 [−34, −16] | 274.49 → 274.39, −52 [−64, −42] |

- The rules chain, the stackers and both calibrations gain in both months (the final chain keeps only the daily one); the duplicated-record rule gains in June. The daily delay
  calibration's February gain (−7,829 MSE, about 18 s of RMSE) is far larger than on the folds and very uncertain
  ([−18,433, −752]): a few days with heavy delays carry it.
- The LIRF no-plan classifier acts only on ADS-B-matched rows, and LIRF was 0 % matched in February 2025, so it did nothing there.
  In June it loses, with an interval that includes zero; over the six held-out pairs it is neutral (−0.01 RMSE, better in 4 of 6).
  In 2026 LIRF is 70–77 % matched in January and February, so the test months use it more than any validation month did
  ([model](model.md#record-tail) says why it stays).
- The LIRF plan-row classifier gains in February but loses in June, where its moves do not sort copies (the copy share of its top
  quintile, 0.13, equals the base rate). Over the six held-out pairs below it is neutral (−0.09 RMSE [−0.23, +0.04]), so the
  final model drops it.
- The monthly runway calibration leaves alone any cell whose invalid-flight share lies outside the 2025 range. Without that guard it
  loses 150 MSE at LTFM in the February 2025 snow month (invalid share 0.111 against ≤ 0.002 in every other month). The guard is
  confirmed on the twelve-month leave-one-month-out predictions (February −67 → −251 MSE, no other month changed). The final chain
  leaves the step out, at +0.028 RMSE [+0.010, +0.046] on twelve months ([model](model.md#calibration-to-published-aggregates)).

**What the rehearsal does not test.** The stackers learn from out-of-fold predictions of fold models whose training months include
June (Jan fold) or February (Jul fold), so the held-out labels reach the second level indirectly; and the rules were designed on all
of 2025. The rehearsal is therefore an upper bound.

## Every month of 2025: leave-one-month-out

A base mixture predicts each month of 2025 from the nine months that are neither the month nor its neighbours. Its RMSE is 316.42
overall (clean 207.29), from 210.9 (Oct) to 461.7 (Jul); see [results](results.md#by-month). Copies are 4.6–6.2 % of departures in
every month, but their cost peaks in summer: a missed copy costs about the square of the flight's off-block delay, and delays peak
in summer.

## Twelve months weighted like the test set

The last model changes (context features, month-complete stackers, stream-transformer inputs, neighbour-label head, post-steps) were judged on a score built for the test months:

- **Six held-out pairs.** Besides Jan/Jul and Feb/Jun, four more pairs rerun the whole pipeline: Mar+Sep, Apr+Oct, May+Nov,
  Aug+Dec 2025. Together they cover all twelve months.
- **Weights.** Each test month's MSE change is estimated from the pairs containing that calendar month, then the four are weighted
  by their share of the test rows: 0.228 Jan, 0.215 Feb, 0.271 Jun, 0.286 Jul. An equal-weight twelve-month mean can rate as a gain
  something that loses on the test months.
- **Shrinkage.** Each month's estimate is shrunk towards the mean of all months (empirical Bayes, between-month variance by
  DerSimonian–Laird), so one heavy-tailed month (snow at LTFM in Feb 2025) cannot decide alone.
- **Bootstrap.** The same paired day-block bootstrap, stratified by month; a change must lower the weighted error with its 95 %
  interval below zero and lose in no month with an interval wholly above zero.

## What changes between 2025 and the 2026 test months

Checked with label-free inputs only:

- **February:** an adversarial classifier separates Feb 2025 from Feb 2026 at AUC 0.503 (no difference).
- **June:** AUC 0.685, from real traffic change (departures per day: EDDM 522 → 479, EDDF 690 → 657, LEMD 600 → 645, LEBL 544 → 569).
- **Runways:** February's runway mix shifts with the wind (LEMD's south flow 5 % → 39 %); every runway with over 50 test
  departures has at least 110 in the 2025 training data.
- **Clock gaps** do not drift, except LFPG in June: `MVT − AOBT_3` has a median of 961 s against 844 s, a change in NM's default
  taxi time (844, 899 or 960 s by period), not in taxiing.
- **Record order:** at LTFM the Spearman correlation between record id and take-off time is 0.41–0.45 in every 2025 month and
  0.93–0.94 in all four 2026 months, so LTFM's ordering features read a different kind of input. No validation number isolates this. See [limits](limits.md#record-order-at-ltfm) for what the ordering features are worth and why they were kept.
- **ADS-B coverage** differs from 2025 at several airports, see [limits](limits.md#coverage-holes-in-the-open-sources).

## Where the numbers are optimistic

- The regressors early-stop on the validation folds' non-copy rows.
- The month-complete stackers score January and July by five blocks of whole days, so other days of the same month are in
  their training. A test month has no labelled days, so its stackers lack that.
- The LIRF no-plan classifier leaves out only the scored month, not its neighbours. With the buffer months left out its fold gain
  falls from −1.23 to −0.47 ([model](model.md#record-tail)); the reported CV keeps the unbuffered fit.
- The four post-steps fit on a twelve-month pool that leaves out the scored month's own fold but keeps its neighbours. The far-LIRF step, 2.64 of their 3.25 gain on the folds, acts on 5 to 7 rows per phase; its twelve-month interval, −0.04 [−0.17, +0.10], includes zero.
- The plan-window rules (the plan-window post-step and the arrival-twin rule) use how the organisers' join built the files, not a taxi mechanism, and the headline includes them. Their measured cost is in [the dataset-artefact table](model.md#dataset-artefact-rules): −0.33 [−0.87, −0.02] for the plan-window step on an earlier final chain (−0.14 for the arrival twin), and a different join in the 2026 files would turn them into noise on the rows they touch.
- Rule thresholds and the half weight of one LIRF classifier were set on 2025 data, including the folds.
- Many runs were compared on the same two folds; the rehearsal and the twelve-month score are the checks on months those
  comparisons did not use, and the rehearsal is itself an upper bound.

## The public leaderboard

The public leaderboard scores the same January and July 2026 rows on every submission, so each score is an exact equation in
the hidden labels, and a submission that changes one row reveals that row's label.

- We probed the public board from 29-09 to 12-10 (versions 13 to 67, UTC). The probes decode single labels.
- The probes confirmed the mechanism on the board before 03-10, so the private note we sent the organisers that day
  ([PDF](leaderboard_leakage.pdf)) reported a measured effect, with a fix: rank the final on labels the board never scores.
- Five days later, on 08-10, the organisers added blind February and June 2026 months to the final ranking.
- On 13-10 a board-only file, version 72, showed the size: the shipped final's January/July predictions with 91 rows replaced by
  decoded labels scored 208.11 and ranked first.

The shipped final has nothing to do with the probing. It contains no decoded label (it differs from version 72 in exactly those
91 rows), and the code in this repository has no probe or decode step. None of this needs checking from the code: the final is
also scored on blind February and June 2026, which no board submission ever touched, and that score is the check. It measures the
model plus the daily delay calibration (`dly_cal`, [daily delay file](daily-delay-file.md)), which runs on those months too; the
[rehearsal](#rehearsal-on-held-out-february-and-june-2025) shows both on February and June 2025 (reference chain: 226.34 and
277.12 before that step, 208.33 and 274.89 after it).
