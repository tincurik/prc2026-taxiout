# The daily delay file (`dly_cal`)

> **For reviewers: this step can look like leakage.** It reads a published daily aggregate of the same off-block records the
> labels come from. The organisers explicitly allowed EUROCONTROL-published data for it (13-10-2026), so please confirm it with the
> challenge's human coordinators rather than judge it from the code. `TAIL_SKIP=dlyh` switches the step off.

`dly_cal` is a record-tail step. It reads one published EUROCONTROL number per airport and day, the daily all-cause
pre-departure delay `DLY_ALL_PRE_2`, and nothing else. EUROCONTROL computes that number from the same off-block records as the
challenge labels. So the step constrains each scored day's predictions with a published aggregate of that day's labels. That is
why it helps. It reads no flight's own label: the number is one sum over all of an airport's departures that day.

**Permission.** Asked whether this file may be used on the test days, the organisers answered on 13-10-2026 that data EUROCONTROL
publishes may be used. Earlier answers point the same way: the published monthly taxi-out for the test months, an aggregate of the
same labels at a coarser grain under the same EUROCONTROL terms, was allowed (27-09, again 08-10), and EUROCONTROL's daily airport
files may be treated as open data for the challenge if their licence allows (17-09). The answers are paraphrased and dated in
[data sources](data-sources.md#organiser-statements-on-the-inputs).

**Why the step belongs in the model.** The organisers named the deployment: a post-operational model (19-09), run on days that
are over. EUROCONTROL's Performance Review Unit publishes this total for every such day, on ansperformance.eu, the portal that
also hosts this challenge; all 1,200 scored airport-days have one. A model built for post-operational use should use what
post-operational use has; leaving out a published input would validate a harder, tactical case that is not the one being scored.
A tactical model could not use it. The shipped chain runs the step. `TAIL_SKIP=dlyh` turns it off, at a cost of **+0.72 RMSE** on
twelve months, measured on an earlier version of the chain ([what it is worth](#what-it-is-worth)).

## What `DLY_ALL_PRE_2` is

For each airport and day, the all-cause pre-departure delay in minutes, `Σ max(BLOCK − SCHED, 0)` over the day's departures. It
is a daily airport total, not a value for any flight. On 2025 it matches the challenge labels with correlation 0.9997 and median
ratio 1.000, because both come from the same airport-reported off-block records.

| | |
|---|---|
| source | EUROCONTROL Performance Review Unit, `ansperformance.eu` |
| files | `https://www.eurocontrol.int/performance/data/download/csv/all_pre_departure_delays_{2024,2025,2026}.csv` |
| retrieved | 28-09-2026 |
| coverage | daily per airport; the 2026 file reaches August 2026, so it covers all four test months |
| terms | "copied in whole or in part provided EUROCONTROL is mentioned as source, not for commercial purposes, not modified without permission". No named open licence. Cited as EUROCONTROL Performance Review Unit ([credits](credits.md)) |
| redistribution | none: [`code/scripts/fetch_external.py`](../code/scripts/fetch_external.py) downloads it at run time |

## How `dly_cal` uses it

For each airport-day, the step solves for one multiplicative factor on the day's ordinary rows (predictions below 3,600 s) that
makes the implied total, `Σ max(MVT − SCHED − prediction, 0)`, equal the published one. It applies 0.6 of that change. Rows
predicted above 4,000 s whose `MVT − SCHED` exceeds the prediction by more than 1,200 s move by `−b · clip(z, ±2) · prediction`,
with `z` the airport's robust z-score of the day's gap and `b` fitted on 2025 out-of-fold rows. Days without a published value
stay unchanged. Every setting was chosen on 2025, and the same code runs for all four scored months
([`code/overnight/dly_cal.py`](../code/overnight/dly_cal.py)).

## What it is worth

We rebuilt an earlier version of the final chain without the step, refitted the later post-steps on the changed predictions, and
scored both chains on twelve months (the final January and July and rehearsal months 2 to 6, out of fold) as a paired score with
a day-block bootstrap interval. That version still read a second ADS-B feed and ended in the monthly runway calibration, both
dropped later; it scored 279.19 against the shipped chain's 279.61. The third row is the shipped chain's own tail step.

| measurement | with `dly_cal` | without | cost of switching it off |
|---|---:|---:|---:|
| twelve months, paired score, earlier chain (worse in 12 of 12 months) | | | **+0.72 RMSE [+0.52, +0.93]** |
| January + July CV, earlier chain, with the post-steps | 279.19 | 280.60 | +1.41 |
| January + July CV, shipped chain, the tail step alone (nothing after it refitted) | 283.13 | 284.13 | +0.99 |
| January + July CV, reference chain ([results](results.md)) | 285.75 | 286.88 | +1.13 |
| held-out February 2025 rehearsal, stage 3b chain | | | +18 s |

Each row is one measurement on one chain; the first is the headline. The cost spreads over the months (+0.41 to +1.98 RMSE in
eleven of them, +9.77 in one rehearsal month) and shows at nine of ten airports.

The switch is `TAIL_SKIP=dlyh` in [`code/overnight/tail_chain.sh`](../code/overnight/tail_chain.sh).

See also: [data sources](data-sources.md), [model](model.md#calibration-to-published-aggregates), [validation](validation.md).
