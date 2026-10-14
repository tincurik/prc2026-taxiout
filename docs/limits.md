# Limits and failure modes

## Labels with no warning

The two largest errors are LFPG departures from January 2025 with no flight plan and an ordinary take-off, whose recorded
off-block lies 23.4 h and 16.2 h before take-off. They hold 36.5 % of the reference chain's cross-validated squared error and nothing in the inputs
flags them. Seven of the 77 rows above 20,000 s in 2025 are like this. Any test month can contain a few, and they move its RMSE by
more than any modelling choice. Without these two rows the reference chain scores 227.44 s instead of 285.44 s. Every score in this repository includes them;
the 227.44 is given only to show their weight.

## Long genuine taxis are shrunk

Rows that truly took 1–2 h are predicted about 990 s too short, and rows above 2 h about 5,600 s too short. The inputs rarely show
why a taxi is long (a remote hold, a de-icing queue, a closed runway). The conditional mean then stays near the ordinary taxi,
which is the right answer under squared error but a poor estimate of those rows.

## LIRF (Rome Fiumicino)

LIRF holds 24 % of the reference chain's cross-validated squared error.

- The copy classifiers' AUC stops near 0.88–0.91. Copies of delayed flights depend on the handler and the day in ways no available
  field records fully.
- Its summer error is the highest of any airport. The LIRF no-plan classifier does not gain in the June 2025 rehearsal, and in the
  test set it acts mostly on January and February 2026, where LIRF is 77 % and 70 % ADS-B matched against 0 % in the same months
  of 2025, so no validation month resembles them ([validation](validation.md)).
- LIRF's ADS-B coverage swings between 0 % and 94 % across months; in June and July 2026 it is under 1 %.

## Coverage holes in the open sources

![Share of departures with an ADS-B match, by airport and month](../writeup/figures/fig_coverage.png)
*Istanbul and Paris are almost never seen on the ground; Madrid, Heathrow and Rome switch between years.*

- **LTFM** has almost no open ADS-B ground coverage (under 1 %) and **LFPG** 1–33 %: the stackers can only use clocks there.
- **EDDF** fell from 91–98 % in every 2025 month to 73–76 % in January, June and July 2026 and 26 % in February 2026; the start
  (29 January) lies in the source's own daily files. The cause is unknown. In the CV months the stackers lowered the MSE of matched
  EDDF rows from 14,015 to 11,780 s² (RMSE 118 → 109 s). If each of the roughly 10,000 EDDF departures of February 2026 that lack
  a match they would have had at the 2025 level lost that whole gain, the test RMSE would be at most about 0.07 s higher.
- **LEMD, EGLL** are far better covered in 2026 than in the same months of 2025 (LEMD 0 % → 97 % in January); **LIRF** June and
  July are far worse.
- **OPDI** own-flight events are almost absent at EDDM from 21 January to 19 February 2026 (20 % of departures against 84 % a year
  earlier). The regressors then see these features as missing; the effect is not measured. Removing OPDI from every member costs
  0.51 CV RMSE.

## Weather months

LTFM in February 2025 (snow) has the worst error on rows up to 2 h of any airport outside LIRF in the rehearsal: 363 s, against
205 s in June 2025 and 209 s / 259 s in the January / July folds (reference chain). The monthly runway calibration, tested and not in the final chain, skipped the airports that de-ice after off-block
(EDDF, EDDM, LFPG, LSZH) in January, because the published mean excludes de-iced flights.

## Dependence on published aggregates

The daily delay calibration needs EUROCONTROL to have published the figure for the scored days (it has, for all 1,200 scored airport-days).
A day without a published value is left unchanged.

## Validation optimism

See [validation](validation.md#where-the-numbers-are-optimistic): early stopping and thresholds set on the folds, three parts
that are not fully out of fold (the month-complete stackers, the LIRF no-plan classifier, the post-step pool), and a rehearsal
that is an upper bound.

## Record order at LTFM

The ordering features ([features](features.md): a record's id minus the median id of the 250 departures nearest in take-off order on each side, the count of flights past their `EOBT` but not yet off-block, route-centred NM durations, days of runway works at Paris-CDG) read the airport's record ids as a stand-in for the order in which the
plan was made. At LTFM that order changed between years: the Spearman correlation between record id and take-off time is 0.41–0.45 in every 2025 month and 0.93–0.94 in all four 2026 months ([validation](validation.md#what-changes-between-2025-and-the-2026-test-months)). A model that learned from 2025 how the id deviation relates to the wait therefore reads a different kind of input in the test months at LTFM. The check was made on LTFM only.

What is known about their worth is old and small. In earlier runs on the 2025 folds (late September 2026) the ordering block entered as a member and as a residual model on the rows without an ADS-B match. The member changed the validation-fold CV by −0.42 [−0.78, −0.12] (neutral under the keep rule of that time, a paired interval that includes zero, kept); the LOBT-anchored member with it by −0.10 [−0.19, −0.02] (kept); the residual model on the non-ADS-B rows by −1.64 on the folds, but +0.47 on a cross-fit by month, where its gain turned out to be a month-level recalibration, so it was dropped. Other variants that added the ordering block were rejected (+0.43 to +1.36).
No ablation was run on the final chain, so the effect of the ordering features on the 2026 months at LTFM is not measured. They were kept because the measured effect on 2025 was small and mostly favourable and the feature is one input among many; the risk is that at LTFM it carries a relation that no longer holds. If this model is reused on other months, an ablation of the ordering features at LTFM is the first check to run.

## Bias by airport

On rows up to 2 h the model is a few seconds low at LTFM, EGLL, LFPG and LEMD in both folds (−1 to −14 s) and both rehearsal
months (−5 to −22.5 s). The final chain has no step that pulls a runway-month toward a published mean (the tested runway calibration moved a
cell only a quarter of the way).

## Changes between years

LTFM's record ids follow take-off order in 2026 but not in 2025; NM's default taxi time at LFPG changed level; traffic changed by
−8 % to +7.5 % per day at some airports in June. No cross-validation number isolates the first.

## Reproducibility of numbers

CatBoost on GPU is not bit-reproducible, so a rerun lands within seed noise of the recorded CVs (one draw is about ±0.1 RMSE),
not on them. The delayed-row gate therefore averages five seeds. Every other fitter is seeded.
