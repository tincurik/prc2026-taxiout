# Using and extending the model

How to run the model on new months, drop an input, and add a feature, a member or an airport. Every command and path below was
read from the code in [`code/`](../code/README.md) (flags from the scripts' own argument lists); the worked example in section 3
was also run on the synthetic data of the smoke test. Where a step is **not a switch** in the code, the page says so and names
the lines to edit; nothing here has been measured on a changed model unless a number is given.

**Conventions.** Run every command from `code/`. `EC_PHASE` picks the data slice a script works on: unset is the 2025 side (the two
season folds), `final` is the months to predict, `pseudo2` … `pseudo6` are the five rehearsals (their command-line names) that each hide two months of 2025
([reproduction](reproduction.md)). A phase keeps its files in `data/processed_<phase>/`, `outputs/overnight_<phase>/` and
`reports/overnight_<phase>/`. A *step* has an id (`mix_v34_cat_lr15_ord`, `ar_134na_avg4`); its predictions are
`outputs/overnight[_<phase>]/{oof,test}/<id>.parquet` and its settings and score `reports/overnight[_<phase>]/results/<id>.json`
(recorded versions: [`code/recipes/`](../code/recipes/)). A feature file is `data/processed[_<phase>]/features_dep_<name>.parquet`,
one row per departure, keyed by `MVT_ID_mvt`; models and stackers name it as `--extra <name>`.

## 1. Predict new months

**What the model needs per month.** Two files in the layout of the challenge's `final_ranking.parquet` and
`final_submitting.parquet`:

- the ranking file holds every movement (departures and arrivals) of whole calendar months at the ten airports. Departures have
  `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` empty; arrivals keep their in-block and taxi-in times, because features read them
  ([task and data](task-and-data.md#what-the-challenge-files-contain));
- the template holds `MVT_ID_mvt` and an empty `TAXITIME_SEC_mvt` for every departure of the ranking file
  (`scripts/fetch_final.py` checks that every template id is a departure of the ranking file).

Besides these, the months need the open sources of [data sources](data-sources.md): weather (METAR), OPDI events, the EUROCONTROL
tables, and ADS-B ground points for every day of the month. The steps that fetch them read the months from the ranking file, so
they follow it.

**What can be reused.** Stages 0–3 (the 2025 side), the five rehearsals and the stream members' 2025 out-of-fold files
(`bash pipeline/39_bnd1.sh base`, then `members` in each rehearsal) read 2025 only and never see the new months, so they are built once
and reused. The stackers' training pairs come from the rehearsals; a new month costs the `final` phase and nothing else.
Back up the previous `data/processed_final`, `outputs/overnight_final` and `reports/overnight_final` first (or work in a second
checkout), because the phase directories are reused and files are rewritten in place.

**Steps.**

```
# 1. put the two files in a directory; replace the ones in data/raw if they exist (the script refuses to overwrite a different file otherwise)
python scripts/fetch_final.py --from-dir <dir> --refresh

# 2. data, features, members and the reference chain of the phase: weather/OPDI/EUROCONTROL tables, the ADS-B days of the months, features,
#    test-side member refits, then the chain with its month-complete stackers (hours, see "Time and hardware"); FINAL_DIR makes the
#    data stage use the files of step 1 instead of the organisers' bucket
FINAL_DIR=<dir> EC_PHASE=final bash pipeline/50_final_phase.sh data seed features members chain

# 3. the shipped bundle chain (BND_NB=1): stream-transformer members and the bundle chain (the same two commands in each of the five
#    rehearsal phases pseudo2 ... pseudo6 first), then the post-steps' pools in the rehearsal phases and the four post-steps on the bundle's output
BND_NB=1 EC_PHASE=final bash pipeline/39_bnd1.sh members
BND_NB=1 EC_PHASE=final bash pipeline/39_bnd1.sh chain
bash pipeline/39_post_steps.sh pools
EC_PHASE=final bash pipeline/39_post_steps.sh run

# 4. coverage gate, then the plan-window post-step, the template check and the sha256
EC_PHASE=final python scripts/final_coverage.py --require adsb2m
bash pipeline/40_final_file.sh outputs/overnight_final/test/ar_134naclh_tww.parquet
```

Step 2's chain already produces the stage-3b file `ar_132_prc2`, so skipping step 3 and running `bash pipeline/40_final_file.sh outputs/overnight_final/test/ar_132_prc2.parquet`
gives a complete file of a chain without the stream-transformer inputs, the neighbour-label head and the post-steps; step 3 runs
the stackers a second time. The post-steps fit on the pool of the final phase's Jan/Jul out-of-fold rows and the five held-out pairs ([model](model.md#four-post-steps-on-the-chain-output)).
Without `FINAL_DIR` the data stage tries the organisers' bucket, which needs S3 credentials even if the files are already there. The coverage gate stops
when a feature file or member covers a new month worse than 0.8 times its training share; a month with no ADS-B ground points at an
airport will stop there, and the message names the file ([limits](limits.md#coverage-holes-in-the-open-sources)).

**Things that are fixed to 2025–2026 in the code.** The model itself has no calendar. The download lists have: the EUROCONTROL
tables are listed for the years 2024–2026 (`PRC_CSV` in `scripts/fetch_external.py`), the adsb.lol release lists for 2025 and 2026
(`preferred_urls` in `scripts/adsb_fetch.py`), and the daily calibration reads the years 2025 and 2026 of the published table
(`overnight/dly_cal.py`, [daily delay file](daily-delay-file.md); so does `invalid_share` in `overnight/prc_cal2.py`, the runway
calibration of the earlier chains). A month of 2027 needs those lists extended. The weather window
follows the ranking file (`FINAL_MONTHS` in `fetch_external.py`). The shares in `overnight/month_weighted.py` (`SHARE`) are the four
test months' row shares: use that script for the rehearsals, not as a score of new months.

## 2. Run without ADS-B, or without the published statistics

**Without the published statistics.** One switch and one edit:

- `TAIL_SKIP=lplan,dlyh,prc25,prc2 BND_NB=1 EC_PHASE=final bash pipeline/39_bnd1.sh chain` leaves out the daily-delay calibration
  (`dlyh`, reads `data/external/prc`) as well as the plan-row classifier and the runway-month calibration (`prc2`, with its side branch
  `prc25`, which reads `data/external/prc_dashboard`), the two steps the final chain already leaves out. `overnight/tail_chain.sh` lists the
  steps in its header. The chain still ends at the duplicated-record rule (`ar_134na_dup`), which the four post-steps read (`bash pipeline/39_post_steps.sh run`), and
  the file step takes their output as in section 1. Without the daily step the chain before the runway step was dropped scored
  1.41 CV RMSE worse, 0.72 on twelve months ([daily delay file](daily-delay-file.md#what-it-is-worth)).
- the members also read EUROCONTROL's published day-level traffic and delay as a feature family, `ext_prc`. It is a name in the
  `EXT` list of `pipeline/20_members.sh`, `pipeline/50_final_phase.sh` (stage `members`) and `pipeline/38_stx_members.sh`. Removing it is an
  edit of those three lines, followed by the members' refits and the blend; it is **not a switch** and its effect has not been measured.

**Without ADS-B.** ADS-B enters in four places: the `adsb2fullrel` members' feature family, the `adsbdir4` and `adsbrev`
stacker inputs, the match features `adsb2m` (`FEAT` of `pipeline/30_chain.sh`, `--feat` of `stack12.py`) and the stream
transformers' extras.

- *Rows without a track:* a departure with no ADS-B match is handled by design: the stackers correct it from clocks and the chain
  prediction (`--unmatched 1`), as they already do for Istanbul and Paris, where open ground coverage is almost absent
  ([limits](limits.md#coverage-holes-in-the-open-sources)).
- *No ADS-B at all:* the code has **no switch for it**. It would mean removing `adsb2fullrel` from `EXTA` in the member scripts, dropping
  `adsbdir4` and `adsbrev` from `M12_EXTRA` (the environment variable that lists the extra feature files the month-complete stackers read) and the match features `adsb2m`, and refitting. The first stacker of the project corrected a chain that had no ADS-B
  input and was worth −12.45 RMSE ([results](results.md#from-the-baseline-to-the-final)); this is the reference for what an ADS-B-free
  chain leaves out, not a measured variant of the shipped one.

## 3. Add a feature family (worked example)

A feature family is one script that writes `features_dep_<name>.parquet`. Models and stackers pick it up by name; there is no other
registry. Three rules: **label-free** (only clocks, categories and other flights' records, as for every feature,
[features](features.md)), **one row per departure** keyed by `MVT_ID_mvt` (NaN is allowed, trees handle it; a feature that is missing
for a whole month is stopped by the coverage gate), and **built in every phase**.

*Example: take-offs from the same runway in the ten minutes before this one.* (The feature itself is an illustration, not part of the
model.)

```python
# overnight/features_rwy10.py
import numpy as np, polars as pl
from ocommon import PROC          # PROC follows EC_PHASE: data/processed or data/processed_<phase>

d = pl.read_parquet(PROC / "features_dep.parquet",
                    columns=["MVT_ID_mvt", "apt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]).to_pandas()
d["key"] = d.apt + "_" + d.RUNWAY_mvt.fillna("NA")
t_all = d.MVT_TIME_UTC_mvt.to_numpy().astype("int64")          # seconds
out = np.zeros(len(d), dtype="float32")
for _, g in d.groupby("key"):
    t = np.sort(t_all[g.index])
    q = t_all[g.index]
    out[g.index] = np.searchsorted(t, q, "left") - np.searchsorted(t, q - 600, "left")
d["rwy_prev10"] = out
d[["MVT_ID_mvt", "rwy_prev10"]].to_parquet(PROC / "features_dep_rwy10.parquet", index=False)
```

1. **Build it in every phase**, the 2025 side first, because every phase has its own feature directory:
   `python overnight/features_rwy10.py`, then `EC_PHASE=<pseudo2..pseudo6|final> python overnight/features_rwy10.py` for the six phases.
   The file is built from `features_dep.parquet`, which `overnight/features.py` writes in each phase. (To have the stages run it, add the
   line to `pipeline/10_features.sh` and `overnight/pf_feats1.sh`.)
2. **Let a member read it.** Members take `--extra`. Refit one with the new name added to its list, as `pipeline/20_members.sh` does for its
   CatBoost members (EXT is `ext_metar,ext_prc,ext_osm,ext_opdiev,ext_ord`):

   ```
   cd overnight
   python -u mix.py --id mix_v34_cat_lr15_ord_rwy10 --fs3 1 --p_from mix_v18_fs3_deep --reg_model cat --reg_clip 5000 --es_clip 3600 \
     --reg_target resid --reg_params '{"learning_rate":0.15,"iterations":8000}' --extra ext_metar,ext_prc,ext_osm,ext_opdiev,ext_ord,rwy10
   python -u final.py --from mix_v34_cat_lr15_ord_rwy10 --scale 1.15
   ```
   `mix.py` joins every column of the file except the ones named in `NON_FEATURES` (`overnight/bench.py`), so any numeric column
   becomes a feature; `--extra` is also accepted by `mlp.py` and `stx.py`. The second command is the test-side refit
   (iterations = the folds' mean × 1.15).
3. **Let the stackers read it.** The stackers' inputs are the files named in `--extra` (`adsb_stack.py`, `stack12.py`) and, for the shipped
   chain, in `M12_EXTRA`, a comma separated list of feature-file names (without `features_dep_`) read by `35_m12_stackers.sh` ([`pipeline/39_bnd1.sh`](../code/pipeline/39_bnd1.sh) builds it):
   `M12_EXTRA=adsbdir4,adsbrev,ef_10111040_stxctx_memb,ctx,ts_kdu_f,rwy10 EC_PHASE=final bash pipeline/35_m12_stackers.sh x c l w avg`.
   The file must exist in the final slot and in all five rehearsal phases; `39_bnd1.sh chain` checks the same for its own extras.
4. **Check it** (section 6). A member judged on the two folds: `python overnight/compare.py <reference id> <new id>`. A stacker input judged on
   twelve months: `stack12.py` over every phase, then `month_weighted.py`.

The commands of steps 1 and 2 ran on the smoke test's synthetic data (CPU, a few seconds, `mix.py` with its caps); the numbers there are
noise and are not quoted.

## 4. Add or swap a member

A member is a model that predicts the physical taxi `reg` (and shares the copy probability of the copy classifier `mix_v18_fs3_deep`, the column `p_imp` in the code). Its files are
`oof/<id>.parquet` (the two folds) and `test/<id>.parquet`/`test/<id>_raw.parquet` (the refit on all of 2025, written by `final.py`
or by the neural scripts' `--final 1`).

- **The low-touch way** is the one the context transformer (`stx_ctx`) uses: leave the blend alone and give the stackers the member's disagreement.
  `python overnight/features_memb.py --all 1 --add <id> --out <name>` writes a disagreement file in which the new member counts with weight
  0.05 next to the eleven; name it in `M12_EXTRA` in place of `memb2` (the disagreement file of the eleven regressors alone).
- **The full way** is to put it in the blend. The eleven regressors are the list `R` in `pipeline/30_chain.sh`; the blend is
  `blend.py --id blend_v48 --p_from mix_v18_fs3_deep --regs $R` (NNLS weights, cross-fitted) and `final_blend.py --parts $R --weights_from blend_v48`
  for the test side. Add the id to `R`, add the member's training and `final.py --from` lines to `pipeline/20_members.sh` and the `members`
  stage of `pipeline/50_final_phase.sh` (the rehearsals refit every member), and give the blend a new id, because `blend_v48` is the recorded
  one. `features_memb.py --blend <id>` follows the blend you name.
- **Swapping** is the same with the id replaced. The rules chain, stackers and tail read the blend through their `--base` ids
  (`rc_v34_more18`), so nothing after the blend changes name.
- **A new kind of model:** copy the `--id`, `--extra`, `--final` and `--refit_only` conventions of `mlp.py` or `stx.py`: write
  `oof/<id>.parquet` and `test/<id>_raw.parquet` with `MVT_ID_mvt` and `reg`, and everything downstream can read it.

## 5. Add an airport

The airport list is not one setting; it is the places below. All of them have to know the airport, and the model needs labelled months
of it to learn from (the airport enters as a category, so a new one has nothing to start from without them).

| place | what to add |
|---|---|
| `overnight/features.py` `TZ` | the airport and its time zone. **Required:** departures at an airport outside `TZ` are dropped from the feature table. OPDI extraction (`overnight/opdi_extract.py`) follows `TZ` too |
| `scripts/fetch_external.py` | `AIRPORTS`, the holiday region in `subdiv`, and the OSM taxi network (fetched per airport into `data/external/osm/<ICAO>.json`) |
| `scripts/adsb_fetch.py` `APTS` and `overnight/adsb_geo.py` `REF` | the airport's reference point, for the ADS-B ground and approach zones and the taxi map |
| `scripts/fetch_prc_dashboard.py` `APTS`, `overnight/prc_cal2.py` `APTS` | the airport in the monthly runway calibration (only for the earlier chains; the final chain does not run it) |
| `scripts/witness_post.py` `TZ`, and `G7` there if the airport records off-block on whole minutes | the plan-window post-step's clock |
| `overnight/prc_cal.py` `DEICE` | whether the airport records off-block after de-icing (read by the monthly runway calibration only) |
| airport rules: `LIRF` in `rules_chain.py`, `lirf_far.py`, `lirf_bel.py`, `clockfix.py`, `gate2.py` | **not generic.** They encode Rome's recording habits; a new airport starts without them and the mixture handles its copies (`lirf_far.py --apt` takes another airport, its docstring names Barcelona as the next candidate) |

After the edits, rebuild the features and refit members and stackers for every phase, because the category set changes. The stackers read
their airport list from the data (`stack12.py`), and `LIRF` is left uncorrected by default through `--skip`. The recorded numbers describe
the ten airports only.

## 6. Check a change

- **Does everything still run?** `bash smoke/run_smoke.sh [work dir]` runs every stage up to the bundle chain (3c) on synthetic data on the CPU (2,477 s on Python 3.14,
  2,632 s on 3.12, about 2 GB of disk) and ends with `SMOKE OK`. It does not measure accuracy
  ([smoke test](../code/README.md#smoke-test-cpu-no-data-no-network)).
- **Two chains on the two folds:** `python overnight/compare.py <reference id> <candidate id> [--n 1000]` is the paired bootstrap over whole
  days (all rows, rows up to 20,000 s, rows up to 7,200 s); negative = the second is better. It reads `outputs/overnight[_<phase>]/oof/<id>.parquet`.
- **Two chains on twelve months, weighted like the test months:** `python overnight/month_weighted.py <candidate id> <reference id>
  [--phases final pseudo2 ... pseudo6] [--refit_sd <MSE>]` ([validation](validation.md#twelve-months-weighted-like-the-test-set)). Both ids
  must exist in `final` (`oof/`) and in each rehearsal phase (`test/`): for a changed stacker, run it over every phase
  (`python overnight/stack12.py --id <id> --extra <names> --arms m12 --seeds 3 --phase_oof 1`; averaging and the tail then run per phase, as `35_m12_stackers.sh avg` and
  `bash overnight/tail_chain.sh <tag> <avg id>` do under `EC_PHASE`). It prints the change per month and the weighted estimate with its interval.
  `--refit_sd` adds the noise of a refit (about 0.1 RMSE for a GPU CatBoost step, [reproduction](reproduction.md#what-has-been-replayed)) when the change refits a model.
- **Is the file valid?** `python scripts/validate_file.py <file>` (EC_PHASE picks the template).

## 7. Reuse in OPDI

OPDI derives flight events from ADS-B tracks. Three parts of this pipeline port to it, each with its input and output in OPDI and APDF terms. In the
challenge files the airport's recorded off-block is `BLOCK_TIME_UTC_mvt` (AOBT as the airport recorded it), the recorded take-off is `MVT_TIME_UTC_mvt` (ATOT),
and the scheduled off-block is `SCHED` (SOBT). All times are UTC seconds; durations are seconds.

| our step | what it does | OPDI / APDF event it maps to | input | output |
|---|---|---|---|---|
| ground-event extraction ([`overnight/features_adsb2.py`](../code/overnight/features_adsb2.py)) | per departure, cuts the aircraft's ground segment out of the ADS-B points (a segment ends at an airborne point of 90 kt or more and is cut at gaps of 15 minutes) and times its events: first point heard, first movement (ground speed of 1 and 3 kt), 15 m and 30 m from the first point, last point within 30 m of it, start of the take-off roll (last ground speed under 30 kt), stops and the end of the last one, and whether the first point is the parked aircraft at its stand (under 1 kt, within 60 m) | ATOT is the `take-off` event; first movement is the nearest ADS-B event to the off-block time (OPDI's `exit-parking_position`, where the stand and the track allow it), but it is not the off-block: the open feeds usually start mid-taxi, see [ADS-B timing against the label](#8-notes-on-what-was-measured); roll start is next to `entry-runway` | ground points per airport-day: UTC time, latitude, longitude (deg), ground speed (kt); stand coordinates (deg) learned from the arrivals' parked positions | one row per departure, each event as take-off minus the event (`a2_gs1`, `a2_roll`, …, s), plus stops, stopped seconds, path length (m) and gap (s) |
| copy detection ([`overnight/mix.py`](../code/overnight/mix.py), [model](model.md#the-mixture)) | the probability that a departure's recorded AOBT is its SOBT copied in, not an event: `p` = P(`|BLOCK − SCHED|` ≤ 30 s) | a data-quality flag on AOBT; where `p` is high a port can take AOBT from the track instead of the record | the recorded AOBT, SOBT, the flow manager's own AOBT, LOBT and EOBT, whether a flight plan matched, airline, stand; label-free | `p` per departure, calibrated; we use it as the weight of the mixture |
| physical-taxi component ([`overnight/mix.py`](../code/overnight/mix.py), the regressors) | the taxi-out time of departures whose record is not a copy: take-off minus off-block, in seconds, from stand, runway, surface load, weather and the ADS-B events above | the interval from AOBT (or its track proxy) to ATOT | the same features and the clean rows only (`p` low) | `reg` in seconds |

What ports as it is: the three steps need no label of the month they predict, only the recorded clocks and the tracks. What does not port: the
Rome-specific rules ([model](model.md#rules-chain)), the calibrations to EUROCONTROL's daily and monthly aggregates (they read published totals of the
same records) and the four post-steps, which correct this pipeline's own residuals. OPDI's own ground events are sparse (the parking-position exit
is on under 0.1 % of departures in the extract we read), which is the gap the ADS-B extraction fills, so the extraction reads the ground points
directly instead of OPDI's events. Our field names carry the unit only in the documentation (`a2_gs1` is seconds); a port to OPDI would put it in the name
(`..._s`, `..._kt`, `..._m`).

### A monthly run, and what it would report

The PRC job is the one of [section 1](#1-predict-new-months) with the month closed: the airports' movement table and the flight-plan table for the month are in, so the label-free features can be built.

| step | command (from `code/`) | in | out |
|---|---|---|---|
| 1 ADS-B days of the month | `make adsb DAYS=<YYYY-MM> PLAN=1`, then without `PLAN=1` | the days; about 105 GB streamed and 6 GB written for 30 days, about 1.6 hours on three workers | `data/external/adsblol/points/<day>.parquet` and the sha256 manifest |
| 2 the phase | stages 0 to 4 of [section 1](#1-predict-new-months) with the new files; the rehearsals and the 2025 side are built once | the month's two files (ranking, template), the open sources | the chain's predictions for the month |
| 3 the file | the coverage gate, the plan-window step, `check_final.py` | the chain output | one row per departure: `MVT_ID_mvt`, estimated taxi-out seconds |

Per-airport budget the monthly report can carry, from the two validation folds of the reference chain (RMSE in seconds on rows with a label up to 2 h, so that the label artefacts of
[limits](limits.md) do not decide it; the all-row figure is in [results](results.md#by-airport)):

| airport | January | July | ADS-B match, January / July |
|---|---:|---:|---|
| EDDF | 121.4 | 128.3 | 0.91 / 0.94 |
| EDDM | 79.3 | 143.5 | 0.97 / 0.37 |
| EGLL | 193.1 | 244.7 | 0.68 / 0.29 |
| EHAM | 115.9 | 98.9 | 0.99 / 0.98 |
| LEBL | 90.5 | 225.5 | 0.98 / 0.94 |
| LEMD | 154.0 | 169.5 | 0.00 / 0.30 |
| LFPG | 244.7 | 232.4 | 0.02 / 0.11 |
| LIRF | 293.3 | 505.3 | 0.01 / 0.94 |
| LSZH | 113.7 | 106.4 | 0.97 / 0.96 |
| LTFM | 208.7 | 259.2 | 0.00 / 0.00 |

LIRF is the largest single budget item (the Rome recording habits, [model](model.md#rules-chain)); LTFM has no ADS-B match in either month.

A monthly figure per airport is meaningful against these only with the airport's ADS-B match share next to it, since the error is about 45 s lower where the track is known
([results](results.md#by-airport)); and a month with a Rome date stamp or a day-long record moves an airport's all-row RMSE by hundreds of seconds, by the record, not by the model.

**What would have to change for OPDI to carry the ADS-B step.** The model reads the ground points (`features_adsb2.py`), not events, so OPDI's events would replace the track features with fewer
inputs: OPDI's parking-position exit is on under 0.1 % of departures in the extract read here, so that port is a design task, not a switch, and nothing here has been measured on it.
The copy-detection and physical-taxi steps need no track at all and are the part that moves as it is.

## 8. Notes on what was measured

Three questions a reader of the OPDI reuse above would ask. The numbers are from the author's own runs on the 2025 files (their working files are not part of this repository, so they can be recomputed from the inputs but not looked up here); nothing was run for this page.

### ADS-B timing against the label

The label is take-off minus the airport's recorded off-block, so a track event could stand in for the off-block only if it were close to it. It is not. The event is the first movement
(`a2_gs1`: ground speed of 1 kt or more; take-off minus the event is the feature). The table compares the event's time with the airport's recorded off-block (`BLOCK`) and with the Network Manager's off-block `AOBT_3`,
on the 2025 departures matched to a track with a label above 0 and up to 3,600 s (the adsb.lol points; positive = the event is later than the recorded off-block):

| airport | departures | event − `BLOCK`, median (quartiles), s | within ±60 s | event − `AOBT_3`, median (quartiles), s | within ±60 s |
|---|---:|---|---:|---|---:|
| EDDF | 218,429 | +315 (+64, +455) | 23 % | +376 (+164, +555) | 7 % |
| EDDM | 140,992 | +50 (+24, +111) | 56 % | +71 (−99, +220) | 18 % |
| EGLL | 94,441 | +640 (+244, +972) | 0.5 % | +693 (+337, +1,039) | 4 % |
| EHAM | 240,791 | 0 (−17, +209) | 59 % | +198 (+41, +375) | 16 % |
| LEBL | 173,659 | +258 (+31, +446) | 32 % | +218 (−17, +446) | 13 % |
| LEMD | 143,902 | +392 (+153, +529) | 4 % | +388 (+180, +552) | 7 % |
| LFPG | 17,941 | +758 (+600, +942) | 0.9 % | +734 (+540, +965) | 1.2 % |
| LIRF | 67,468 | +631 (+444, +863) | 3.6 % | +511 (+266, +702) | 4.9 % |
| LSZH | 126,148 | +182 (+31, +364) | 35 % | +205 (−8, +352) | 12 % |
| LTFM | 1 matched | not computed | | not computed | | |

At EDDM and EHAM the first movement is within a minute of the airport's off-block for more than half of the departures (EHAM's median is 0 s); at the other airports the track usually starts mid-taxi, a median of 3 to 13 minutes after it.
`AOBT_3` is further from the track event than `BLOCK` at five of the nine airports (EDDF, EDDM, EGLL, EHAM, LSZH) and closer at four (LEBL, LEMD, LFPG, LIRF), so neither recorded clock tracks the physical start of movement. (Computed for this page from the 2025 feature files; the script is a few lines of `polars` over `features_dep_adsb2m.parquet` and the movement table.) Only a track that starts at
a parked aircraft (under 1 kt, within 60 m of the stand) has an off-block to read, and those are 2–29 % of departures (EDDM 29 %, EHAM 26 %, LSZH 21 %, LEBL 15 %, EDDF 8 %, LEMD 7 %, EGLL 3 %, LIRF 2 %, LFPG and LTFM 0 %).
On those rows (counted on an earlier version's ADS-B points) a track-derived label, calibrated against the official one, has an RMSE of 87 (EDDF), 68 (EDDM), 111 (EHAM), 137 (LEBL), 139 (LEMD), 85 (LSZH) and 167 s (EGLL), against 103, 73, 92, 178, 126, 91 and 167 s for the final model's out-of-fold predictions on matched rows:
no better than the model, and absent at LFPG, LTFM and LIRF where the error is. The model therefore uses these events as inputs to the stackers, never as a label or as the off-block time. No benchmark against the organisers' off-block beyond this one was made.

### Wake category and the type sequence

Only the own aircraft's wake category (`WK_TBL_CAT_flt`) and its aircraft type are inputs of the shipped chain, as categorical columns of the first-level models (`CATS` in `overnight/bench.py`).
The sequence in front of the aircraft is not an input. A runway-configuration block with wake-weighted counts of same-runway departures in the 20 minutes before and after (heavy 3, medium 1.5, light 1), the share of departures on the aircraft's own runway and a cadence term was built and tried as a block of the second-level models: neutral (change in RMSE 0.03 s, interval −0.12 to +0.06 s on the final-month and one pseudo-test fold), so it was not shipped and its script is not in this repository. A wide-body flag is not an input either.
The leader–follower pair (the class of the previous departure on the same runway, within 150 s) is *not* a feature. It was looked at only as a residual: the mean error of the stacked model for a medium aircraft behind a heavy one is −23 s at LFPG, −15 s at LTFM (medium behind medium +6) and −7 s at LEMD, about 0 at EDDF, EDDM, EHAM, LSZH and LEBL, and +13 against +8 at EGLL (medium behind medium against behind heavy); heavy behind heavy is over-predicted by 23 s at EGLL.
That is a physical signature of the spacing behind a heavy aircraft, but the rows are about 2 % of LFPG's departures, so pooled it is small. Whether adding the pair as a feature gains was not tested; nothing in this repository says it is useless or useful.

### Defects in the inputs, by source

| source | defect | rows in 2025 | how the model reads it |
|---|---|---|---|
| airport record (`BLOCK`) | schedule copied in as the off-block | 5.4 % of departures; LIRF 18 % | copy classifier and mixture ([model](model.md#the-mixture)) |
| airport record | off-block a day before take-off | 13 rows above 3,600 s carrying 31 % of the cost of the large labels; LIRF datestamp 12 | `MVT − AOBT_3` rule where the NM stamp is also shifted, LIRF third regime |
| airport record | the same take-off recorded twice (mostly LFPG) | 118 pairs | linear rule for the matched half |
| airport record | LIRF no-plan departures over 20,000 s after schedule; a two-carrier group about two hours early | 48 of 50 are copies; 17 of 20 in the group | predict `MVT − SCHED`; group mean ([task and data](task-and-data.md#how-the-recorded-off-block-goes-wrong)) |
| NM flight table | off-block `AOBT_3` a day early, or a 20-minute default at airports without a connected A-CDM (LTFM) | the date-shifted rows above; LTFM throughout (A-CDM there is local and not integrated with the Network Manager) | the `> 20,000 s` rule |
| NM plan join | plans attached only inside `\|BLOCK − LOBT\| ≤ 3,606 s` | all 2,062,577 departures with a plan | clip and the witness step ([model](model.md#plan-window-post-step)); a property of how the dataset was built, not of aircraft |
| ADS-B feeds | coverage swings by airport and month; rows first heard after take-off | LTFM 0–11 %, LFPG 2–33 %, EDDF 29 % in February 2026 ([limits](limits.md#coverage-holes-in-the-open-sources)) | inputs of the stackers; a row first heard after take-off is corrected by about −25 s on average in cross-validation |
| ordering id `MVT_ID` | at LTFM its relation to take-off order changes between years | Spearman 0.41–0.45 in 2025 months against 0.93–0.94 in 2026 months ([validation](validation.md#what-changes-between-2025-and-the-2026-test-months)) | kept as an input; no ablation, see [limits](limits.md#record-order-at-ltfm) |

Coverage of the three open inputs by airport, on the 2,085,047 departures of 2025:

| airport | departures | ADS-B track matched (all of 2025) | METAR missing | OSM stand: exact name / by base name or nearby number / none | OSM route to a runway missing |
|---|---:|---:|---:|---|---:|
| EDDF | 230,141 | 95 % | 0 | 96.8 % / 2.4 % / 0.8 % | 0.8 % |
| EDDM | 167,334 | 85 % | 0.04 % | 58.1 % / 29.8 % / 12.1 % | 12.1 % |
| EGLL | 239,546 | 43 % | 0 | 98.5 % / 1.4 % / 0.01 % | 0.01 % |
| EHAM | 247,951 | 98 % | 0 | 98.2 % / 0.6 % / 1.1 % | 1.1 % |
| LEBL | 179,705 | 97 % | 0 | 89.7 % / 2.3 % / 8.0 % | 8.0 % |
| LEMD | 212,242 | 68 % | 0 | 98.8 % / 1.0 % / 0.2 % | 0.2 % |
| LFPG | 239,552 | 10 % | 0 | 96.9 % / 2.8 % / 0.2 % | 0.2 % |
| LIRF | 160,704 | 44 % | 0.01 % | 99.0 % / 1.0 % / 0.05 % | 0.07 % |
| LSZH | 134,907 | 97 % | 0.0007 % | 96.4 % / 0.03 % / 3.6 % | 3.6 % |
| LTFM | 272,965 | 0.01 % | 0 | 99.2 % / 0.7 % / 0.1 % | 0.1 % |

METAR is missing for at most 0.04 % of departures at any airport. OSM has a stand for almost every departure except at EDDM (a stand is
missing for 12 % of departures there) and LEBL (8 %). The ADS-B column is the share of departures with a matched track in any of the ADS-B days used (the Jan/Jul figures of [results](results.md#by-airport) are for those two months only).

The counts are the ones in [task and data](task-and-data.md#how-the-recorded-off-block-goes-wrong), [limits](limits.md) and [validation](validation.md). An excess-taxi output (take-off minus an unimpeded taxi) is not produced: the model predicts the recorded label, which includes the record's own defects.

## Time and hardware per step

As recorded in [`code/README.md`](../code/README.md#stages); "not timed" means the run did not record it.

| step | hardware | time |
|---|---|---|
| download of the new months' ADS-B days | network, disk | about 3.5 GB streamed a day; not timed |
| `50_final_phase.sh … features` | CPU, 16 GB RAM or more; `adsb_direct.py` uses a GPU | the ADS-B steps take hours on 8 workers; not timed |
| `50_final_phase.sh … members` (test-side refits) | one CUDA GPU; the largest CatBoost member needs about 19 GB of RAM while loading | not timed apart from stage 2 as a whole: about 5 GPU-hours for the 2025 folds and the refits |
| the reference chain of the 2025 side or of a rehearsal (`CTX=0`, Jan/Jul stackers) | CPU, 10 threads and up to 16 GB per stacker; the delayed-row gate uses the GPU | about 25 min with the 5-seed gate, on the machine of the recorded run |
| `50_final_phase.sh … chain` for `final` (month-complete stackers) | the same; with a card visible the stackers use it | the four stackers took about 2.3 h and 2.9 h in two parallel lanes; the tail about a minute |
| `39_bnd1.sh members` | one GPU, 14 GB RAM | about 15–17 min per member and phase |
| `39_bnd1.sh chain`, the four stackers (xgb, cat, lgb2, covw) | CPU (or a GPU with `STACK_GPU=1`, LightGBM through OpenCL) | about 2–3 h on CPU, one after the other; 2.3 h and 2.9 h in two parallel lanes with two stackers each |
| the tail, the four post-steps, the plan-window post-step, the file | CPU | minutes |
| the whole smoke test | CPU, 2 GB of disk | 41–44 min at 8 days a month, 35 min at 3 |

## Limits of these recipes

- The commands follow the final chain. A change to a stage that the rehearsals feed (members, base chain) needs all five
  rehearsals rerun, because the stackers train on them.
- GPU CatBoost is not bit-reproducible: two runs of the same change differ by about ±0.1 CV RMSE, so a change smaller than that needs
  several seeds or `--refit_sd` before it counts ([reproduction](reproduction.md)).
- Everything is post-operations: features read other flights' records before and after the movement, so the model is a tool for
  estimating the taxi time of flights that have happened, not a forecast.
