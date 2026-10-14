# Pipeline code

The code of a post-operational taxi-out estimator for PRC performance monitoring: it builds the final prediction file stage by stage, and a CPU smoke test runs the whole chain
on small synthetic data. **To check that it runs, start with `make smoke SMOKE_DAYS=3`** (no data, no network, [below](#smoke-test-cpu-no-data-no-network)); to bring in the ADS-B it reads, `make adsb DAYS=<days>`
(resumable, checksummed, with the cost printed first). This tree is a selection of the files the recorded run used, copied from the author's development
tree; [`.code_export`](.code_export) lists what the copy changes (licence headers, comments and help texts, default ids; the stage scripts written for this tree, and what was checked, are listed in the file). It holds no challenge data, no external data and no model weights; the one prediction file it holds is the shipped file in [`final/`](final/README.md).
The write-up around it is in [`docs/`](../README.md); the model is described in [`docs/model.md`](../docs/model.md).

```
Makefile     the one entry point: make help (targets below)
pipeline/    one shell script per stage (00 data ... 50 test months), run from this directory: bash pipeline/<script>
overnight/   the modelling code (features, members, blend, rules chain, stackers, tail) and the model recipes' runner scripts
scripts/     data import / download, phase preparation, the plan-window post-step, validation of the file
recipes/     one JSON file per step: its parameters and the cross-validation it recorded. ar_134na_* up to ar_134na_dup (the bundle and its tail)
             and ar_134naclh_* (the four post-steps, stage 3d) are the chain of which the shipped file averages three draws (`ar_134naclh_tww_bag3_wit_zf3`);
             base / ids name the steps a recipe reads. The intermediate chains of stages 3b and 3c (ar_131 ... ar_133) have no recipe here: their recorded runs read an ADS-B set this tree does not build.
             recipes/other_chains/ holds the recipes of the earlier chain ar_103 (kept with its recorded cross-validation, not read by the final chain).
             A recipe's "phase" is the phase of the recorded run (EC_PHASE=final for the final chain; finalb2 and finalb3 are the two further seed draws of the bag).
smoke/       the synthetic-data generator and the end-to-end CPU smoke test
```

## Environment

`requirements.txt` has every version pinned. `pyarrow==26.0.0` is needed for the prediction file's bytes to match the recorded ones (another
version writes the same values with different bytes); `torch`, `catboost`, `xgboost` and `lightgbm` are pinned because the members' numbers
depend on them. The stages run on Linux with bash, GNU coreutils (`sha256sum`) and Python.

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Python: the recorded results were produced under Python 3.12 ([`docs/reproduction.md`](../docs/reproduction.md)). The pinned versions here install
and pass the smoke test under **both Python 3.12.15 and Python 3.14.4** (results below). Under 3.12 on Linux, `overnight/ocommon.py` starts worker
processes with `forkserver` instead of `fork` (a fork after polars has started its threads deadlocks the worker pool; 3.14 already defaults to it).

## Make targets

`make help` lists them; `make -n <target>` prints the commands without running anything. They call the stage scripts below in order.

| target | stages | what it runs |
|---|---|---|
| `make data FILES=<archive>` | 0 | imports the organisers' archive, builds the movements table, downloads the open sources and the ADS-B ground points |
| `make features` | 1 | every label-free feature file |
| `make members` | 2 | the copy classifier and the 11 regressors (one CUDA GPU) |
| `make chain` | 3, 3b | the chain on the 2025 side, then the five rehearsals |
| `make final` | 4, 3c, 3d, 5 | the four test months, the stream-transformer members and stackers in every phase, the four post-steps, the prediction file |
| `make final-bag [BAG_K=3]` | 5a, 5b, after `make final` | draws 2 and 3 of the seed bag (about 7 h each), then their average with draw 1 and the plan-window step: the shipped file |
| `make bag-draw DRAW=<k>` | 5a | one extra draw only (two can run side by side, one per card) |
| `make bag-file` | 5b | the average and the plan-window step, from draws that exist |
| `make smoke [SMOKE_DAYS=3] [SMOKE_DIR=<dir>]` | all, on synthetic data (about 12 GB of RAM) | the [smoke test](#smoke-test-cpu-no-data-no-network); `SMOKE_DAYS` is the days a month of the synthetic data (default 8) |
| `make test` | none | unit tests of the three plan-window rules (twin window, duplicate pair, plan clip and witness push) on synthetic rows: 20 tests, a second, no data |
| `make adsb DAYS=<days> [PLAN=1] [VERIFY=1]` | the ADS-B part of 0 | the [ADS-B fetch](#fetching-the-ads-b-days): days as `2026-01`, `2026-01-05:2026-01-11` or a comma separated mix; `PLAN=1` prints the cost, `VERIFY=1` re-hashes the files |

## Stages

Run from this directory, in this order. Every script lists its Python commands in order; `PY=<python>` selects the interpreter (default `python`),
`THREADS` the stackers' thread count (default 10). Intermediate files go to `data/processed*/`, `outputs/overnight*/` and `reports/overnight*/` here
(about 6 GB of features and tens of GB of caches; they are ignored by git).

| stage | command | what it does | hardware and time |
|---|---|---|---|
| 0 data | `bash pipeline/00_data.sh files_list.zip` | imports the organisers' archive, builds `data/processed/movements.parquet`, downloads the test months' files (S3 credentials in the environment, [`scripts/fetch_final.py`](scripts/fetch_final.py); `FINAL_DIR=<dir>` uses the two files from a directory instead, with no network or credentials), the open external data and the adsb.lol ground points | network; adsb.lol is streamed (median 3.5 GB a day, about 1.7 TB for the 16 months, resumable per day: [`make adsb`](#fetching-the-ads-b-days) does it with checksums and a stated cost), about 55 GB kept; hours to days, not timed |
| 1 features | `bash pipeline/10_features.sh` | every label-free feature file the models read | CPU, 16 GB of RAM or more; the ADS-B steps take hours on 8 workers (not timed) |
| 2 members | `bash pipeline/20_members.sh` | the copy classifier and the 11 regressors: out-of-fold predictions on the two season folds and the test refits | one CUDA GPU (XGBoost, CatBoost, PyTorch), about 5 GPU-hours; the largest CatBoost member needs about 19 GB of RAM while loading |
| 3 chain | `CTX=0 bash pipeline/30_chain.sh` | blend, rules chain, ADS-B stackers (Jan/Jul rows only) and the record tail on the 2025 side; gives the base prediction the rehearsals need | CPU (10 threads, up to 16 GB per stacker) and one GPU for the delayed-row gate; about 25 min with the 5-seed gate on the machine of the recorded run |
| 3b rehearsals | `EC_PHASE=pseudo2 bash pipeline/50_final_phase.sh data seed features members chain file` (then `pseudo3` ... `pseudo6`; `make chain` does all five) | five runs that each hide two 2025 months (Feb+Jun, Mar+Sep, Apr+Oct, May+Nov, Aug+Dec) like the test months, refit every member on the other ten and run the chain; their held-out predictions train the month-complete stackers | as stages 1–3, test side only; no stage 2 out-of-fold work is repeated |
| 4 test months | `EC_PHASE=final bash pipeline/50_final_phase.sh data seed features members chain` | the same stages on the four test months; stage `chain` calls `35_m12_stackers.sh` (month-complete stackers) and the tail; stage `file` runs `pipeline/40_final_file.sh` (left out of the command above: it makes the stage-3b chain's file `outputs/overnight_final/test/ar_132_prc2_wit_zf3.parquet`, not the shipped one; the shipped file is made at stage 5) | the four stackers took about 2.3 h and 2.9 h in two parallel lanes (two CPU stackers each), the tail about a minute; the member refits need the GPU |
| 3c bundle chain | `bash pipeline/39_bnd1.sh base`, then `EC_PHASE=<pseudo2..pseudo6, final> bash pipeline/39_bnd1.sh members`, then `EC_PHASE=final bash pipeline/39_bnd1.sh chain` and `bash pipeline/40_final_file.sh outputs/overnight_final/test/ar_133_prc2.parquet` | the month-complete stackers of stage 3b with two more inputs, each a switch (`BND_CTX`, `BND_KDU`): the stream-transformer members `stx_ctx` and `ts_kdu3` (see below); gives `ar_133_prc2` | `base` and `members` need the GPU (about 16 and 25 min for the two 2025 fits, about 15-17 min per member and phase, 14 GB of RAM); `chain` as stage 3b (CPU, about 2-3 h) |
| 3d final chain | `BND_NB=1` for stage 3c (`base`, `members` in every phase, then `chain` in `pseudo2` ... `pseudo6` and in `final`; ids `ar_134na`, extras with `stxnb_f` in place of `ts_kdu_f`), the tail stopping at `ar_134na_dup` (no monthly runway calibration), then `bash pipeline/39_post_steps.sh pools`, `bash pipeline/39_post_steps.sh run` and `bash pipeline/40_final_file.sh outputs/overnight_final/test/ar_134naclh_tww.parquet`; `bash pipeline/39_post_steps.sh plan` lists every input and which command makes it | the final chain: the neighbour-label member `stx_nb` instead of `ts_kdu3`, and the four post-steps of the model page (copy-weight re-mix, runway-neighbour corrector, far-row mixture, arrival-twin window), whose coefficients are fitted on the rehearsal phases' copies of the chain (`ar_134na_dup`, `ar_134naclh_rmx` in `pseudo2` ... `pseudo6`); gives `ar_134naclh_tww_wit_zf3` | `nb` member refit about 17 min on the GPU per phase, stackers as stage 3c per phase, the post-steps CPU about 2 min |
| 5 the file | `bash pipeline/40_final_file.sh [chain file]` | the plan-window post-step on the chain output, the template check, the sha256; without an argument it takes stage 3c's `ar_133_prc2`; pass `outputs/overnight_final/test/ar_134naclh_tww.parquet` for one draw of the final chain (stage 3d), or `outputs/overnight_final/test/ar_132_prc2.parquet` for the stage-3b chain's file | CPU, minutes |
| 5a seed-bag draws | `bash pipeline/45_bag_draw.sh 2`, then `... 3` (`PLAN=1` prints the plan and touches nothing) | one more draw of the whole final chain in its own phase `finalb<k>`, with `BAG_SEED=100*k`: the test members, the denoising autoencoder and the three MLP members are refitted with new seeds, the delayed-row gate's five seeds and the stackers' final fits are offset by `BAG_SEED`, then the chain, the tail and the four post-steps run on them. Fixed across draws: the 2025 out-of-fold side and the stream-transformer columns (`stx_ctx`, `stx_nb`: `stx.py` has no seed offset). Needs stages 3b-3d finished in the final phase and in the five rehearsals | about 75 min of members, about 5 h of stackers; one CUDA GPU, 10 threads; resumable (a member whose file exists is skipped) |
| 5b the bag file | `bash pipeline/46_bag_file.sh [chain id [K]]` (defaults `ar_134naclh_tww`, 3) | the row-wise mean of the K draws' chain files (`scripts/bag_build.py`; it also records each draw's sha256, the pairwise rms differences and the estimated gain), the plan-window post-step (`zf3`, window 3,606 s) and `check_final.py`; gives `ar_134naclh_tww_bag3_wit_zf3`, the shipped file | CPU, minutes |

The result of stage 3d is `outputs/overnight_final/test/ar_134naclh_tww_wit_zf3.parquet`, draw 1 of the shipped file's chain; stages 5a-5b average three such draws (next section). The result of stages 0-5 through 3b is
`outputs/overnight_final/test/ar_132_prc2_wit_zf3.parquet`; stage 3c ends in `ar_133_prc2_wit_zf3.parquet` the same way. All three have two columns (`MVT_ID_mvt`, `TAXITIME_SEC_mvt`) in the order of
`final_submitting.parquet`. A stage can be given on its own: `EC_PHASE=final bash pipeline/50_final_phase.sh members chain file`.
Before the file stage, `python scripts/final_coverage.py --require adsb2m` checks that no feature file or member covers a test month worse than
training; the three known gaps of the OPDI source are allowed by name.

Where the data goes: `data/raw/` the challenge files (`training_2025-*.parquet`, `ranking.parquet`, `submitting.parquet`, `final_ranking.parquet`,
`final_submitting.parquet`), `data/external/<source>/` each open source with its `SOURCE.md`, `data/external/adsblol/points/<day>.parquet` the
ground points, `data/processed*/` the features. Sources and licences: [`docs/data-sources.md`](../docs/data-sources.md).

The code is GPL-3.0-or-later (the repository's `LICENSE`); every `.py` and `.sh` file starts with `# SPDX-License-Identifier: GPL-3.0-or-later`.

### The stream-transformer members of stage 3c

Both are the same network, `overnight/stx.py`: every movement of an airport, departures and arrivals, is a token in time order, and a transformer
over chunks of 512 tokens lets each departure see the clocks, plans and features of the flights around it, with a learned per-head time-decay bias in
attention. Each has an out-of-fold side (stage `base`: the Jan/Jul folds of 2025, which the recorded OOF came from) and a test-side refit per phase
(stage `members`), and each turns its output into feature columns that the month-complete stackers read (the list `M12_EXTRA` of extra feature files, [extending](../docs/extending.md#3-add-a-feature-family-worked-example)).

- **`ts_kdu3`** (switched by `BND_KDU` in `pipeline/39_bnd1.sh`): the transformer with ADS-B modality dropout (a share of the tracked departure tokens is trained with the ADS-B block hidden, so
  the net also predicts well where a flight was not tracked) and self-distillation from an EMA copy that sees the full chunk, also on the label-free months'
  tracked tokens; 3 seeds, 23 epochs for the refit. Its output goes through `overnight/features_stx.py` into the five columns `stxf_reg`, `stxf_reg_gap`,
  `stxf_sig`, `stxf_p1800`, `stxf_p3600` (`features_dep_ts_kdu_f`). Recipe: `recipes/ts_kdu3.json`.
- **`stx_ctx`** (switched by `BND_CTX`): the same transformer whose departure tokens also carry the label-free departure-stream windows of `features_dep_ctx` (without the
  16 columns that hold the base chain's own predictions, `overnight/ctx_member.py`; 14 epochs for the refit). It joins the 11 regressors in the
  member-disagreement features (`features_memb.py --add stx_ctx` gives `features_dep_ef_10111040_stxctx_memb`), which replace `memb2` (the disagreement of the eleven regressors alone) in the stackers' inputs.

`pipeline/38_stx_members.sh` runs one member in one phase (`kdu`, `ctx`, and `oof_kdu`, `oof_ctx` for the 2025 side; `dry` prints the commands);
`pipeline/39_bnd1.sh` is the wrapper that runs both, then the stackers and the tail. Neither member reads a label of the test months.

The ADS-B density cap (`overnight/density_cap.py`, caps written by `scripts/density_cap/caps.py`) is an optional A/B switch of the ADS-B features and the stackers; the final chain runs without it.

## The shipped file

| | |
|---|---|
| file | [`final/`](final/README.md): `ar_134naclh_tww_bag3_wit_zf3.parquet`, the mean of three seed draws of the stage-3d chain (`make final-bag`, stages 5a-5b) after the plan-window step |
| sha256 | `final/SHA256` |
| check | `EC_PHASE=final python scripts/check_final.py final/<file>.parquet --id ar_134naclh_tww_bag3_wit_zf3 --sha256_file final/SHA256`: the template's ids and order, finite non-negative seconds, every step of the recipe chain in `recipes/` (the bag's own recipe first: draws, seed offsets, estimated gain), the hash |

The bag has no cross-validation score of its own (the three draws share one out-of-fold side); `recipes/ar_134naclh_tww_bag3_wit_zf3.json` records the single-draw chain's CV and the variance argument's estimate of the bag's gain (about 0.5 RMSE), an estimate, not a measurement. A rerun gives a file of the same quality, not necessarily the same bytes ([what is not reproduced bit for bit](#what-is-not-reproduced-bit-for-bit)).

## Fetching the ADS-B days

```
make adsb DAYS=2026-01 PLAN=1                 # the plan and the cost, nothing fetched
make adsb DAYS=2026-01                        # a month; also 2026-01-05:2026-01-11 (a range) or 2026-01-05,2026-02 (a mix)
make adsb DAYS=2026-01 VERIFY=1               # re-hash the files on disk against the manifest
```

`scripts/adsb_days.py` wraps the per-day fetch (`scripts/adsb_fetch.py`, which streams a day's adsb.lol archive and keeps only the ground and terminal-area points of the ten airports). A day whose
file is on disk and matches its sha256 in `data/external/adsblol/points.sha256` is skipped; a missing, cut-off or changed file is fetched again, so an interrupted run is restarted with the same command.
The manifest is in `sha256sum -c` format. It guards your files, not the upstream archive (which publishes no checksum). A day the archive does not hold is reported and left out.

Cost per day, from the 96 days the recorded run fetched (one worker; the plan prints it for your days): median 3.5 GB read from the network (streamed, not stored; largest 4.3 GB), 212 MB written
(largest 242 MB), 572 s (9.5 min; largest 1,352 s). A 30-day month is about 105 GB read, 6 GB of disk and 4.8 hours on one worker, about 1.6 hours on the default three. The 124 days kept for the recorded
run take 18 GB.

## Smoke test (CPU, no data, no network)

```
make smoke SMOKE_DAYS=3 [SMOKE_DIR=<work dir>]         # the quick check: 3 synthetic days a month
bash smoke/run_smoke.sh [work dir]                     # the recorded run: 8 days a month (SMOKE_DAYS=8)
```

It copies this tree to a work directory (default `/tmp/prc_smoke`), generates synthetic challenge files, ADS-B points and external data
(`smoke/make_*.py`; the numbers are made up, there is no real signal), and runs every stage above on them: import, features, the 11 members,
the base chain, the five rehearsal phases, the final phase with the month-complete stackers, the post-step and the template check, then scores the
file against the synthetic truth (`smoke/score_smoke.py`: the file must be complete and its RMSE below that of a constant predictor that already
knows the test months' mean, which a misaligned or broken file fails; then it prints `SMOKE OK`). The python on `PATH` must have `requirements.txt` installed. About 2 GB of disk and about 12 GB of RAM.

**Time.** `SMOKE_DAYS` sets how many days a month the generator makes, and the time follows it. The 8-day run took 41 to 44 minutes (below); a 3-day run took 34.7 minutes (2,081 s) with a peak of 7.3 GB of RAM and ended with `SMOKE OK`. The time falls much less than the rows do, because the stages that do not scale with rows (starting 11 members, CatBoost and the five rehearsal phases) are most of it. Your time depends on your CPU. Read the time from your own run.

What the smoke switches (only with its own settings, `smoke/sitecustomize.py` through `PYTHONPATH`): `EC_DEVICE=cpu` moves every GPU model
(XGBoost, CatBoost, PyTorch) to the CPU; `EC_SMOKE_CAP=n` caps boosting rounds at `n` and neural-net epochs at 3. Without these variables the code
runs as recorded (GPU). It tests that every stage runs to its end with the shipped code and pinned libraries and that the files each stage hands to
the next fit together. It does not test accuracy, GPU behaviour, the downloads (the fetch scripts are only checked up to import and `--help`), or the
scale of the real data.

The smoke test runs the stages through 3c (the chain `ar_133`). Stage 3d (the neighbour-label member and the four post-steps of the final chain) is not part of it: those scripts have been run only on the real data, where their output was compared with the recorded twins ([`docs/reproduction.md`](../docs/reproduction.md)). Nor are the bag stages 5a-5b: `pipeline/45_bag_draw.sh` and `46_bag_file.sh` were checked with `bash -n`, `PLAN=1` and `make -n`, and `scripts/bag_build.py` reproduces the shipped bag from the recorded draws; no new draw has been run through them.

**How to read the score line.** The synthetic labels are made up, so the file's RMSE sits near the spread of the labels and says nothing about the model's accuracy. The scorer prints
three reference points so that the line is not read as "worse than a baseline": the constant at the test months' mean (the check: the file must be below it), the per-airport mean of the
synthetic 2025 training months (the baseline a real model has to beat; on made-up data the file lies a little above it, which is what no signal looks like) and the per-airport mean of the
test labels themselves (an oracle no model can use). On the real data the baseline is 660.89 s on the two folds and the model 279.61 s ([results](../docs/results.md)).

Result of the last full run (12-10-2026, Python 3.14.4, 8 days a month, a fresh virtual environment from `requirements.txt`, CPU only, on a 24-thread machine that was
loaded by other jobs at the same time): the whole script took **2,477 s** (about 14 min up to the ar_132 file, about 27 more for stage 3c) and ended with

```
smoke final file: 5,760 departures, RMSE 841.9 s (constant mean 855.9, airport mean 831.1)
SMOKE OK
```

That run printed the older form of the line (831.1 is the oracle, the airport mean of the test labels). The scorer now prints all three reference points; run on the ar_132 file of another smoke
directory (a 24-thread machine, 8 days a month, RMSE 841.4 s) it printed `constant at the test mean 855.9; airport mean of the training months 833.9, ...; airport mean of the test labels 831.1, an oracle`:
the file is 7.5 s above the baseline and 14.5 s below the constant, which is what made-up data without signal looks like.

The same script under Python 3.12.15 (12-10-2026, same environment recipe, same machine) took **2,632 s**, RMSE 841.6 s, `SMOKE OK`. The RMSE is that of the ar_133 file; the ar_132 file of stage 4 gives 842.1 s on 3.14.
An earlier 3.12 run aborted once, after a stage had finished, with `terminate called without an active exception` (a native library thread torn down at interpreter exit); the shim below reports
the script's own exit status when that happens after the script has finished, and the run above did not need it.

`smoke/bin/python` is the smoke test's interpreter shim. With polars 2.0.0 on Python 3.14 a script that has finished its work now and then never exits
(all threads idle while the interpreter shuts down; seen with `rules_chain.py`, `prc_cal2.py` and `final_blend.py`, a few times in the runs on a loaded
machine, and not in 40 runs of a small polars script on the same machine; the last full run needed no kill). The shim starts the interpreter, and a script that is still alive 60 s after it began
shutting down is killed with the exit status it had; a script that finished and is then killed by a signal during shutdown is reported with its own exit status; nothing else about a run changes. Outside the smoke test the same hang can occur: if a stage
script has printed its result and does not return, stop it and run the stage again.

## Comparing two chains

[`overnight/compare.py`](overnight/compare.py) is the paired day-block bootstrap of the RMSE difference between two out-of-fold files, on all, trim and
clean rows. [`overnight/month_weighted.py`](overnight/month_weighted.py) is the write-up's month-weighted comparison over the held-out months of the final
phase and the five rehearsals (per-month change in MSE with a paired day bootstrap, between-month shrinkage, weighted by the test months' shares); its
docstring has the estimator. `recipes/` holds the recorded cross-validation of every step of the final chain.

## What is not reproduced bit for bit

- **CatBoost on a GPU is not bit-reproducible.** Two runs of the same chain differ by about 7 s RMS on the prediction file and by about ±0.1 in
  cross-validation RMSE, starting at the delayed-row gate (`gate2.py`). The gate averages five seeds to dampen this, and the shipped file is that
  chain; a replay gives a file of the same quality, not the same bytes. [`docs/reproduction.md`](../docs/reproduction.md) has the measured replays.
  `EC_DEVICE=cpu` also runs CatBoost on the CPU, whose trees differ from the GPU's: use it for checking that things run, not for matching numbers.
- **The data, feature and member stages have not been replayed from a clean checkout at full scale**; only the chain from the features on was replayed
  against the recorded files, and the whole code path was run end to end on the synthetic data above.
- **External data changes.** adsb.lol is a rolling archive, OPDI and the EUROCONTROL tables are updated by their owners; the
  retrieval dates and hashes are in each source's `SOURCE.md` ([`docs/data-sources.md`](../docs/data-sources.md)). Three OPDI gaps in January and
  February 2026 are allowed by the coverage check.
- **ADS-B costs about 1.7 TB of streaming for the 16 months** (3.5 GB a day, [above](#fetching-the-ads-b-days)) and hours of CPU time; there is no small version that gives the recorded numbers.
- **Named in the code but not in this tree** (each is said so where it is named): the hyper-parameter search that set the stream transformer's width multiplier (`overnight/stx.py --ff`; the values are in
  `recipes/`), the script that drew the fixed hidden-track rows of the `<phase>_hm` analysis option (`overnight/ocommon.py`; no stage of the final chain uses it), and the Kaggle notebooks on which part
  of the ADS-B days were downloaded (`scripts/adsb_fetch.py` carries an output-directory override and no progress hook for them; `make adsb` runs the same extraction locally).
- **Some pieces of the recorded run are not in this tree:** part of the ADS-B days were downloaded through other machines (the script here runs the
  same extraction locally), and the arrival-witness table of the rules chain came from an interactive analysis; `scripts/build_witness_cache.py`
  rebuilds it and, on the real movement table, gives the recorded table's 1,129 rows with identical values.
