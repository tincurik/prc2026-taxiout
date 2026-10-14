# Glossary

**One word per concept.** The repository uses these words and no others for them:

| we write | for | never |
|---|---|---|
| **pipeline** | the code that turns the challenge files into the prediction file, in stages 0 data … 5 prediction file, with the seed-bag stages 5a–5b ([reproduction](reproduction.md#stages)) | |
| **chain** | the model's prediction steps in order: blend, rules, stackers, record tail, post-steps | stack, cascade |
| **step** | one box of the chain, a rule, a stacker or a calibration | module, layer |
| **member** | one of the eleven regressors of the blend; the stream transformers are further members, whose outputs feed the stackers | base model |
| **rehearsal** | the whole pipeline run on a held-out pair of 2025 months masked like the test months: February + June (reported in [validation](validation.md#rehearsal-on-held-out-february-and-june-2025)), March + September, April + October, May + November, August + December | pseudo-final, pseudo-phase |
| **phase** | the run a rehearsal or the test months belong to, named in commands (`EC_PHASE`) | |
| **test months** | January, February, June and July 2026, the months to predict | final months |
| **final model** | the model whose predictions we deliver; its chain is the **final chain** | champion |
| **reference chain** | the final model without its last additions (context features, month-complete stackers, stream-transformer inputs, four post-steps, plan-window post-step) and with a LIRF plan-row classifier the final model drops (CV 285.44); the per-step, per-airport and rehearsal tables use it because every step of it has been measured on its own | |
| **copy** | a departure whose recorded `BLOCK` is its schedule (`\|BLOCK − SCHED\| ≤ 30 s`), not an event (in full: *schedule copy*) | SCHED copy |
| **physical taxi** | the part of the label that is the aircraft moving | regression part |


| term | meaning |
|---|---|
| **A-CDM** | Airport Collaborative Decision Making: airlines, handler, airport and NM share target times so that a flight waits for its ATFM slot at the stand, not on the taxiway |
| **AOBT / ATOT / SOBT** | actual off-block time, actual take-off time and scheduled off-block time (the standard A-CDM names). In the data files the airport's `BLOCK` is its AOBT, `MVT` is the ATOT and `SCHED` is the SOBT; `AOBT_3` is the Network Manager's separate estimate of the off-block time, a different quantity that this repository always writes `AOBT_3`, never bare "AOBT"; the member table's anchor column uses it. Units: every clock is UTC seconds and every duration is seconds; feature names do not carry the unit (`a2_gs1` is a time in seconds), so the unit is given in [features](features.md) and in the code's docstrings |
| **ADS-B** | the broadcast of an aircraft's position; open receivers collect it and publish history (adsb.lol) |
| **airport codes** | EDDF Frankfurt, EDDM Munich, EGLL London Heathrow, EHAM Amsterdam, LEBL Barcelona, LEMD Madrid, LFPG Paris CDG, LIRF Rome Fiumicino, LSZH Zurich, LTFM Istanbul |
| **anchor** | the clock that a regressor's target is measured from: `y − (MVT − AOBT_3)` or `y − (MVT − LOBT)` |
| **AOBT_3** | NM's own estimate of the actual off-block time, independent of the airport's `BLOCK` |
| **ATFM** | Air Traffic Flow Management: slot regulations that hold a flight at the stand to fit en-route capacity |
| **blend** | the weighted average of the eleven regressors, with weights from non-negative least squares |
| **BLOCK** | the airport's recorded off-block time: the quantity subtracted from take-off to make the label |
| **buffer month** | a month left out of training next to a validation month, so neighbouring days cannot leak |
| **clean / trim rows** | rows with label ≤ 7,200 s / ≤ 20,000 s, used to report errors with and without the heavy tail |
| **conditional mean** | the squared-error-optimal prediction given the inputs; it shrinks long, unpredictable taxis |
| **CV** | cross-validated score: RMSE on the January and July 2025 folds, from models that did not train on those months or their neighbours; three parts are looser ([validation](validation.md#where-the-numbers-are-optimistic)) |
| **d_sched, d_aobt, d_lobt …** | take-off minus that clock (`MVT − SCHED`, `MVT − AOBT_3`, …); each is a candidate taxi time |
| **draw** | one full run of the final chain with one set of random seeds; the seed bag averages three |
| **DAE** | denoising autoencoder, trained with swap noise on the input rows; here a regressor head sits on its encoder |
| **EOBT / IOBT** | the estimated / initial off-block time of the flight plan |
| **far schedule copies** | copies whose label is more than an hour ([error budget](error-budget.md)) |
| **far-LIRF record mix** | the post-step for LIRF rows with no NM off-block and a plan copy above 50,000 s ([model](model.md#four-post-steps-on-the-chain-output)) |
| **fold** | one of the two validation months (Jan, Jul 2025) with its training and buffer months |
| **giant labels** | labels over two hours that are not copies and that nothing in the inputs hints at ([error budget](error-budget.md)) |
| **LIRF datestamp** | a Rome record whose off-block is the schedule's local date plus the true off-block time of day, so the label is about one day plus the taxi |
| **LOBT** | the last filed off-block time of the flight plan in NM |
| **matched (ADS-B)** | a departure with an ADS-B take-off within 300 s of its recorded take-off |
| **mixture** | `pred = p · (d_sched + δ) + (1 − p) · regressor`: the copy and ordinary branches weighted by the copy probability |
| **month-complete stacker** | a stacker trained on predictions for every month of 2025, each made by a model that had not seen that month |
| **MSE, SSE** | mean and summed squared error; RMSE is the square root of the MSE |
| **MVT** | the recorded take-off (movement) time |
| **NM** | EUROCONTROL's Network Manager, which holds the flight plans |
| **NNLS** | non-negative least squares, used to fit the blend weights |
| **OPDI** | the Open Performance Data Initiative of EUROCONTROL; its flight events give some aircraft's surface events |
| **out-of-fold (OOF)** | a prediction made by a model that did not train on that row's fold; the exceptions are listed in [validation](validation.md#where-the-numbers-are-optimistic) |
| **paired day-block bootstrap** | compare two runs on the same rows, resampling whole days 1,000 times; gives the 95 % interval of the difference |
| **plan cliff** | a flight re-planned more than 1 h after schedule never copies, so its copy probability is set to 0 |
| **plan window** | a flight plan is attached to a departure record only when `\|BLOCK − LOBT\| ≤ 3,606 s`, which bounds the label |
| **PRC** | the EUROCONTROL Performance Review Commission, which publishes the data, and the data portal it runs |
| **record tail** | small steps for specific record faults applied after the stackers ([model](model.md#record-tail)) |
| **residual** | label minus prediction: what a later stage still has to explain |
| **RMSE** | root mean squared error in seconds, the challenge score; lower is better |
| **SCHED** | the scheduled off-block time of the airport record |
| **stacker** | a second-level model that predicts the first level's residual from ADS-B and clock inputs |
| **seed bag, K=3** | the shipped file is the mean of K = 3 draws of the whole final chain, each refitted with other random seeds (`BAG_SEED` 0, 200, 300); averaging cancels the draws' independent noise (estimated gain about 0.5 RMSE; no cross-validation score of its own) ([reproduction](reproduction.md#the-seed-bag)) |
| **witness step, `zf3`** | the plan-window post-step that makes the file: the `zf3` rules of `scripts/witness_post.py` (window 3,606 s) clip planned departures to the plan window and lift LIRF rows whose arrival record shows a plan ([model](model.md#plan-window-post-step)); plan-window step, witness step and `zf3` are one thing |
| **bundle chain** | the chain of stage 3c (`ar_133`, extended by the neighbour-label member into `ar_134na_dup`): stackers that read the stream-transformer members, before the four post-steps ([reproduction](reproduction.md#names-used-in-this-repository)) |
| **validation twin** | the run of the final chain's steps that was judged in the held-out phases, so its predictions for a held-out month never saw that month. The shipped chain is a rebuild of it in the final phase; the two differ by draw noise (rms 33 s on the test months). Tables marked "twin" or "validation run" are of it |
| **thorough tier** | the full test of a change: all twelve held-out months (the final phase and the five rehearsals) with a paired day-block bootstrap; the *fast tier* uses the final phase and one rehearsal only |
| **honest (prediction)** | made by a model that did not see the month it predicts (out of fold or from a rehearsal) |
| **post-step** | one of the four small steps applied to the finished chain (LIRF copy re-mix, runway-neighbour level, far-LIRF record mix, arrival-twin window rule), plus the plan-window step that makes the file |
| **stage** | one script of the pipeline (`code/pipeline/`): data, features, members, chain, test months, prediction file |
| **stream transformer** | a network that reads every movement of an airport, departures and arrivals, as a token in time order; its outputs are inputs of the stackers ([model](model.md#residual-stackers)) |
| **ΔRMSE [95 % CI]** | the change in RMSE a step makes, with the paired day-block bootstrap interval; negative is better |
| **ΔMSE** | the same in mean squared error (s²) |
| **twelve months** | the score over all twelve months of 2025, each predicted by a model that had not seen it, weighted like the test months ([validation](validation.md#twelve-months-weighted-like-the-test-set)) |
| **taxi-out** | the time between off-block and take-off |
