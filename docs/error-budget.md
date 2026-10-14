# Where the remaining error is

The model's squared error, split by what kind of row carries it. The pool is four held-out months of 2025 (January and July out of fold from the validation folds,
February and June from the rehearsal), 671,292 departures, scored with the final model's chain, whose stackers read the stream-transformer inputs
([model](model.md#residual-stackers)). The January and July rows come from the rebuilt final chain, the February and June rows from its [validation twin](glossary.md) (the rebuild has no February or June predictions). Its RMSE on this pool is 260.01 s. The first three kinds are defined from the labels alone, so they are the same rows for any chain; the large misses depend on the chain's errors.

## Five kinds of row

| kind of row | rows | share of squared error | RMSE (s) |
|---|---:|---:|---:|
| Rome date stamps (label about one day plus the taxi, [glossary](glossary.md)) | 8 | 28.1 % | 39,916 |
| Giant labels (over two hours, not a copy, nothing in the inputs hints at them) | 82 | 10.9 % | 7,769 |
| Far schedule copies (copies whose label is more than an hour) | 651 | 7.8 % | 2,332 |
| Large misses on ordinary rows (error above 1,800 s) | 599 | 11.2 % | 2,910 |
| Ordinary rows | 669,952 | 42.0 % | 169 |

The first three kinds are **artefacts**: 741 rows hold 46.8 % of the squared error. The inputs do not say which flight is one, and a
model that is right about every physical taxi still loses that half. Without them the RMSE is 189.77 s instead of 260.01 s.
Of the 599 large misses on ordinary rows, Rome holds 6.1 % of the total (flights where the model hedged towards a copy that did not
happen) and Istanbul 1.7 % (taxis longer than predicted).

## The reachable part: a queue level shared by the flights around it

The ordinary rows are 99.8 % of the departures and 42.0 % of the error, and the error by label size is spread, not a shift: the
probability and taxi calibrations hold within each size bin. What remains is that a few dozen
flights taking off within the same half hour at the same airport are all slow or all fast together. The runway has a queue; the
queue's length is the same for each of them, and no row carries it as a feature.

Where does it sit? In the airports whose aircraft are not heard by open ADS-B receivers while they taxi, so no track says the queue is long: Istanbul
(LTFM), Paris Charles de Gaulle (LFPG), Heathrow (EGLL) and Rome (LIRF). Their ordinary rows without a track hold 24.3 % of all
squared error, more than every other airport and track combination together. The complete table is
[`error_bulk_cells.csv`](../writeup/data/error_bulk_cells.csv).

How large is the shared level? An oracle (not a model) adds to each departure the mean error of the other departures at the same airport within a window of ±15, ±30
or ±60 minutes, shrunk towards zero for sparse windows. It takes the RMSE down by 2.97, 2.90 and 2.49 s
([`level_oracle.csv`](../writeup/data/level_oracle.csv)). It is a ceiling, not an expectation: it reads the other flights' labels,
which the model cannot. It says the lever is real but small, about one percent of the RMSE, and that the right way to capture it is
a representation that lets each departure see its neighbours' behaviour, as the stream transformer does
([the stream](model.md#residual-stackers)), not one more per-row feature.

## What this means for the design

- The mixture and the record tail go after the artefact half where a signal exists (copy probabilities, plan windows, the Rome
  regimes); the rest is out of reach of any input we have.
- The month-complete stackers and the stream-transformer inputs go after the ordinary half. The departure-stream aggregates already gave each
  departure its neighbours' clocks; the stream transformer lets it read their plans, tracks and features as well, in time order.
- The ordinary-row ceiling from the oracle is about 3 s of RMSE, so further modelling of the physical taxi buys little next to the
  rows nobody can see.
