# Documentation

One page per topic. Each page opens with what it covers and defines its terms in plain words; [`glossary.md`](glossary.md) has the full list.

## Read in this order

| # | page | what you get | read time |
|---|---|---|---|
| 1 | [`task-and-data.md`](task-and-data.md) | the task, the ten airports, the data, and why the label holds two processes | 8 min |
| 2 | [`model.md`](model.md) | the model stage by stage: copy probability, taxi time, the mixture, rules, stackers, calibration | 30 min |
| 3 | [`features.md`](features.md) | every input family and the operational reason it exists | 5 min |
| 4 | [`validation.md`](validation.md) | how a change is judged: folds on the test months' calendar months, paired bootstrap, rehearsal | 8 min |
| 5 | [`results.md`](results.md) | numbers by stage, airport and month, and what did not help | 18 min |
| 6 | [`error-budget.md`](error-budget.md) | where the remaining squared error sits, row kind by row kind | 4 min |
| 7 | [`limits.md`](limits.md) | where the model is weak and what would break it | 6 min |

## Use it

| I want to | page |
|---|---|
| rebuild the prediction file, check its hash, run the smoke test | [`reproduction.md`](reproduction.md), [`../code/README.md`](../code/README.md) |
| run it on new months, drop an input, add a feature, a member or an airport, reuse it in OPDI | [`extending.md`](extending.md) |
| check where every input comes from and under what terms | [`data-sources.md`](data-sources.md), [`daily-delay-file.md`](daily-delay-file.md), [`credits.md`](credits.md) |
| look up a term or an airport code | [`glossary.md`](glossary.md) |

The same story as one illustrated page: [`../writeup/index.html`](../writeup/index.html) (offline, open it in a browser) or [`../writeup/README.md`](../writeup/README.md). As a paper: [`../paper/main.pdf`](../paper/main.pdf).
