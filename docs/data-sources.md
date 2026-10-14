# External data: sources and licences

This page lists every input besides the challenge files, with its terms. The repository stores no external data file and no
challenge file.

| source | used for | terms |
|---|---|---|
| adsb.lol global ADS-B history | ground tracks of departures: match, first heard, taxi map, motion | Open Database Licence (ODbL), as stated by adsb.lol; attribution kept here |
| METAR / SPECI via the Iowa Environmental Mesonet | weather at the ten airports | freely available without an account; the METARs are public weather reports |
| EUROCONTROL Performance Review data (ansperformance.eu, the portal that also hosts this challenge) | airport traffic, ATFM delay causes, daily pre-departure delay; the monthly taxi-out per runway is read only by a calibration step that was tested and is not in the final chain | may be copied in whole or in part provided EUROCONTROL is mentioned as source, not for commercial purposes, not modified without permission; no named open licence. Cited as EUROCONTROL Performance Review Unit. Its use is set out in [model](model.md#calibration-to-published-aggregates); the daily delay file has [its own page](daily-delay-file.md) with its URL, terms, cost and the organiser messages |
| EUROCONTROL Performance Review day statistics, feature family `ext_prc` | 17 day-level columns per airport and UTC day: departures and arrivals (`airport_traffic`), arrival ATFM delay by cause and the share of arrivals delayed over 15 minutes (`apt_dly`), the share of regulated departures late for their slot (`atfm_slot_adherence`), and a strike-day flag; `scripts/fetch_external.py` downloads the daily CSVs of 2024-2026. Read by members 1 and 3–8 of the eleven regressors ([model](model.md#the-mixture)) and by the stream-transformer members; not by the copy classifier | the portal and EUROCONTROL terms of the row above: copying allowed with EUROCONTROL named as source, not for commercial purposes, not modified without permission; no named open licence. Covered by the organiser's 13-10 answer that data EUROCONTROL publishes may be used ([statements](#organiser-statements-on-the-inputs)). Each flight gets the values of its own take-off day at its own airport, published after the day. They are traffic, arrival-delay and slot statistics, not the pre-departure delay ([daily delay file](daily-delay-file.md)) or any flight's taxi-out. Not ablated: no run removes them, so their effect on the score is not known. To remove them, delete `ext_prc` from the `EXT`/`--extra` lists of stages 20, 38, 45 and 50 and refit the members |
| OPDI v0.0.2 flight events (EUROCONTROL PRC and OpenSky) | the aircraft's own surface events, 11–13 % of flights | described as freely usable, reusable and distributable provided the source is attributed; no named licence. Derived from OpenSky data; the only OpenSky-derived input. Removing it from every member costs 0.51 CV RMSE |
| OpenStreetMap via Overpass | runway, taxiway, apron and stand geometry | ODbL 1.0, © OpenStreetMap contributors |
| OurAirports | runway threshold coordinates (fallback to OSM) | public domain |

Retrieval dates of the recorded build (UTC), so a rebuild can tell a revised source from a bug:

| source | retrieved | fixed by |
|---|---|---|
| adsb.lol | per day, as needed | each day is a fixed GitHub release, `v<YYYY.MM.DD>-planes-readsb-prod-0` of `adsblol/globe_history_<year>` |
| METAR / SPECI (IEM) | 15-09-2026 | the archive query per station and year in `fetch_external.py` |
| OpenStreetMap (Overpass) | 15-09-2026 | the snapshot of that date (`timestamp_osm_base` 2026-09-15 in each file); Overpass serves the live map, so a later fetch can differ |
| OurAirports | 15-09-2026 | none: the files are updated in place |
| OPDI v0.0.2 | 15-09-2026; the February and June 2026 files 10-10-2026 | the version in the URL |
| EUROCONTROL day statistics (`ext_prc`) | 15-09-2026; the 2026 files fetched again 28-09 and 10-10-2026 | none: the yearly files grow and can be revised |
| EUROCONTROL daily delay and taxi-out files | see [the hashes below](#challenge-data) | sha256 below |

`fetch_external.py` writes each file's URL, retrieval time and sha256 to a `files.jsonl` beside it at fetch time.

The GPLv3 licence covers the code and documentation only. No external data is redistributed here: whoever runs the pipeline
downloads each source and is bound by its own terms, including the non-commercial ones above.

An earlier version also read airplanes.live ADS-B (no open licence); the shipped chain does not; it was worth 0.3 RMSE.

## Organiser statements on the inputs

The organisers answered data questions in 2026, mostly on the challenge's Discord channel. Below are the statements this repository
relies on, paraphrased and dated; the answers are not quoted.

- External open data: allowed if the documentation declares it (08-09). This page is that declaration.
- Open trajectory data, including a departure's own ground positions after pushback: open trajectory data with good ground
  coverage from which off-block times can be extracted is acceptable (19-09); open data sources may be used to devise a better
  model (26-09); asked about a flight's own post-pushback ADS-B track as a feature, the organiser answered that the open-data rule
  applies and anything that helps may be used (01-10). This covers the adsb.lol ground tracks.
- Post-operational model: the organisers described the model as for post-operational, not tactical, use (19-09). We read this
  as allowing data published after the day of operation; the daily calibration step relies on that reading, and the 13-10
  answer below covers its file.
- Published monthly taxi-out for the test months: asked whether the published monthly average taxi-out per airport, and per
  runway on the airport dashboard, for January and July 2026 may be a model input, the organiser said yes (27-09). Asked again for
  the final, with the remark that the file carries EUROCONTROL's terms and no named open licence, the organiser said nothing had
  changed and the step was still fine (08-10). That file fed the monthly runway calibration, which was tested and is not in the
  final chain; the shipped file reads nothing from it ([model](model.md#calibration-to-published-aggregates)).
- OPDI: asked whether OPDI counts as open although its data files name no licence, the organiser confirmed it (04-10).
- Unlicensed data in general: answering a participant's general question about data from a public website with no licence, the
  organiser said that without an open licence the data is not open (01-10). The answer names no file. The 08-10 confirmation
  above came a week later, to a question that cited it.
- EUROCONTROL daily airport files: asked about them and their non-commercial clause, the organiser answered that they can be
  considered open data for the challenge, provided their licence applies as open data (17-09).
- EUROCONTROL's published daily pre-departure delay: asked whether it may be used on the test days, the organiser answered that
  data EUROCONTROL publishes may be used (13-10). This covers the daily delay file and the day statistics of `ext_prc`.

The daily pre-departure delay file, the input of the daily delay calibration, has [its own page](daily-delay-file.md). The 13-10
answer above covers it.

## Challenge data

The 2025 training files and the 2026 test files are not redistributed. Stage 0 fetches every source at run time. The files
behind the results were fetched between 15-09 and 10-10-2026 (dates above); published series can be revised later, so a rebuild may read slightly different
values. The table lists the files the two calibration steps read in the recorded build, so a rebuild can check that it reads
the same values (`sha256sum data/external/prc*/<file>`):

| file (in `data/external/`) | from | retrieved (UTC) | sha256 |
|---|---|---|---|
| `prc/all_pre_departure_delays_2025.csv` | eurocontrol.int performance data download | 28-09-2026 | `d742eeb626e5f9f7e0c756e2ae3cf90078f4627e989a2e31b8ed21ba8f5510be` |
| `prc/all_pre_departure_delays_2026.csv` | same | 28-09-2026 | `5952d20ef5f9c4245018dd6986064c5c4aa15d568a6c0c2dcdcbc73bf8a43fd2` |
| `prc/taxi_out_additional_time_2025.csv` | same | 15-09-2026 | `4aedf19520d3f84bebc418b56835762c907b714a9c19e751bb8658f9e62ce598` |
| `prc/taxi_out_additional_time_2026.csv` | same | 10-10-2026 | `b72040e0987e6c7d5706279d645ef9f814a6506165c83f89d8509022154a1076` |
| `prc_dashboard/runway_tot_ref_time_tot_add_time_tot_flt_type_avg_txot_time.csv` | ansperformance.eu airport dashboards, extracted by [`code/scripts/fetch_prc_dashboard.py`](../code/scripts/fetch_prc_dashboard.py) | 10-10-2026 | `c144b493e3e3b85df4859227e2ddf82ad508fadf099ab282fe2a045859c4d317` |
