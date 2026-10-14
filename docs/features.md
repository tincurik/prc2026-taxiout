# Features and the operational reasons for them

No row's own label is an input: no departure's `BLOCK` or taxi time feeds its own features. Three kinds of feature are model outputs trained
on labels, the month-complete ADS-B model, the neighbours' predictions and the stream-transformer columns; all are out-of-month (a month's values come from models
that did not see it). Statistics that use labels (the delayed-row gate, the LIRF regime rates, the tail steps) are fitted on
training months only, or month-out-of-fold.

| family | what it holds | operational reason |
|---|---|---|
| **Clocks** | take-off minus `AOBT_3`, `LOBT`, `EOBT`, `IOBT`, `SCHED`; gaps between clocks; flight times from the plan | NM's `AOBT_3` is an independent off-block estimate, so `MVT − AOBT_3` is the single strongest predictor. Gaps between clocks measure re-planning, and copies happen where nobody re-planned |
| **Recording process** | whether an NM plan matched, callsign or destination mismatch between the airport and NM records, airline, stand prefix | who records the off-block decides whether it is copied: copy rates run from 4 % to 55 % between LIRF's large carriers |
| **Arrival-side copy habits** | share of arrivals of the same airline (±6, ±24, ±72 h) or of the airport (±1, ±3, ±12 h) whose in-block equals their schedule; the previous arrival at the stand: gap, delay, taxi-in, copied or not | arrivals are not masked, so a handler's habit of copying the schedule is visible on its arrivals. Departures are copies in 13.2 % of cases when the stand's previous arrival was one, against 5.2 % otherwise |
| **Surface load** | departures in the 5–60 min before and after take-off; queue at take-off; departures and arrivals during the taxi; the same per runway; other flights' `MVT − AOBT_3` nearby | taxi time grows with the number of aircraft competing for taxiways and the runway |
| **Stand and runway history** | arrivals' taxi-in at the same stand (±3, ±30 days); other departures' `MVT − AOBT_3` at the same stand and stand × runway; local-midnight crossing; scheduled local hour | stand-to-runway distance and routing are stable, so neighbours' taxi times measure them without a map |
| **Geometry** (OpenStreetMap) | stand position, taxi-network distance from stand to runway threshold, straight-line distance | the physical length of the taxi |
| **Weather** (METAR, 102 columns) | at schedule time, at take-off and per day: snow, freezing, de-icing conditions, thunderstorms, low visibility, wind, temperature | de-icing after off-block (EDDF, EDDM, LFPG, LSZH) and low-visibility procedures lengthen taxi; storms stop departures |
| **Network and ATFM** (EUROCONTROL) | airport-day traffic, arrival ATFM delay by cause, regulated share and slot adherence, strike days | a congested or regulated day changes stand and runway sequencing |
| **Record ordering and backlog** | a record's id minus the median id of the ±250 departures nearest in take-off order; flights past their `EOBT` but not yet off-block; route-centred NM flight-time deviations; days of runway works at Paris-CDG (LFPG) | the NM flight id follows the planned off-block, so an id that is low for its take-off slot marks a flight that left late against its plan; a backlog of flights waiting to push predicts the own wait |
| **Own-flight surface events** (OPDI) | take-off minus first taxiway entry, last parking exit, last runway entry (11–13 % of flights) | a direct, independent observation of the own taxi where available |
| **ADS-B ground track** | the own aircraft found by callsign within ±90 min of take-off; take-off minus first heard, first movement, last point within 30 m of the first fix, roll start, end of the last stop; stops, stopped seconds, path length; stand coordinates learned from parked positions | the track shows when the aircraft actually moved. Most aircraft are first heard *after* off-block (median +225 s, transponders are off at the stand) |
| **Pushback** | reverse-motion steps: consecutive fixes under 60 s and 2–150 m apart whose bearing differs from the reported heading by more than 135° | pushback is tail-first; detecting it marks the start of the movement even when ground speed is noisy |
| **Month-complete ADS-B model** | a taxi-time prediction from ADS-B and clocks trained on all twelve months (each month predicted from the other eleven); receiver descriptors per airport-day | the stackers only see two validation months; this carries what the other ten months know about each airport's tracks |
| **Member disagreement** | each regressor's gap to the blend, their weighted spread and range | the blend gives each member one global weight; which member is right depends on the row (ADS-B members where a track exists, plan-anchored members where a plan exists) |
| **Departure-stream context** | for each flight, label-free aggregates of the airport's other departures around it: mean delay against schedule and plan, overdue backlog, throughput against the airport-hour norm, arrivals' delay, neighbours' predictions | a flight's taxi shares a component with the departures around it (runway queue, de-icing waves, stand congestion); the neighbours' delay is the earliest visible trace of that queue |
| **Stream-transformer columns** | the outputs of a sequence model that reads every movement of the airport, departures and arrivals, as a token in time order: its prediction of the residual, its gap to the chain, its spread, its chance of a label above 30 and 60 min, and a denoised level of the neighbours' labels on the runway and the airport | the windows above are hand-made; a sequence model reads the neighbours directly, and its neighbour level is the queue the flight shares with the departures around it ([model](model.md#residual-stackers)) |

## Details worth knowing

- **Why stands and runways stay although they drift.** Stand and runway identities separate 2025 from 2026 a little (12 % and 9 %
  of the drift classifier's gain). They enter as histories: taxi times at the same stand or runway in nearby days, recomputed in
  the test months from those months' own records. A renamed stand starts a new history instead of carrying a stale one. Flight
  numbers and callsigns would enter as identities, so they are left out.
- **ADS-B matching.** A departure is *matched* when an ADS-B take-off lies within 300 s of its recorded take-off. Coverage
  differs enormously by airport and month: see [limits](limits.md).
- **Why the arrivals matter.** Both the copy habits and the stand history are read from unmasked arrival records. This is only
  possible because the challenge is post-operations; a real-time system could not use the "after" half of those windows.
