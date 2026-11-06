# Operator console

How the console in `console/` presents a shift. HTTP contract: `console/openapi.yaml`. The globe is a Kepler picture, not charge physics and not job assignment.

Update this file after a significant UI change.

## Layout

A full-screen dark console in three columns:

| Zone | What the operator sees |
|---|---|
| Top | scenario id, shift length, goal (Priorities / Revenue), Run / Pause, Step, More |
| Left | satellite list: charge, temperature, calibration age, action |
| Center | 3D globe and timeline |
| Right | inspector: satellite / jobs / events |
| Bottom | shift summary; a click opens three more numbers |

A shift step is 5 minutes. Timeline clocks: `step × 5` min (`00:00` on step 0). The shift cursor is `state.step`; hovering the timeline sets `previewStep` and does not move the shift.

## Color

One palette for the tree, globe, and timeline. Tokens live in `console/static/css/console.css` (`:root`).

### Satellite action (policy signal)

| Color | Token | Where | Meaning |
|---|---|---|---|
| White `#e8ebef` | `--sig-idle` (`--text`) | globe dot, tree status | idle — the same color as waiting jobs |
| Green `#3ddc6a` | `--sig-job` | same | executing a job |
| Violet `#8b7fd4` | `--sig-cal` (`--accent`) | same | calibrate — the same color as jobs in progress |
| Grey `#8a8a8a` | `--unavailable` | tree | satellite unavailable |
| Faded (opacity 0.38) | `.is-faded` | satellite / job row | cannot take or assign now (`can_execute` is false). An overdue job that was **never** in access is faded too |

Globe legend: “green — job, white — idle, violet — calibrate · faded — cannot take now”. A faded dot is the action color mixed with `#3a404c`. Idle with `take=1` stays bright: the policy chose to wait.

### Health (charge and temperature)

| Color | Token | Threshold | Where |
|---|---|---|---|
| Green `#5cad5c` | `--ok` | SOC ≥ reserve | charge bar; the dot if there is no threat |
| Orange `#f0a040` | `--warn` | reserve > SOC ≥ critical | charge bar and the dot — charge warning |
| Red `#e05555` | `--danger` | SOC < critical **or** \|T\| ≥ 130 °C | the dot; the charge bar only on discharge; the temperature number on overheat; overdue / infeasible jobs |

Thresholds `reserve_soc_pct` / `critical_soc_pct` come from the shift JSON (`model`). The console temperature limit is ±130 °C, matching the chart dashes. The dot is red if the satellite is discharged **or** overheated; orange is a charge warning only. The SOC bar is not painted red because of temperature.

### Other

| Color | Use |
|---|---|
| Lilac `#8b7fd4` (`--accent`) | selected satellite, Run button, “now” mark on a chart, job in progress, satellite calibrate |
| Lilac curve `#c4bdf5` | policy quality `q` on the timeline |
| White vertical | current shift step on the timeline |
| White ball with a lilac ring | hovered step |
| `--warn` dash on the charge chart | reserve line |
| `--danger` dash on the charts | critical SOC / ±130 °C |
| Green `--ok` / orange `--warn` | timeline marks ok / warn |

Job statuses in the inspector: waiting — white text, active — accent, done — ok, overdue and infeasible — danger.

## Loading a shift

Start is a JSON drop screen (initial parameters, including each satellite’s orbit). Preview: file name, size, satellite count, steps, planned job count. “Start shift” posts the JSON to `/api/scenario`, then fetches orbits and a policy run in parallel. While that loads — an overlay “Recomputing policy” (the same spinner as for events).

The demo button reads `console/src/mocks/P01_intro.json` (a copy of `example/scenarios/P01_intro.json`). Any initial JSON from `example/scenarios/` can be dropped.

## Top bar

- **Orbi.tar** — wordmark on the left, then the **scenario id** and “N steps (H h)”.
- **Priorities / Revenue** — OpsEnv reward goal and the checkpoint from step k (`priority` → `ckpt_68120`, `revenue` → `ckpt_35714`). Switching asks for confirmation. The modal scores a **+10 step** fork: the live run prefix is shared, from k another network (`POST /api/dispatch/whatif`). Six rows match the bottom summary: stay at k+10 → switch at k+10, green ↗ / red ↘. Until a frame arrives — “computing…”. “Switch” replays the tail with the new network (`POST /api/dispatch/resume`); “Cancel” leaves the run. Computing can start before frame k arrives — then “Switch” stays disabled.
- **Run** — autoplay at 450 ms per step; **Pause** stops it.
- **Step** — one tick. If the current step has a draft event, it is committed and the tail is recomputed by the network (see Events).
- **More**: jump to step N, reload the scenario, export a command journal (`/api/result`, 120 MB limit).

Errors (no journal, file too large, recompute failure) use a “Message” modal.

## Satellite tree

Row: health dot · id · bar+SOC% · temperature °C · steps since calibration · action. Action: job in green, idle in white, calibrate in violet. The charge bar is orange on a warning and red on discharge. Temperature is red when \|T\| ≥ 130 °C. The dot is red on discharge or overheat.

Sort: jobs first, then calibrate, idle, unavailable; then by id. Satellites that **cannot take anything now** (`take=0` in the frame) are faded and always last. Row hint: “Cannot assign a job now”.

Idle with `take=1` is not faded — the network could have taken work and did not.

Filters: All / Issues (not ok) / Critical. Search by id, job_id, and the action label.

A click selects the satellite: highlight in the tree, card on the right, globe HUD on the dot.

Frame data is `orbi` physics through the `/api/dispatch` stream (energy, temp, calibration_age, action after repair).

## 3D globe

A Kepler picture (`/api/tracks`, fallback slice `/api/orbits`). Not `solar_w` and not the contact windows from the JSON.

- Earth texture, stars, orbit lines (translucent white), satellite dots.
- The sun is a directional light from the same Kepler world’s ECEF.
- Dot color = action on the **viewed** step (`action_tape`, else the current action): green job, white idle, violet calibrate. If the satellite is idle and `assignable=false`, the dot fades.
- HUD: step, selected id, legend “green — job, white — idle, violet — calibrate”.
- Drag rotates the camera; a short click on a dot selects the satellite.
- Hovering the timeline moves the dots and the sun to `previewStep` without changing `state.step`.

The globe does not assign jobs.

## Timeline

Under the globe. The curve is **policy quality `q` from the frame**, not a client heuristic over metrics.

`q = 0.5 · sigmoid(value / remaining_job_usd) + 0.5 · heuristic`, where `value` is the critic of the same network (`ORBI_MODEL`, currently `ckpt_68120`), and the heuristic is SOC ≥ reserve, P3 still on time, a non-idle share, and no new overdue jobs. The fill follows the current shift step; the line covers the full length as frames arrive.

Header: step count · “Event marks N” · **“Current policy advantage: N%”** (`round(q × 100)` on the current step, else a dash). On the right — the viewed step’s clock.

Marks are the first threshold crossing in already arrived frames, catalog `console/src/timeline-marks.json`:

| id | Tone | When |
|---|---|---|
| `revenue_half` | ok (green) | revenue ≥ 50% of the original list potential |
| `overdue` | warn (orange) | overdue ≥ 1 |
| `soc_reserve` | warn | at least one satellite below reserve |
| `seed_done` | ok | ≥ 80% of the original jobs closed |

A click on a mark opens its card on the Events tab and sets `previewStep`. A click on the track is hover only. The white stick is the current step; the ball is the hovered step.

Caption: “Current step N · You hovered step M”.

## Inspector: satellite

JSON parameters (capacity, power) plus live frame state (charge %, temperature, calibration age, action).

Two charts over the full shift, series up to the last arrived frame:

1. **Charge, %** — 0…100, reserve and critical dashes, a “now” vertical, caption “current · reserve · critical”.
2. **Temperature, °C** — −150…150, ±130 °C dashes.

Below: the action tape by step; rejected commands (empty after repair in the normal case); a text status (“idle”, “executing a job”, “no job that can be taken”). Field “Can take a job”: yes / no from `Environment.can_execute` on the current step (not the top-K mask).

## Inspector: jobs

List: id, priority, status (waiting / busy / done / overdue / infeasible). Filters: all, done, overdue, highest priority. Search.

Order (all filters): active → waiting that **can** be assigned → done → overdue that **were** in access → overdue that **never** opened → infeasible → waiting that cannot be assigned now (the tail). Overdue jobs with no access stay faded in the overdue group; they do not drop to the end of the whole list.

Faded rows (opacity 0.38): waiting/infeasible that cannot be assigned now; overdue that was never in `open`.

Card: kind (relay / downlink), step window, executors, revenue, progress `done/needed · who led`, status, **“Was in access”** — “on step N” (first frame with `open`) or “Never in access”. Block “Current-state analysis” is a human-readable x-ray.

Access source: `/api/dispatch` frames, fields `open` (job indices) and `take` (0/1 per satellite). `first_open_step` is computed on the client from arrived frames 0…view.

Jobs added with `add_jobs` appear in the list after the tail is recomputed (stream meta).

## Inspector: events

A draft exists only on the current shift step (`eventComposeAt === step`). “+”: a new urgent job, a satellite outage, a cancelled pass; or `cosmo-B-events-1.0` JSON. Until “Step” is pressed the status is “draft” and it can be edited or cancelled.

Step commits the event and calls `/api/dispatch/resume`: ticks before k keep the old commands, from k a PPO sample, `apply_event` before the action. An overlay “Recomputing policy” stays until the new step’s frame is ready. The left curve does not roll back; the `q` tail and every shift field (satellites, jobs, metrics) come from the new stream.

A click on a timeline mark shows its card here (green or orange frame) even if there are no operator messages yet.

## Bottom summary

A centered row, delta to the left of each figure (green ↗ better, red ↘ worse, grey `—` / `0%` with no move).

| Figure | Number | Delta source |
|---|---|---|
| P3 done on time `done/due` | live-run accumulator | **the other policy** on the same step (`jobs_completed`) |
| Overdue | same | **+10 steps** of the current policy |
| Revenue $ | same | **the other policy** on the same step |
| Critical charge (brownout) | same | **+10 steps** of the current policy |
| Charge below reserve | same | **+10 steps** of the current policy |
| Rejected commands | same | no delta (0 after repair) |

Horizon is 10 steps or the end of the shift, whichever is closer. Percent: `|b − a| / max(|a|, ε)`. For overdue and charge, a larger number is worse (red ↘). For revenue and completed jobs the comparison is `ORBI_ALT_MODEL` (`ckpt_35714`, revenue): more of ours than the other network — green ↗.

The other policy starts when the JSON is imported (`GET /api/dispatch/alt`). It is recomputed only with `POST /api/dispatch/resume` after an event or a confirmed goal switch. A step without a message does not touch the other policy’s tail.

The goal-switch modal does not compare two trajectories from step 0. It is a **fork from the current k**: “stay” is the live-run frame at min(k+10, end); “switch” is `POST /api/dispatch/whatif`. Cache key is step+goal+events; prefetch of the current (and next, when autoplaying) step. After resume the cache is cleared.

A click on the row opens: jobs total, completed, $ potential.

## Loading overlay

Full screen, spinner, caption “Recomputing policy”. Blocks clicks. Shown when a shift JSON is loaded and when the policy is recomputed after events.

## Placeholders

- **Compare variants** (`CompareView`) — a full-screen placeholder “Priorities vs Revenue”.
- Comparing two policies on run charts (`example.baseline`) is separate. On the console the bottom summary compares revenue and completed jobs with `ORBI_ALT_MODEL` on the same step (an independent run from 0). The goal-switch modal is a fork from k to +10 steps.
