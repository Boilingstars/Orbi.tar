# example

A shift dispatcher on Orbi.tar. Physics is shared: charge, temperature, calibration, and command admissibility live in `orbi`. The reward is the job's `value_usd` on completion. There is no progress credit.

```bash
py -3 -m example.train
py -3 -m example.evaluate --scenario p02 --policy greedy --events none
py -3 -m example.baseline --scenario p02 --goal priority
```

Scenarios: `scenarios/P01_intro.json` (16 satellites), `P02_shift.json` (48), `P03_energy.json`, `P04_demand.json`. Short names: `p01` … `p04`. Each file is initial parameters, including per-satellite orbits. Jobs are sampled from seed. Relay and downlink A/B are drawn from the active terminals (`job_difficulty` is the chance B sits near the mean coverage radius).

`control.py` is the dispatcher (job-slot ranking, fleet or cluster addressing). `satellite.py` subclasses the `orbi` shell: extra attributes, neighbor memory, extra `can_take` rules.

## Relay and downlink

Relay keeps `remaining_steps` across a pause or a change of satellite.

Downlink is one continuous pass by one satellite. It has a single window `release_step`–`deadline_step`. Idle, calibrate, lost view, or a new executor resets `remaining_steps` to `work_steps`. An assignment of satellite `i` is legal only when the ticks still needed fit in both the time left to the deadline and that satellite's current access run:

- the current holder needs the remaining ticks
- anyone else needs the full duration, because a switch resets progress

The same rule is used for the action mask, `can_execute()`, and `step()`. Idle stays legal, so a downlink can be dropped for a better request.

## Observation

`N_max = 48`, `K = 64` job slots. Rows beyond the real fleet are zeros: zero capacity is how padding is told from a satellite.

| Tensor | Shape | Contents |
|--------|-------|----------|
| `sats` | `48 x 14` | SoC, margin above reserve, temperature, calibration, availability, windows (any active ground point in view), `solar_w`, steps to eclipse and to sun, capacity, heater, power balance |
| `jobs` | `K x 14` | priority, value, progress, slack, kind, eligible satellites, last executor, steps to `release`, slack in work-ticks |
| `plan` | `192 x 14` | the same features for unfinished jobs that did not fit in the action slots |
| `edges` | `48 x K x 8` | eligible, executable now, energy ok, sees A and B, last executor, remaining window, next rise, uniqueness |
| `neighbors` | `48 x M x 7` | on-board view of other spacecraft (`M` from `neighbor_memory`, default 12) |
| `global` | `7` | step, open jobs, priority 3, downlink demand, cap, mean and min SoC |
| `mask` | `48 x (K+2)` | valid actions |

A satellite action is a job slot, idle, or calibrate. Ground control packs `K = 64` job slots ([control.py](control.py)). Neighbor memory is [satellite.py](satellite.py): `competitors` (default) or `nearest_index`. `_baseline_repair` is off unless `baseline_repair=true`.

## Network

`controller=dispatcher` (default) is one forward per tick over the whole fleet. Self-attention over satellites coordinates a single job and the ground slots. Self-attention over jobs sees the whole queue. Cross-attention follows the eligible-pair mask. The pointer scores satellite–slot pairs; a job-first matching assigns at most one satellite per slot and may skip. `idle` and `calibrate` are leftover heads. There is one critic per tick: reward is assigned to the world step, not to a satellite.

`controller=ctde` keeps that critic and replaces the actor with a shared on-board network: own `sats[i]`, the job table, own edges, and `neighbors[i]`. Fleet mean/min SoC are zeroed for the actor. MaskablePPO and the `N × (K+2)` action stay the same.

Weights do not depend on fleet size: a model trained on 48 plays on 16. Profile `full` is width 128 with two blocks. `light` is for a short CPU check. Network code: [models.py](models.py).

```bash
py -3 -m example.train
```
