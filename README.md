# Orbi.tar

A training and evaluating environment for RL agents that operate a satellite constellation. Physics is live Kepler: each satellite keeps its orbit from the initial JSON (RAAN, period, inclination, altitude). On every 5-minute tick the environment advances battery state of charge, temperature, and calibration age from sunlight and payload load. A command is idle, calibrate, or a job (relay or downlink). A job needs line of sight to a source point A and a sink point B on Earth; one satellite must see both, with no inter-sat relay. Admissibility also depends on energy reserve, thermal limits, calibration lifetime, and a shared downlink slot limit.

Jobs, events, and the active ground-point subset are sampled from `seed`. Orbits are not redrawn on `reset`.

Repository: [https://github.com/Boilingstars/Orbi.tar](https://github.com/Boilingstars/orbi-tar.git).
License: [MIT](LICENSE).

## KeplerWorld

`KeplerWorld` in `orbi/kepler.py` is the constellation geometry. Position is the analytic Kepler problem (ECI, then ECEF with Earth rotation). Eclipse is a cylindrical Earth shadow. `step` advances time; energy and temperature live on the satellite template in `orbi/satellite.py`. Visibility of job points A and B is `sees_job`: elevation at both ground points must meet `min_elevation_deg`. Typical one-sat coverage radius is `coverage_radius_m()` from mean altitude. Ground terminals come from `orbi/ground_points.json` (inhabited land, 100 km covering; rebuild with `py -3 -m orbi.coverage`). Each reset draws `ground_points` active sites from that catalog (`m` in `train.toml`). Job pairs A/B are near (distance ≤ coverage) with chance `1 - far_job_p`, and farther than coverage with chance `far_job_p`.

The fleet and the element set are fixed at load. Satellites cannot be added, and orbits are not recomputed.

## Install

Python 3.10+.

```bash
py -3 -m pip install -e ".[dev]"
py -3 -m pip install -e ".[train]"
py -3 -m pip install -e ".[console]"
```

`train` installs torch, stable-baselines3, and the tqdm/rich progress bar. `console` installs the evaluation server. The environment step and a custom reward do not need either extra.

## First steps

```python
from orbi.env import OpsEnv
from example.task import make_task

env = make_task("p01", goal="priority").make_env(seed=0)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`reset(seed=...)` returns a dict observation and an action mask. `info` after the last step carries `summary`. Replay is `reset(seed)` plus the stored commands.

A task is an initial JSON plus a reward and an observation. Templates are `Reward`, `Observation`, and `Task` in `orbi/env.py`; `example/task.py` binds the example dispatcher.

```python
from orbi.env import Reward, Observation, Task

class MyReward(Reward):
    name = "my-task"

    def on_complete(self, job):
        return float(job["value_usd"])

class MyObservation(Observation):
    def encode(self, env):
        return {"step": env.k}

env = Task("p01", MyReward(), MyObservation()).make_env()
```

`Reward.score` applies the shared step terms: job progress, reserve violation, brownout, blocked commands, and executor switches. A new objective implements `on_complete` and `on_miss`.

The single gym entry is `orbi.env.OpsEnv`. Train, greedy (`orbi.greedy`), and the console all use it.

## Training

[train.toml](train.toml) sets the scenario, goal, timesteps, seed, network profile, `ground_points` (`m` active terminals), and `far_job_p` (`x`). Seed drives jobs, events, sun, initial SoC, and which terminals are active; it does not change orbits.

```bash
py -3 -m example.train
```

Command-line flags override the file. Run logs are written under `agent/runs/` and are not part of the repository.

[example/](example/) is the programmatic interface: dispatcher (`control.py`), custom satellite subclass, attention policy, and scenarios `P01`–`P04`. Observation layout and the network are described in [example/README.md](example/README.md).

## Evaluation console

[console/](console/) serves a loaded shift and a policy evaluation. FastAPI keeps one scenario in memory and streams the step trace; nginx serves the operator UI. The globe draws satellite positions from the same `KeplerWorld`. Layout and color rules: [console/INTERFACE.md](console/INTERFACE.md).

```bash
docker compose -f console/docker-compose.yml up --build
```

## Tests

```bash
py -3 -m pytest -q
```

## Layout

```
orbi/            KeplerWorld, satellite template, OpsEnv, greedy, ground points
example/         dispatcher, custom satellite, policy, initial JSON
console/         evaluation server and operator UI
train.toml       training config
```

## Developers

How one shift tick is represented to a policy. Feature tables live in [example/README.md](example/README.md); this section is the data path.

### Input

An initial JSON is the world template: per-satellite orbits and power, model limits, time grid, job-generation counts. `OpsEnv.reset` builds `KeplerWorld` and `Satellite` objects, then samples jobs from seed. The policy never sees the JSON; it sees the tensors built from live state.

### Encoding the fleet

`ShiftObservation.encode` snapshots every satellite into a row `sats[i]`. That row is satellite `sat_ids[i]` at the current 5-minute step: charge, temperature, calibration age, availability, contact with any active ground point, solar power, and time to shadow or sun. Rows past the real fleet stay zero (`N_max = 48`). Zero capacity is how padding is told from a live spacecraft.

### Encoding the queue

Ground control (`example/control.py`) ranks unfinished jobs and packs them into `jobs`, a fixed 32-slot table. Each slot is one candidate the policy may name this tick: priority, value, progress, slack, kind, eligibility size. Slot index is not a stable job id. Jobs that do not fit still sit in `plan` for the critic, but they are not actions. A cluster subset of `sat_ids` limits which spacecraft appear as eligible executors.

### Pairing satellites with slots

`edges[i, slot]` is the bipartite link between `sats[i]` and `jobs[slot]`: listed as eligible, executable now, energy ok, contact present (`sees_job` on A and B), last executor. A zero edge means that satellite cannot be assigned that slot. `mask[i]` is the same graph flattened into legal actions for `i`: the 32 job slots, idle, and calibrate. The environment will sample only where the mask is true. Idle stays open on padding rows so the categorical is never empty.

`neighbors[i]` is a fixed-size memory of other spacecraft for on-board actors (`example/satellite.py`): competitors on shared executable slots, or nearest indices in `sat_ids`. Global tensors stay as they were; a CTDE actor reads `sats[i]`, `jobs`, `edges[i]`, `mask[i]`, and `neighbors[i]`, not the rest of the fleet.

### Output

The policy returns one integer per satellite: a job slot, idle, or calibrate. Choosing idle, calibrate, or another job abandons the previous assignment for that tick; `remaining_steps` are kept. `OpsEnv` maps slots back to `job_id` and steps Kepler plus satellite energy/thermal. Optional `_baseline_repair` (off by default) is a greedy joint-action projection for conflict-free assignment. Duplicate jobs and extra downlinks become idle.
