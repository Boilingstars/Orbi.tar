# Working in this repository

Style for changes under `orbi/`, `tools/`, and a task package such as `example/`. Comments and names stay in English.

## Where a change belongs

### `orbi/`

The simulator. Kepler, coverage, the satellite's battery and temperature, `OpsEnv.step`, action masks, loading and checking a scenario, and the greedy policy that only reads an observation.

A change here is something every agent experiences: an orbit, a tick, whether a command is legal, the shape of the gym step. Do not put a network, a training loop, a task reward, a job catalog, or a plot here.

### `tools/`

The contracts a task package implements, and the harness around them. `Agent`, `Reward`, `Observation`, job families and `register_job_field`, `Run` / `Trainer` / `launch`, the run logger, and evaluation of one actor.

A change here is an interface or the loop that calls it. Do not put scenario JSON, a network, one package's reward numbers, or charts of one run here.

### `example/` and any package beside it

One task. Its satellite subclass, dispatcher, reward, observation tensors, job subclasses, network, training recipes, scenarios, plots, and baselines. It imports `orbi` and `tools`. It does not copy the physics.

A new objective or a new policy is a module in that package, or a new package with the same shape: `task.py` binds the reward and the observation, `agent.py` names the actors, `train.py` says which actor learns by which function. `console/` is the operator UI and is not a task package.

## How to write it

Open a file on the thing a reader came for. The class, then its main methods, then the private functions those methods call. A helper does not sit above the class it serves.

Inside a function, a blank line separates steps. A short comment on the line above says what that step does: what is computed, what is updated, where it is written. Do not comment a one-line function, and do not restate the syntax.

Separate `if`s are separate blocks. Two `if`s in a row have a blank line between them. `elif` and `else` stay on that same condition. A comment belongs to the `if` under it, so the blank line goes above the comment:

```python
if job is None:
    return False, "unknown_job"

if not self.available(k):
    return False, "satellite_unavailable"
```

One implementation. Call it. A report, a horizon walk, a hyperparameter list, and an episode loop each live in one place. A new reward term goes on the reward that already exists. A new decoder does not sit beside the old one.

Names and feature indices that other files already use stay as they are. A style edit does not change a mask, a reward number, or a tensor column.

## Check

```bash
py -3 -m pytest -q
```

Do not start a full training run to confirm a refactor. A few environment steps are enough when the step itself changed.
