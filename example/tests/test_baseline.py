"""Benchmark in `example.baseline`: fleets, load, Hungarian, event tape, metrics, outputs.

The full-shift checks share one small benchmark run (N=16, seed 0). RL joins it
when a trained checkpoint is found under `experiments/runs` or in
`ORBI_BENCH_MODEL`; otherwise the RL cases skip.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

import example.baseline as bench
from example.baseline import (
    FLEET_SIZES,
    GREEDY,
    HUNGARIAN,
    RL_DETERMINISTIC,
    RL_STOCHASTIC,
    EventTape,
    GreedyMethod,
    HungarianMethod,
    Options,
    TickView,
    Workload,
    check_counts,
    fleet_config,
    fleet_order,
    hungarian_action,
    make_env,
    make_method,
    plane_index,
    play,
    shift_outcome,
)
from orbi.env import N_MAX as ENV_N_MAX
from orbi.env import EventConfig, digest, load, resolve_scenario

TABLE = {
    16: (80, 67, 117, 5, 9, 2),
    24: (120, 100, 175, 8, 14, 3),
    32: (160, 133, 233, 11, 19, 4),
    40: (200, 167, 292, 13, 23, 5),
    48: (240, 200, 350, 16, 28, 6),
}


def _checkpoint() -> str | None:
    named = os.environ.get("ORBI_BENCH_MODEL")
    if named:
        return named
    runs = sorted(Path(bench.REPO, "experiments", "runs").glob("*/model.zip"), key=lambda p: p.stat().st_mtime)
    return str(runs[-1]) if runs else None


@pytest.fixture(scope="module")
def base() -> dict:
    return load(resolve_scenario("p03"))


@pytest.fixture(scope="module")
def policy(base):
    path = _checkpoint()
    if path is None:
        pytest.skip("no trained checkpoint under experiments/runs")
    model, _info = bench.load_policy(path, base)
    return model


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """One full common-tape shift at N=16, seed 0, for every available method."""
    model = _checkpoint()
    opts = Options(
        model=model,
        fleet_sizes=(16,),
        seeds=(0,),
        methods=("greedy", "hungarian", "rl") if model else ("greedy", "hungarian"),
        rl_secondary=True,
        output=str(tmp_path_factory.mktemp("bench")),
    )
    out = bench.run_benchmark(opts)
    with open(out / "episodes.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return opts, out, rows


def test_constants_match_the_simulator():
    base = EventConfig()
    assert bench.N_MAX == ENV_N_MAX == 48
    assert bench.TOP_K == 64
    assert (bench.BASE_POOL_MIN, bench.BASE_POOL_MAX) == (base.pool_min, base.pool_max)
    assert (bench.BASE_ADDED_JOBS_MIN, bench.BASE_ADDED_JOBS_MAX) == tuple(base.add_jobs_count)
    assert bench.BASE_OUTAGE_SATS == tuple(base.outage_sats)
    assert FLEET_SIZES == (16, 24, 32, 40, 48)


def test_every_fleet_builds_without_touching_the_scenario(base):
    before = digest(base)
    by_id = {s["id"]: s for s in base["satellites"]}
    for n in FLEET_SIZES:
        cfg = fleet_config(base, n)
        env = make_env(cfg, Workload.for_fleet(n).events(live=False), seed=0)
        assert env.n_sats == n
        assert cfg["model"]["downlink_parallel_limit"] == TABLE[n][5]

        # Every kept satellite is the scenario's own record, field for field.
        for sat in cfg["satellites"]:
            assert sat == by_id[sat["id"]]
    assert digest(base) == before


def test_fleets_span_all_six_planes_and_are_nested(base):
    planes = plane_index(base["satellites"])
    assert sorted(set(planes.tolist())) == list(range(6))
    order = fleet_order(base["satellites"])
    assert sorted(order) == list(range(48))
    previous: set[str] = set()
    for n in FLEET_SIZES:
        ids = {s["id"] for s in fleet_config(base, n)["satellites"]}
        keep = [i for i, s in enumerate(base["satellites"]) if s["id"] in ids]
        counts = np.bincount(planes[keep], minlength=6)
        assert (counts > 0).all()
        assert counts.max() - counts.min() <= 1
        assert previous <= ids
        previous = ids

    # The 16-satellite fleet is not the first two planes.
    first16 = {s["id"] for s in base["satellites"][:16]}
    assert {s["id"] for s in fleet_config(base, 16)["satellites"]} != first16


def test_tensor_sizes_stay_at_n_max_and_k(base):
    for n in FLEET_SIZES:
        env = make_env(fleet_config(base, n), Workload.for_fleet(n).events(live=False), seed=0)
        assert env.N == 48 and env.K == 64
        assert env.observation_space["sats"].shape == (48, 14)
        assert env.observation_space["jobs"].shape == (64, 14)
        assert env.observation_space["edges"].shape == (48, 64, 8)
        assert env.observation_space["mask"].shape == (48, 66)
        assert list(env.action_space.nvec) == [66] * 48


def test_checkpoint_fits_the_benchmark_tensors(base, policy):
    env = make_env(fleet_config(base, 16), Workload.for_fleet(16).events(live=False), seed=0)
    for key, space in env.observation_space.spaces.items():
        assert policy.observation_space[key].shape == space.shape
    assert list(policy.action_space.nvec) == [66] * 48


def test_incompatible_checkpoint_stops_with_a_clear_error(base, tmp_path):
    import gymnasium as gym

    class Wrong:
        observation_space = gym.spaces.Dict({"sats": gym.spaces.Box(0, 1, (16, 14), np.float32)})
        action_space = gym.spaces.MultiDiscrete([34] * 16)
        policy = object()
        device = "cpu"

    path = tmp_path / "wrong.zip"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(SystemExit, match="Incompatible RL checkpoint"):
        bench.check_checkpoint(Wrong(), path, base)


def test_rl_without_a_checkpoint_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="--model"):
        bench.run_benchmark(Options(model=None, output=str(tmp_path)))


def test_workload_scales_with_the_fleet(base):
    for n, (initial, pool_min, pool_max, add_min, add_max, downlink) in TABLE.items():
        work = Workload.for_fleet(n)
        assert (work.initial_jobs, work.pool_min, work.pool_max, work.add_min, work.add_max, work.downlink_limit) == TABLE[n]
        assert work.p_add_jobs == EventConfig().p_add_jobs
        assert work.p_outage == EventConfig().p_outage
        assert work.outage_len == tuple(EventConfig().outage_len)
        assert 1 <= work.outage_sats[0] <= work.outage_sats[1] <= 3


def test_reset_applies_the_load_and_greedy_reads_the_new_limit(base):
    for n, (initial, pool_min, pool_max, add_min, add_max, downlink) in TABLE.items():
        work = Workload.for_fleet(n)
        env = make_env(fleet_config(base, n), work.events(live=True), seed=3)
        obs, _ = env.reset(seed=3)
        assert env.downlink_limit == downlink
        assert sum(job_id.startswith("J") for job_id in env.jobs) == initial
        assert (env.events_cfg.pool_min, env.events_cfg.pool_max) == (pool_min, pool_max)
        assert tuple(env.events_cfg.add_jobs_count) == (add_min, add_max)

        greedy = GreedyMethod()
        greedy.begin(n, 3)
        greedy.act(obs, env)
        assert greedy.diagnostics()["downlink_limit_used"] == downlink


def _view(n_sats: int, slots: list[dict], limit: int, soc=None) -> TickView:
    return TickView(
        sat_ids=[f"S{i}" for i in range(n_sats)],
        soc=np.asarray(soc if soc is not None else [0.5] * n_sats, dtype=float),
        slots=slots,
        holders={},
        last_executor={},
        downlink_limit=limit,
    )


def _obs(mask: np.ndarray) -> dict:
    return {"mask": mask.astype(np.float32), "sats": np.zeros((mask.shape[0], 14), dtype=np.float32)}


def _slot(job_id: str, kind: str, value: float, remaining: int = 1, work: int | None = None) -> dict:
    return {"id": job_id, "kind": kind, "value_usd": value, "remaining_steps": remaining, "work_steps": work or remaining}


def test_hungarian_beats_a_greedy_order_on_a_crossed_pair():
    # A-X is the single best pair, but A-Y plus B-X is worth more.
    k = 2
    mask = np.zeros((2, k + 2), dtype=bool)
    mask[:, k] = True
    mask[0, 0] = mask[0, 1] = mask[1, 0] = True
    view = _view(2, [_slot("X", "relay", 10.0), _slot("Y", "relay", 9.0)], limit=1)
    action, info = hungarian_action(_obs(mask), view)
    assert action.tolist() == [1, 0]
    assert not info["capacity_repaired"]


def test_hungarian_one_job_per_satellite_and_one_satellite_per_job():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n, k = 6, 8
        mask = np.zeros((n, k + 2), dtype=bool)
        mask[:, :k] = rng.random((n, k)) < 0.5
        mask[:, k] = True
        slots = [_slot(f"J{j}", "relay", float(rng.uniform(1, 10)), int(rng.integers(1, 4))) for j in range(k)]
        action, _ = hungarian_action(_obs(mask), _view(n, slots, limit=2))
        jobs = [a for a in action if a < k]
        assert len(jobs) == len(set(jobs))
        assert action.shape == (n,)
        assert all(mask[i, a] for i, a in enumerate(action))


def test_hungarian_repairs_the_downlink_cap_and_refills_relays():
    # Three satellites prefer three downlinks; the cap is one, so two move to relays.
    k = 5
    mask = np.zeros((3, k + 2), dtype=bool)
    mask[:, k] = True
    mask[:, :5] = True
    slots = [
        _slot("D0", "downlink", 30.0),
        _slot("D1", "downlink", 20.0),
        _slot("D2", "downlink", 10.0),
        _slot("R0", "relay", 2.0),
        _slot("R1", "relay", 1.0),
    ]
    action, info = hungarian_action(_obs(mask), _view(3, slots, limit=1))
    downs = [a for a in action if a < 3]
    assert len(downs) == 1 and downs[0] == 0
    assert sorted(a for a in action if 3 <= a < 5) == [3, 4]
    assert info["capacity_repaired"] and info["dropped_downlinks"] == 2


def test_hungarian_skips_worthless_and_masked_pairs():
    k = 2
    mask = np.zeros((2, k + 2), dtype=bool)
    mask[:, k] = True
    mask[0, 0] = True
    view = _view(2, [_slot("Z", "relay", 0.0), _slot("Y", "relay", 5.0)], limit=1)
    action, _ = hungarian_action(_obs(mask), view)
    assert action.tolist() == [k, k]


def test_hungarian_downlink_restart_costs_the_full_work():
    k = 1
    mask = np.zeros((2, k + 2), dtype=bool)
    mask[:, k] = True
    mask[:, 0] = True
    view = _view(2, [_slot("D", "downlink", 12.0, remaining=1, work=4)], limit=1)
    view.holders = {"D": "S1"}
    weights = bench.pair_weights(mask, view)
    assert math.isclose(weights[0, 0], 12.0 / 4 + bench.HUNGARIAN_TIE_SOC * 0.5)
    assert math.isclose(weights[1, 0], 12.0 / 1 + bench.HUNGARIAN_TIE_SOC * 0.5)


def test_hungarian_on_live_ticks_respects_mask_and_cap(base):
    n = 24
    work = Workload.for_fleet(n)
    env = make_env(fleet_config(base, n), work.events(live=True), seed=5)
    obs, _ = env.reset(seed=5)
    method = HungarianMethod()
    method.begin(n, 5)
    for _ in range(6):
        action = method.act(obs, env)
        mask = np.asarray(obs["mask"], dtype=bool)
        assert all(mask[i, a] for i, a in enumerate(action))
        slots = [a for a in action[: env.n_sats] if a < env.K]
        assert len(slots) == len(set(slots))
        downs = sum(env._candidates[a].job["kind"] == "downlink" for a in slots)
        assert downs <= env.downlink_limit
        obs, *_ = env.step(action)
    reasons = {row["reason"] for row in env.trace}
    assert "ground_capacity" not in reasons and "duplicate_job_in_step" not in reasons


def test_shift_outcome_counts_each_job_once():
    class Env:
        k = 10
        n_sats = 4
        jobs = {
            "J0": {"id": "J0", "kind": "relay", "deadline_step": 8, "completed_step": 5, "value_usd": 3.0, "priority": 1},
            "J1": {"id": "J1", "kind": "relay", "deadline_step": 9, "completed_step": None, "value_usd": 1.0, "priority": 3},
            "J2": {"id": "J2", "kind": "downlink", "deadline_step": 10, "completed_step": 10, "value_usd": 6.0, "priority": 3},
            "J3": {"id": "J3", "kind": "downlink", "deadline_step": 7, "completed_step": None, "value_usd": 2.0, "priority": 2},
            "J4": {"id": "J4", "kind": "downlink", "deadline_step": 15, "completed_step": None, "value_usd": 4.0, "priority": 1},
            "J5": {"id": "J5", "kind": "relay", "deadline_step": 12, "completed_step": 9, "value_usd": 2.0, "priority": 1},
        }

    out = shift_outcome(Env())
    assert out["jobs_generated_total"] == 6
    assert out["jobs_due_total"] == 4 and out["jobs_future_deadline"] == 2
    assert out["missed_total"] == 2 and out["missed_relay"] == 1 and out["missed_downlink"] == 1
    assert out["pending_with_future_deadline"] == 1
    assert out["unfinished_at_shift_end"] == 3
    assert out["completed_total"] == 3
    assert math.isclose(out["revenue_usd"], 11.0)
    assert math.isclose(out["revenue_per_completed_job"], 11.0 / 3)
    assert math.isclose(out["completion_rate_total"], 2 / 4)
    assert math.isclose(out["q_selection_downlink"], (6.0 / 1) / ((6.0 + 2.0) / 2))
    assert out["completed_p3"] == 1 and out["due_p3"] == 2


def test_no_completions_give_null_value_per_job():
    class Env:
        k = 5
        n_sats = 2
        jobs = {"J0": {"id": "J0", "kind": "relay", "deadline_step": 3, "completed_step": None, "value_usd": 3.0}}

    out = shift_outcome(Env())
    assert out["revenue_per_completed_job"] is None
    assert out["q_selection_total"] is None


def test_tape_is_hidden_until_its_tick(base):
    n, seed, cut = 16, 2, 10
    cfg = fleet_config(base, n)
    work = Workload.for_fleet(n)
    tape = EventTape.record(cfg, work, seed)
    assert tape.events and all(event["at_step"] < 288 for event in tape.events)

    # The same policy sees the same observations whether or not the later tape exists.
    early = [event for event in tape.events if event["at_step"] <= cut]
    full = make_env(cfg, work.events(live=False), seed, copy.deepcopy(tape.events))
    short = make_env(cfg, work.events(live=False), seed, copy.deepcopy(early))
    obs_a, _ = full.reset(seed=seed)
    obs_b, _ = short.reset(seed=seed)
    arrival = tape.arrival_steps()
    for _ in range(cut):
        for key in obs_a:
            assert np.array_equal(obs_a[key], obs_b[key]), key
        assert all(arrival.get(job_id, -1) <= full.k for job_id in full.jobs)
        greedy = GreedyMethod()
        greedy.begin(n, seed)
        action = greedy.act(obs_a, full)
        obs_a, *_ = full.step(action)
        obs_b, *_ = short.step(action)


def test_methods_reproduce_on_the_same_seed(base):
    n, seed = 16, 1
    cfg = fleet_config(base, n)
    work = Workload.for_fleet(n)
    tape = EventTape.record(cfg, work, seed)
    again = EventTape.record(cfg, work, seed)
    assert tape.tape_hash == again.tape_hash

    # Stochastic RL is seeded per (N, seed), so it must repeat too.
    path = _checkpoint()
    policy = bench.load_policy(path, base)[0] if path else None
    names = [GREEDY, HUNGARIAN] + ([RL_DETERMINISTIC, RL_STOCHASTIC] if policy is not None else [])
    skip = {"shift_wall_s", "decision_ms_mean", "decision_ms_median", "decision_ms_p95", "decision_ms_max", "decision_ms_first"}
    for name in names:
        first = play(make_method(name, policy), cfg, work, seed, "common", tape, max_steps=25)
        second = play(make_method(name, policy), cfg, work, seed, "common", again, max_steps=25)
        a = {k: v for k, v in first.items() if k not in skip}
        b = {k: v for k, v in second.items() if k not in skip}
        assert a == b, name


def test_every_method_plays_a_full_shift(full_run):
    opts, _out, rows = full_run
    assert {row["method"] for row in rows} == set(opts.method_names())
    for row in rows:
        assert int(row["steps_executed"]) == 288
        assert int(row["decisions"]) == 288
        assert int(row["invalid_actions"]) == 0
        if row["method"] == HUNGARIAN:
            assert int(row["ground_capacity_rejections"]) == 0
            assert int(row["duplicate_job_rejections"]) == 0


def test_kinds_sum_to_totals_and_counters_agree(full_run):
    _opts, _out, rows = full_run
    for row in rows:
        values = {k: _num(v) for k, v in row.items()}
        assert row["consistency_errors"] == ""
        assert check_counts(values) == []
        for stem in ("completed", "missed", "jobs_due", "jobs_generated"):
            assert values[f"{stem}_total"] == values[f"{stem}_relay"] + values[f"{stem}_downlink"]
        assert math.isclose(values["revenue_usd"], values["revenue_relay_usd"] + values["revenue_downlink_usd"], abs_tol=1e-6)
        assert values["completed_total"] + values["missed_total"] == values["jobs_due_total"]
        assert values["pending_with_future_deadline"] == 0


def test_value_per_job_is_revenue_over_completions(full_run):
    _opts, _out, rows = full_run
    for row in rows:
        revenue, done = float(row["revenue_usd"]), int(row["completed_total"])
        assert done > 0
        assert math.isclose(float(row["revenue_per_completed_job"]), revenue / done, rel_tol=1e-9)
        assert math.isclose(float(row["revenue_per_satellite"]), revenue / 16, rel_tol=1e-9)


def test_common_tape_is_identical_for_every_method(full_run):
    _opts, out, rows = full_run
    for key in ("tape_hash", "initial_hash", "arrivals_hash"):
        assert len({row[key] for row in rows}) == 1, key
    assert all(row["tape_delivered_ok"] == "True" for row in rows)
    assert all(int(row["future_event_leaks"]) == 0 for row in rows)
    checks = json.loads((out / "report.json").read_text(encoding="utf-8"))["checks"]
    assert checks["common_tape_identical_across_methods"] is True
    assert checks["initial_catalog_identical_across_methods"] is True
    assert checks["greedy_downlink_limit_ok"] is True
    assert checks["workload_after_reset_ok"] is True
    assert checks["counter_consistency_errors"] == 0


def test_outputs_are_written_with_matching_values(full_run):
    opts, out, rows = full_run
    for name in ("config.json", "episodes.csv", "summary.csv", "paired_deltas.csv", "report.json", "workload.json"):
        assert (out / name).is_file(), name
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "preliminary"
    assert json.loads((out / "workload.json").read_text(encoding="utf-8"))["16"]["after_reset"]["common"]["downlink_limit"] == [2]

    # With one seed the summary mean is the episode value itself.
    with open(out / "summary.csv", encoding="utf-8") as fh:
        summary = list(csv.DictReader(fh))
    by_method = {row["method"]: row for row in rows}
    for item in summary:
        if item["metric"] == "revenue_usd":
            assert math.isclose(float(item["mean"]), float(by_method[item["method"]]["revenue_usd"]))

    # Paired deltas are A - B on the same seed.
    with open(out / "paired_deltas.csv", encoding="utf-8") as fh:
        deltas = list(csv.DictReader(fh))
    for item in deltas:
        if item["comparison"] == f"{HUNGARIAN} - {GREEDY}" and item["metric"] == "revenue_usd":
            want = float(by_method[HUNGARIAN]["revenue_usd"]) - float(by_method[GREEDY]["revenue_usd"])
            assert math.isclose(float(item["mean_delta"]), want, abs_tol=1e-9)

    # Plots: present, and at least 200 dpi for the panel width they were drawn at.
    import matplotlib.image as mpimg

    for name in ("completed_tasks.png", "revenue.png", "revenue_per_task.png", "missed_tasks.png", "revenue_per_satellite.png"):
        path = out / "plots" / name
        assert path.is_file() and path.stat().st_size > 10_000, name
        height, width = mpimg.imread(path).shape[:2]
        assert height >= 4.2 * 200 and width >= 5.2 * 200


def test_cli_parses_ranges_and_smoke():
    opts = bench.parse(["--model", "m.zip", "--seeds", "0:29", "--fleet-sizes", "16,48", "--event-mode", "both"])
    assert opts.seeds == tuple(range(30))
    assert opts.fleet_sizes == (16, 48)
    assert opts.event_modes == ("common", "native")
    assert opts.method_names() == [GREEDY, HUNGARIAN, RL_STOCHASTIC]
    smoke = bench.parse(["--smoke", "--methods", "greedy,hungarian", "--rl-mode", "deterministic", "--rl-secondary"])
    assert smoke.seeds == (0, 1, 2)
    assert smoke.method_names() == [GREEDY, HUNGARIAN]
    rl = bench.parse(["--model", "m.zip", "--rl-mode", "deterministic", "--rl-secondary"])
    assert rl.method_names() == [GREEDY, HUNGARIAN, RL_DETERMINISTIC, RL_STOCHASTIC]


def test_results_never_land_in_the_repository_root():
    with pytest.raises(SystemExit, match="repository root"):
        bench.run_benchmark(Options(methods=("greedy",), output=str(bench.REPO)))


def _num(value: str):
    if value in ("", None):
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number
