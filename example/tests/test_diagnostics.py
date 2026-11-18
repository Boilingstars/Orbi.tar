"""Shift diagnostics stay off the default path and flatten for CSV."""

from __future__ import annotations

from example.baseline import GREEDY, fleet_config, make_method, parse, play, Workload
from example.diagnostics import assignment_delta, flatten_diagnose, pair_rho, required_ticks
from orbi.env import load, resolve_scenario


def test_required_ticks_resets_downlink_for_a_new_holder():
    job = {"kind": "downlink", "remaining_steps": 2, "work_steps": 5, "value_usd": 10.0}
    assert required_ticks(job, "S01", "S01") == 2
    assert required_ticks(job, "S02", "S01") == 5
    assert pair_rho(job, "S01", "S01") == 5.0
    assert pair_rho(job, "S02", "S01") == 2.0


def test_parse_diagnose_defaults_off():
    opts = parse(["--model", "m.zip", "--methods", "greedy"])
    assert opts.diagnose is False
    on = parse(["--model", "m.zip", "--methods", "greedy", "--diagnose"])
    assert on.diagnose is True


def test_flatten_diagnose_expands_quantiles():
    flat = flatten_diagnose({"completed_by_value_quantile": {"q0": 1, "q1": 2}, "missed_value_usd": 3.5})
    assert flat["completed_by_value_quantile_q0"] == 1
    assert flat["missed_value_usd"] == 3.5


def test_assignment_delta_counts_order_and_pairs():
    delta = assignment_delta([0, 1, 2], [0, 2, 1], [1, 0, 64], [1, 2, 64])
    assert delta["job_order_change_rate"] > 0
    assert delta["selected_pair_change_rate"] > 0


def test_diagnose_smoke_does_not_break_greedy():
    cfg = fleet_config(load(resolve_scenario("p01")), 16)
    work = Workload.for_fleet(16)
    row = play(make_method(GREEDY), cfg, work, 0, "native", None, max_steps=4, diagnose=True)
    assert "missed_value_usd" in row
    assert row["consistency_errors"] == ""
    assert row["diagnose_ticks"] == 4
