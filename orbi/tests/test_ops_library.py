"""Initial JSON loads and aliases resolve."""

from __future__ import annotations

import pytest

from orbi.env import P01_INTRO, P02_SHIFT, SCENARIOS_DIR, load, resolve_scenario, validate


def test_all_scenarios_validate():
    for path in sorted(SCENARIOS_DIR.glob("P0*.json")):
        scenario = load(path)
        validate(scenario)
        assert scenario["time"]["step_s"] == 300
        assert "orbit" in scenario["satellites"][0]


def test_resolve_scenario_accepts_alias_and_path():
    assert resolve_scenario("p02") == P02_SHIFT
    assert resolve_scenario("shift") == P02_SHIFT
    assert resolve_scenario(str(P01_INTRO)) == P01_INTRO
    with pytest.raises(FileNotFoundError):
        resolve_scenario("no_such_scenario")
