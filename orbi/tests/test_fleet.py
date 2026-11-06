"""The constellation is fixed at construction and does not change on step."""

from orbi.env import EventConfig, OpsEnv, P01_INTRO, load
from orbi.kepler import KeplerWorld


def test_kepler_fleet_stays_fixed():
    world = KeplerWorld.walker(n_sats=4, n_planes=2)
    world.reset(seed=1)
    n = world.n_sats
    world.step(5.0)
    assert world.n_sats == 4
    assert len(world.elements) == n == 4


def test_env_fleet_stays_fixed():
    env = OpsEnv(scenario=load(P01_INTRO), events=EventConfig.off(), seed=0)
    env.reset(seed=0)
    ids = list(env.sat_ids)
    env.step(env.action_space.sample() * 0 + env.idle_index)
    assert env.sat_ids == ids
