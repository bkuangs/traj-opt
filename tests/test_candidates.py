import dataclasses
import math

import pytest

from traj_opt import scenario
from traj_opt.candidates import build_candidates, common_ingress_time, ingress_options


def _sc(**params):
    sc = scenario.generate(1, n_targets=3, per_target=4, target_spread=800)
    return dataclasses.replace(sc, params=dataclasses.replace(sc.params, **params))


@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_spread_angles_are_even_and_keep_the_nearest(k):
    sc = _sc(ingress_spread=True, ingress_per_aircraft=k)
    nearest = _sc(ingress_per_aircraft=1)
    for ac in sc.aircraft:
        opts = ingress_options(sc, ac, sc.target_of(ac.id))
        angs = sorted(o[0] for o in opts)
        gaps = [(b - a) for a, b in zip(angs, angs[1:] + [angs[0] + 2 * math.pi], strict=True)]
        assert max(gaps) - min(gaps) < math.radians(30) + 1e-9
        assert ingress_options(nearest, ac, sc.target_of(ac.id))[0][0] in angs


def test_loiter_switch():
    sc = _sc(loiter=False, ingress_per_aircraft=6)
    T = common_ingress_time(sc)
    geoms = {c.geometry for ac in sc.aircraft for c in build_candidates(sc, ac, T)}
    assert geoms and not any(g.startswith("loiter") for g in geoms)
