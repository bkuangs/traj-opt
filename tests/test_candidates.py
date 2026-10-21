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


def test_zigzag_hits_length_and_is_narrower_than_bulge():
    sc = _sc(loiter=False, zigzag=True)
    T = common_ingress_time(sc)
    n = 0
    for ac in sc.aircraft:
        cs = [c for c in build_candidates(sc, ac, T) if c.alt_offset == 0]
        for z in (c for c in cs if c.geometry.startswith("zigzag")):
            b = [c for c in cs if c.geometry.startswith("bulge") and c.ingress_angle == z.ingress_angle]
            assert abs(z.speed * T - z.length) < 1e-6
            if b:
                assert abs(b[0].length - z.length) < 1.0
                assert z.lateral < min(c.lateral for c in b)
                n += 1
    assert n > 0


@pytest.mark.parametrize("seed", range(10))
def test_zigzag_plans_verify(seed):
    from traj_opt.planner import build_problem, solve

    sc = scenario.generate(seed)
    sc = dataclasses.replace(sc, params=dataclasses.replace(sc.params, loiter=False, zigzag=True, footprint_weight=2.0))
    prob = build_problem(sc)
    for s in ("greedy", "bnb"):
        plan, rep = solve(prob, s)
        assert rep.ok, rep.violations
