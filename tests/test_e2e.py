import itertools

import pytest

from traj_opt import scenario
from traj_opt.planner import build_problem, solve

SEEDS = [0, 3, 7]


@pytest.fixture(scope="module", params=SEEDS)
def problem(request):
    return build_problem(scenario.generate(request.param))


def test_bnb_plan_verifies(problem):
    plan, report = solve(problem, "bnb")
    assert report.ok, report.violations
    assert plan.status in ("CONFLICT_FREE", "PARTIAL")


def test_greedy_plan_verifies_when_found(problem):
    plan, report = solve(problem, "greedy")
    if plan.status == "CONFLICT_FREE":
        assert report.ok, report.violations


def test_deterministic():
    a, _ = solve(build_problem(scenario.generate(5)), "bnb")
    b, _ = solve(build_problem(scenario.generate(5)), "bnb")
    assert a.cost == b.cost
    assert {k: (c.label if c else None) for k, c in a.chosen.items()} == {
        k: (c.label if c else None) for k, c in b.chosen.items()
    }


def test_json_roundtrip(tmp_path):
    sc = scenario.generate(1)
    scenario.save(sc, tmp_path / "s.json")
    assert scenario.load(tmp_path / "s.json") == sc


def test_sector_greedy_valid_and_not_better_than_bnb(problem):
    import math

    from traj_opt.select import assign_sectors

    sec = assign_sectors(problem.sc, problem.cands)
    for tgt in problem.sc.targets:
        for a, b in itertools.combinations(tgt.aircraft, 2):
            d = abs((sec[a] - sec[b] + math.pi) % (2 * math.pi) - math.pi)
            assert d >= math.radians(problem.sc.params.min_sector_deg) - 1e-9
    sg, rep = solve(problem, "sector_greedy")
    bb, _ = solve(problem, "bnb")
    if sg.status == "CONFLICT_FREE":
        assert rep.ok, rep.violations
        assert all(c.ingress_angle == sec[a] for a, c in sg.chosen.items())
        assert bb.cost <= sg.cost + 1e-9
