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
