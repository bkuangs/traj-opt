import dataclasses

import pytest

from traj_opt import scenario
from traj_opt.planner import build_problem, solve
from traj_opt.robust import Dispersions, monte_carlo


def _plan(**params):
    sc = scenario.load("scenarios/crossing.json")
    sc = dataclasses.replace(sc, params=dataclasses.replace(sc.params, **params))
    prob = build_problem(sc)
    plan, _ = solve(prob, "bnb")
    assert plan.status == "CONFLICT_FREE"
    return sc, plan


@pytest.fixture(scope="module")
def nominal():
    return _plan()


def test_zero_dispersion_reproduces_plan(nominal):
    sc, plan = nominal
    r = monte_carlo(sc, plan, 10, Dispersions(wind_sd=0, start_delay_sd=0, cross_track_sd=0, alt_sd=0))
    assert r.p_sep_loss == r.p_terrain == r.p_arrival_miss == 0.0
    assert r.arrival_err_p95 <= Dispersions().dt


def test_deterministic(nominal):
    sc, plan = nominal
    assert monte_carlo(sc, plan, 50, seed=3) == monte_carlo(sc, plan, 50, seed=3)


def test_speed_margin_buys_arrival_robustness(nominal):
    thin = _plan(speed_margin=0.5 / 15)
    assert monte_carlo(*nominal, 200).p_arrival_miss < monte_carlo(*thin, 200).p_arrival_miss


def test_large_errors_lose_separation(nominal):
    sc, plan = nominal
    r = monte_carlo(sc, plan, 100, Dispersions(cross_track_sd=150, alt_sd=60))
    assert r.p_sep_loss > 0


def test_per_aircraft_delivery(nominal):
    sc, plan = nominal
    clean = monte_carlo(sc, plan, 10, Dispersions(wind_sd=0, start_delay_sd=0, cross_track_sd=0, alt_sd=0))
    assert set(clean.p_deliver) == set(plan.chosen) and all(v == 1.0 for v in clean.p_deliver.values())
    r = monte_carlo(sc, plan, 200, Dispersions(wind_sd=4.0))
    p_any_fail = 1.0 - min(r.p_deliver.values())
    assert all(0.0 <= v <= 1.0 for v in r.p_deliver.values())
    assert r.p_arrival_miss >= p_any_fail - r.p_sep_loss - r.p_terrain - 1e-9 and p_any_fail > 0
