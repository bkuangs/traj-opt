import dataclasses

import numpy as np
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


def test_new_terms_off_by_default(nominal):
    sc, plan = nominal
    explicit = Dispersions(gust_sd=0, xtrack_drift_sd=0, turn_xtrack_sd=0, alt_bias_sd=0, is_scale=1.0)
    assert monte_carlo(sc, plan, 50, explicit, seed=1) == monte_carlo(sc, plan, 50, seed=1)


def test_gusts_spread_arrival(nominal):
    sc, plan = nominal
    calm = monte_carlo(sc, plan, 200, Dispersions(wind_sd=0, start_delay_sd=0))
    gusty = monte_carlo(sc, plan, 200, Dispersions(wind_sd=0, start_delay_sd=0, gust_sd=3.0))
    assert gusty.arrival_err.mean() > calm.arrival_err.mean()


def test_drift_shrinks_separation(nominal):
    sc, plan = nominal
    base = monte_carlo(sc, plan, 200, Dispersions(cross_track_sd=0))
    more = monte_carlo(sc, plan, 200, Dispersions(cross_track_sd=0, xtrack_drift_sd=60.0))
    assert np.median(more.min_sep) < np.median(base.min_sep)


def test_turn_error_acts_only_in_turns(nominal):
    sc, plan = nominal  # all direct routes; closest approaches are on straight legs
    off = Dispersions(cross_track_sd=0)
    on = Dispersions(cross_track_sd=0, turn_xtrack_sd=60.0)
    assert np.array_equal(monte_carlo(sc, plan, 50, off).min_sep, monte_carlo(sc, plan, 50, on).min_sep)
    for seed in range(20):
        sc = scenario.generate(seed)
        plan, _ = solve(build_problem(sc), "bnb")
        if any(c is not None and c.geometry.startswith("bulge") for c in plan.chosen.values()):
            break
    a, b = monte_carlo(sc, plan, 50, off), monte_carlo(sc, plan, 50, on)
    assert not np.array_equal(a.min_sep, b.min_sep)


def test_shared_altimeter_bias_does_not_change_separation(nominal):
    sc, plan = nominal
    a = monte_carlo(sc, plan, 100, Dispersions(), seed=2)
    b = monte_carlo(sc, plan, 100, Dispersions(alt_bias_sd=20.0), seed=2)
    assert np.allclose(a.min_sep, b.min_sep)


def test_warning_flags_predictable_misses(nominal):
    sc, plan = nominal
    clean = monte_carlo(sc, plan, 20, Dispersions(wind_sd=0, start_delay_sd=0, cross_track_sd=0, alt_sd=0))
    assert (clean.warning == 0).all()
    r = monte_carlo(sc, plan, 300, Dispersions(wind_sd=4.0))
    assert (r.warning > 0).any() and (r.warning <= plan.T).all()
    assert ((r.warning > 0) <= (r.arrival_err > Dispersions().arrival_tol)).mean() > 0.9


def test_importance_sampling_unbiased(nominal):
    sc, plan = nominal
    d = Dispersions(cross_track_sd=40, alt_sd=15)
    plain = [monte_carlo(sc, plan, 1000, d, seed=s) for s in range(4)]
    widened = [monte_carlo(sc, plan, 1000, dataclasses.replace(d, is_scale=1.3), seed=10 + s) for s in range(4)]

    def est(rs):
        x = np.concatenate([r.weight * r.sep_loss for r in rs])
        return x.mean(), x.std() / np.sqrt(x.size)

    (p, se), (q, sq) = est(plain), est(widened)
    assert p > 0.005 and abs(p - q) < 3 * np.hypot(se, sq)
    assert widened[0].p_sep_loss == pytest.approx(
        float(np.mean(widened[0].weight * widened[0].sep_loss)) / float(widened[0].weight.mean())
    )


def test_is_vars_validated(nominal):
    with pytest.raises(ValueError):
        monte_carlo(*nominal, 5, Dispersions(is_scale=1.3, is_vars=("gain",)))


def test_integral_term_removes_wind_offset(nominal):
    sc, plan = nominal
    windy = Dispersions(wind_sd=1.0, start_delay_sd=0, cross_track_sd=0, alt_sd=0)
    assert monte_carlo(sc, plan, 50, dataclasses.replace(windy, gain_i=0.0), seed=2) == monte_carlo(
        sc, plan, 50, windy, seed=2
    )
    p = monte_carlo(sc, plan, 200, windy, seed=2)
    pi = monte_carlo(sc, plan, 200, dataclasses.replace(windy, gain_i=0.0025), seed=2)
    assert pi.arrival_err.mean() < 0.7 * p.arrival_err.mean()
