import math

import numpy as np
import pytest

from traj_opt.dubins import Chain, mod2pi, shortest


def _poses(n, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        q0 = (*rng.uniform(-2000, 2000, 2), rng.uniform(0, 2 * math.pi))
        q1 = (*rng.uniform(-2000, 2000, 2), rng.uniform(0, 2 * math.pi))
        yield q0, q1, float(rng.uniform(50, 300))


@pytest.mark.parametrize("q0,q1,r", list(_poses(300)))
def test_endpoints_and_heading(q0, q1, r):
    path = shortest(q0, q1, r)
    end = path.sample(np.array([path.length]))[0]
    assert math.dist(end[:2], q1[:2]) < 1e-6
    assert abs((end[2] - q1[2] + math.pi) % (2 * math.pi) - math.pi) < 1e-6


@pytest.mark.parametrize("q0,q1,r", list(_poses(100, seed=1)))
def test_length_matches_samples_and_curvature_bounded(q0, q1, r):
    path = shortest(q0, q1, r)
    s = np.linspace(0, path.length, 4000)
    pts = path.sample(s)
    arc = np.hypot(*np.diff(pts[:, :2], axis=0).T).sum()
    assert arc == pytest.approx(path.length, rel=1e-3)
    # Heading changes no faster than 1/r per unit length.
    dth = np.abs((np.diff(pts[:, 2]) + math.pi) % (2 * math.pi) - math.pi)
    assert (dth / np.diff(s)).max() <= 1 / r + 1e-6


def test_straight_line_is_shortest():
    path = shortest((0, 0, 0), (1000, 0, 0), 100)
    assert path.length == pytest.approx(1000)


def test_lower_bound_is_euclidean():
    for q0, q1, r in _poses(200, seed=2):
        assert shortest(q0, q1, r).length >= math.dist(q0[:2], q1[:2]) - 1e-6


def test_chain_is_continuous():
    a = shortest((0, 0, 0), (500, 500, math.pi / 2), 100)
    b = shortest(a.end(), (0, 1000, math.pi), 100)
    c = Chain([a, b])
    assert c.length == pytest.approx(a.length + b.length)
    pts = c.sample(np.linspace(0, c.length, 5000))
    assert np.hypot(*np.diff(pts[:, :2], axis=0).T).max() < 2 * c.length / 5000 + 1e-6
    assert mod2pi(pts[-1, 2]) == pytest.approx(math.pi)
