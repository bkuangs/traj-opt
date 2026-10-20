import numpy as np
import pytest

from traj_opt.model import Hill, Params, Terrain
from traj_opt.voxels import hits_terrain, inflated_keys

P = Params()


def straight(p0, p1, t0=0.0, duration=60.0, dt=0.1):
    t = np.arange(0.0, duration + 1e-9, dt)
    xyz = np.asarray(p0, float) + (np.asarray(p1, float) - np.asarray(p0, float)) * (t / duration)[:, None]
    return t + t0, xyz


def conflict(a, b):
    return np.intersect1d(inflated_keys(*a, P), inflated_keys(*b, P)).size > 0


def test_head_on_conflicts():
    assert conflict(straight((0, 0, 300), (1200, 0, 300)), straight((1200, 0, 300), (0, 0, 300)))


def test_crossing_same_time_conflicts():
    assert conflict(straight((0, 0, 300), (1200, 0, 300)), straight((600, -600, 300), (600, 600, 300)))


def test_crossing_different_time_is_free():
    a = straight((0, 0, 300), (1200, 0, 300))
    b = straight((600, -600, 300), (600, 600, 300), t0=30.0)  # crosses the point 30 s after a left it
    assert not conflict(a, b)


def test_parallel_tracks():
    a = straight((0, 0, 300), (1200, 0, 300))
    assert conflict(a, straight((0, 40, 300), (1200, 40, 300)))  # 40 m < 50 m
    assert not conflict(a, straight((0, 200, 300), (1200, 200, 300)))


def test_vertical_layers():
    a = straight((0, 0, 300), (1200, 0, 300))
    assert conflict(a, straight((1200, 0, 320), (0, 0, 320)))  # 20 m < 30 m
    assert not conflict(a, straight((1200, 0, 400), (0, 0, 400)))


def test_trailing_within_timing_jitter_conflicts():
    a = straight((0, 0, 300), (1200, 0, 300))
    b = straight((0, 0, 300), (1200, 0, 300), t0=1.0)  # same path, 1 s behind (20 m)
    assert conflict(a, b)


def test_coarse_sampling_rejected():
    with pytest.raises(ValueError):
        inflated_keys(*straight((0, 0, 300), (1200, 0, 300), dt=5.0), P)


def test_no_false_negatives_random():
    """Voxel test must flag every pair the continuous check says loses separation (it may over-flag)."""
    rng = np.random.default_rng(0)
    lost = flagged_lost = flagged_safe = safe = 0
    for _ in range(400):
        a0, a1, b0, b1 = (np.r_[rng.uniform(0, 800, 2), rng.uniform(280, 340)] for _ in range(4))
        dur = 40.0
        a, b = straight(a0, a1, duration=dur), straight(b0, b1, duration=dur)
        h = np.hypot(*(a[1][:, :2] - b[1][:, :2]).T)
        v = np.abs(a[1][:, 2] - b[1][:, 2])
        loses = bool((np.maximum(h / P.sep_h, v / P.sep_v) < 1).any())
        hit = conflict(a, b)
        if loses:
            lost += 1
            flagged_lost += hit
        else:
            safe += 1
            flagged_safe += hit
    assert lost > 20 and flagged_lost == lost


def test_terrain_hit():
    terrain = Terrain((Hill(600, 0, 300, 300),))
    assert hits_terrain(inflated_keys(*straight((0, 0, 300), (1200, 0, 300)), P), terrain, P)
    assert not hits_terrain(inflated_keys(*straight((0, 0, 450), (1200, 0, 450)), P), terrain, P)
