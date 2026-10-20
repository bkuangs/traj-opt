import numpy as np
import pytest

from traj_opt.fuel import LIFT_TO_DRAG, candidate_fuel, power_ratio
from traj_opt.model import Candidate, Trajectory


def _cand(speed, z):
    t = np.linspace(0.0, 100.0, len(z))
    xyz = np.column_stack([t * speed, np.zeros_like(t), z])
    return Candidate(0, "A", 0.0, (0.0, 0.0), "direct", 0, speed, 100.0 * speed, Trajectory(t, xyz, 100.0))


def test_power_ratio_shape():
    assert power_ratio(20.0, 20.0) == pytest.approx(1.0)
    v = np.linspace(10, 30, 2001)
    assert v[np.argmin(power_ratio(v, 20.0))] == pytest.approx(20.0 / 3**0.25, abs=0.01)  # min-power speed


def test_candidate_fuel_level_and_climb():
    assert candidate_fuel(_cand(20.0, np.full(11, 100.0)), 20.0) == pytest.approx(100.0)
    climb = _cand(20.0, np.linspace(100.0, 140.0, 11))
    assert candidate_fuel(climb, 20.0) == pytest.approx(100.0 + 40.0 * LIFT_TO_DRAG / 20.0)
    descent = _cand(20.0, np.linspace(140.0, 100.0, 11))
    assert candidate_fuel(descent, 20.0) == pytest.approx(100.0)
