"""Monte Carlo robustness check: replay a plan under execution errors.

Each trial perturbs every chosen aircraft with a constant wind (ground-speed bias), a start-time error,
a constant cross-track offset, and a constant altitude offset. Each aircraft flies its planned path and
tracks its planned schedule with a proportional speed controller, saturated at [v_min, v_max] airspeed.
That saturation is where a thin speed margin shows up: an aircraft at its limit cannot recover lost time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .model import Plan, Scenario


@dataclass(frozen=True)
class Dispersions:
    wind_sd: float = 1.0  # m/s, along-track ground-speed bias
    start_delay_sd: float = 3.0  # s, schedule error at plan start (+ = late)
    cross_track_sd: float = 10.0  # m
    alt_sd: float = 5.0  # m
    gain: float = 0.1  # 1/s, schedule-tracking speed controller
    arrival_tol: float = 2.0  # s, allowed |ingress time error|
    dt: float = 0.2  # s, simulation step


@dataclass
class Robustness:
    trials: int
    p_sep_loss: float  # P(any pair separation ratio < 1)
    p_terrain: float  # P(any aircraft below terrain + clearance)
    p_arrival_miss: float  # P(any aircraft misses ingress time by > arrival_tol)
    arrival_err_p95: float  # s, 95th percentile of the worst |ingress time error| per trial
    min_sep_p05: float  # 5th percentile of the worst separation ratio per trial
    pair_loss: dict[tuple[int, int], float] = field(default_factory=dict)
    # P(aircraft arrives within arrival_tol with no terrain or separation violation), per aircraft id
    p_deliver: dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        worst = max(self.pair_loss.items(), key=lambda kv: kv[1], default=(None, 0.0))
        s = (
            f"robustness ({self.trials} trials): P(sep loss)={self.p_sep_loss:.1%}  "
            f"P(terrain)={self.p_terrain:.1%}  P(arrival miss)={self.p_arrival_miss:.1%}  "
            f"p95 |arrival err|={self.arrival_err_p95:.1f}s  p05 min sep ratio={self.min_sep_p05:.2f}"
        )
        if worst[1] > 0:
            s += f"  worst pair ac{worst[0][0]}-ac{worst[0][1]} ({worst[1]:.1%})"
        return s


def monte_carlo(sc: Scenario, plan: Plan, trials: int = 200, disp: Dispersions | None = None, seed: int = 0):
    disp = disp or Dispersions()
    p = sc.params
    acs = sc.aircraft_by_id()
    chosen = sorted((k, c) for k, c in (plan.chosen or {}).items() if c is not None)
    if not chosen:
        return None
    rng = np.random.default_rng(seed)
    n = len(chosen)
    shape = (trials, n)
    wind = rng.normal(0.0, disp.wind_sd, shape)
    delay = rng.normal(0.0, disp.start_delay_sd, shape)
    xtrack = rng.normal(0.0, disp.cross_track_sd, shape)
    dalt = rng.normal(0.0, disp.alt_sd, shape)

    v_plan = np.array([c.speed for _, c in chosen])
    v_min = np.array([acs[k].v_min for k, _ in chosen])
    v_max = np.array([acs[k].v_max for k, _ in chosen])
    l_end = np.array([c.speed * c.traj.t[-1] for _, c in chosen])
    l_ing = np.array([c.length for _, c in chosen])

    # Schedule tracking along the planned path (arc length s).
    t = np.arange(0.0, 1.5 * max(c.traj.t[-1] for _, c in chosen), disp.dt)
    s = np.empty((t.size, trials, n))
    s[0] = -v_plan * delay
    for k in range(1, t.size):
        err = v_plan * t[k - 1] - s[k - 1]
        air = np.clip(v_plan + disp.gain * err, v_min, v_max)
        s[k] = s[k - 1] + (air + wind) * disp.dt

    # Ingress time error: first crossing of the ingress arc length.
    crossed = s >= l_ing
    hit = crossed.any(axis=0)
    t_ing = np.where(hit, t[crossed.argmax(axis=0)], np.inf)
    arrival_err = np.abs(t_ing - plan.T).max(axis=1)

    # Positions (with offsets) while each aircraft is still inside its planned horizon.
    active = s < l_end
    xyz = np.empty((t.size, trials, n, 3))
    for i, (_, c) in enumerate(chosen):
        si = c.speed * c.traj.t
        pts = c.traj.xyz
        tan = np.gradient(pts[:, :2], axis=0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True)
        q = s[:, :, i]
        x = np.interp(q, si, pts[:, 0]) + np.minimum(q, 0) * tan[0, 0]
        y = np.interp(q, si, pts[:, 1]) + np.minimum(q, 0) * tan[0, 1]
        tx, ty = np.interp(q, si, tan[:, 0]), np.interp(q, si, tan[:, 1])
        nrm = np.hypot(tx, ty)
        xyz[:, :, i, 0] = x - xtrack[:, i] * ty / nrm
        xyz[:, :, i, 1] = y + xtrack[:, i] * tx / nrm
        xyz[:, :, i, 2] = np.interp(q, si, pts[:, 2]) + dalt[:, i]

    clearance = xyz[..., 2] - sc.terrain.height(xyz[..., 0], xyz[..., 1])
    terrain_bad_ac = np.where(active, clearance < p.terrain_clearance, False).any(axis=0)  # (trials, n)
    terrain_bad = terrain_bad_ac.any(axis=1)
    failed = terrain_bad_ac | (np.abs(t_ing - plan.T) > disp.arrival_tol)

    worst = np.full(trials, np.inf)
    pair_loss = {}
    for (i, (a, _)), (j, (b, _)) in combinations(enumerate(chosen), 2):
        d = xyz[:, :, i] - xyz[:, :, j]
        ratio = np.maximum(np.hypot(d[..., 0], d[..., 1]) / p.sep_h, np.abs(d[..., 2]) / p.sep_v)
        m = np.where(active[:, :, i] & active[:, :, j], ratio, np.inf).min(axis=0)
        worst = np.minimum(worst, m)
        failed[:, i] |= m < 1.0
        failed[:, j] |= m < 1.0
        pair_loss[(a, b)] = float((m < 1.0).mean())

    return Robustness(
        trials=trials,
        p_sep_loss=float((worst < 1.0).mean()),
        p_terrain=float(terrain_bad.mean()),
        p_arrival_miss=float((arrival_err > disp.arrival_tol).mean()),
        arrival_err_p95=float(np.percentile(arrival_err, 95)),
        min_sep_p05=float(np.percentile(worst, 5)),
        pair_loss=pair_loss,
        p_deliver={a: float(1.0 - failed[:, i].mean()) for i, (a, _) in enumerate(chosen)},
    )
