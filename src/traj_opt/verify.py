"""Independent plan verification. Uses continuous geometry and the exact terrain, never the voxel grid.

A failure here on a plan the planner accepted is a planner bug.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .model import Candidate, Plan, Scenario


@dataclass
class Report:
    ok: bool = True
    violations: list[str] = field(default_factory=list)
    min_sep: dict[tuple[int, int], float] = field(default_factory=dict)  # min over time of max(h/sep_h, v/sep_v)
    min_terrain_clearance: dict[int, float] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.violations.append(msg)


def separation_ratio(a: Candidate, b: Candidate, sc: Scenario, dt: float | None = None):
    """Times and max(h/sep_h, v/sep_v) while both are airborne. Ratio < 1 means loss of separation."""
    p = sc.params
    dt = dt or p.verify_dt
    t_end = min(a.traj.t[-1], b.traj.t[-1])
    t = np.arange(0.0, t_end, dt)
    pa = np.column_stack([np.interp(t, a.traj.t, a.traj.xyz[:, i]) for i in range(3)])
    pb = np.column_stack([np.interp(t, b.traj.t, b.traj.xyz[:, i]) for i in range(3)])
    h = np.hypot(pa[:, 0] - pb[:, 0], pa[:, 1] - pb[:, 1])
    v = np.abs(pa[:, 2] - pb[:, 2])
    return t, np.maximum(h / p.sep_h, v / p.sep_v)


def verify(sc: Scenario, plan: Plan) -> Report:
    p = sc.params
    rep = Report()
    acs = sc.aircraft_by_id()
    chosen = {k: c for k, c in plan.chosen.items() if c is not None}

    for ac_id, c in chosen.items():
        ac = acs[ac_id]
        t, xyz = c.traj.t, c.traj.xyz
        if abs(c.traj.t_ingress - plan.T) > p.sync_tolerance:
            rep.fail(f"ac{ac_id}: ingress at {c.traj.t_ingress:.2f}s, common time {plan.T:.2f}s")
        if math.dist(xyz[0, :2], (ac.x, ac.y)) > 1e-6 or abs(xyz[0, 2] - ac.z) > 1e-6:
            rep.fail(f"ac{ac_id}: trajectory does not start at the aircraft")
        d = np.diff(xyz, axis=0)
        dts = np.diff(t)
        ok = dts > 1e-9
        vh = np.hypot(d[ok, 0], d[ok, 1]) / dts[ok]
        if vh.min() < ac.v_min - 1e-6 or vh.max() > ac.v_max + 1e-6:
            rep.fail(f"ac{ac_id}: speed {vh.min():.2f}..{vh.max():.2f} outside [{ac.v_min}, {ac.v_max}]")
        vz = np.abs(d[ok, 2]) / dts[ok]
        if vz.max() > ac.climb_rate + 1e-6:
            rep.fail(f"ac{ac_id}: climb rate {vz.max():.2f} > {ac.climb_rate}")
        r = _min_turn_radius(xyz[:, :2])
        if r < ac.r_min * 0.98:
            rep.fail(f"ac{ac_id}: turn radius {r:.1f} < {ac.r_min}")
        clearance = float((xyz[:, 2] - sc.terrain.height(xyz[:, 0], xyz[:, 1])).min())
        rep.min_terrain_clearance[ac_id] = clearance
        if clearance < p.terrain_clearance:
            rep.fail(f"ac{ac_id}: terrain clearance {clearance:.1f} < {p.terrain_clearance}")
        if xyz[:, 2].min() < ac.z_min - 1e-6 or xyz[:, 2].max() > ac.z_max + 1e-6:
            rep.fail(f"ac{ac_id}: altitude outside bounds")

    for (i, a), (j, b) in combinations(sorted(chosen.items()), 2):
        _, ratio = separation_ratio(a, b, sc)
        m = float(ratio.min()) if ratio.size else float("inf")
        rep.min_sep[(i, j)] = m
        if m < 1.0:
            rep.fail(f"ac{i}-ac{j}: loss of separation (ratio {m:.2f})")
        if a.target_id == b.target_id:
            diff = abs((a.ingress_angle - b.ingress_angle + math.pi) % (2 * math.pi) - math.pi)
            if diff < math.radians(p.min_sector_deg) - 1e-9:
                rep.fail(f"ac{i}-ac{j}: approach sectors {math.degrees(diff):.0f}deg apart")
    return rep


def _min_turn_radius(xy: np.ndarray) -> float:
    """Smallest circumradius over consecutive points thinned to ~5 m spacing."""
    seg = np.hypot(*np.diff(xy, axis=0).T)
    keep = np.concatenate([[True], np.cumsum(seg) // 5.0 != np.cumsum(np.r_[0, seg[:-1]]) // 5.0])
    pts = xy[keep]
    if len(pts) < 3:
        return float("inf")
    a, b, c = pts[:-2], pts[1:-1], pts[2:]
    ab, bc, ca = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1))
    cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ab * bc * ca / (2 * cross)
    r = r[np.isfinite(r)]
    return float(r.min()) if r.size else float("inf")
