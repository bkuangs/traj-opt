"""Flyable candidate trajectories for each aircraft.

Every candidate is built from Dubins paths at radius >= r_min, ends on the ingress heading,
flies at one constant ground speed inside the reduced speed band, and changes altitude no faster
than the climb limit. Candidate = ingress option x geometry x altitude layer.
"""

from __future__ import annotations

import math

import numpy as np

from .dubins import Chain, DubinsPath, shortest
from .model import Aircraft, Candidate, Scenario, Target, Trajectory

GEOMETRY_PENALTY = {"direct": 0.0, "bulge_L": 1.0, "bulge_R": 1.0, "loiter_L": 2.0, "loiter_R": 2.0}


def ingress_options(sc: Scenario, ac: Aircraft, tgt: Target) -> list[tuple[float, tuple[float, float, float], float]]:
    """(approach angle, ingress pose, direct Dubins length), best first, top-K."""
    p = sc.params
    opts = []
    for k in range(int(round(360 / p.ingress_step_deg))):
        ang = math.radians(k * p.ingress_step_deg)
        pose = (tgt.x + p.ingress_radius * math.cos(ang), tgt.y + p.ingress_radius * math.sin(ang), ang + math.pi)
        opts.append((ang, pose, shortest((ac.x, ac.y, ac.heading), pose, ac.r_min).length))
    opts.sort(key=lambda o: (o[2], o[0]))
    return opts[: p.ingress_per_aircraft]


def speed_band(sc: Scenario, ac: Aircraft) -> tuple[float, float]:
    m = sc.params.speed_margin * (ac.v_max - ac.v_min)
    return ac.v_min + m, ac.v_max - m


def common_ingress_time(sc: Scenario) -> float:
    earliest = []
    for ac in sc.aircraft:
        _, v_hi = speed_band(sc, ac)
        earliest.append(min(o[2] for o in ingress_options(sc, ac, sc.target_of(ac.id))) / v_hi)
    return max(earliest) + sc.params.schedule_margin


def _bulge(q0, q1, r: float, side: int, length: float, max_offset: float) -> Chain | None:
    """Dubins through a waypoint offset sideways from the chord midpoint, sized to hit `length`."""
    mx, my = (q0[0] + q1[0]) / 2, (q0[1] + q1[1]) / 2
    chord = math.atan2(q1[1] - q0[1], q1[0] - q0[0])
    nx, ny = -math.sin(chord) * side, math.cos(chord) * side

    def chain(off: float) -> Chain:
        w = (mx + off * nx, my + off * ny, chord)
        return Chain([shortest(q0, w, r), shortest(w, q1, r)])

    grid = np.linspace(0.0, max_offset, 41)
    lens = [chain(o).length for o in grid]
    for i in range(1, len(grid)):
        if lens[i - 1] < length <= lens[i]:
            lo, hi = grid[i - 1], grid[i]
            for _ in range(50):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if chain(mid).length < length else (lo, mid)
            c = chain(hi)
            return c if abs(c.length - length) < 1.0 else None
    return None


def _loiter(q0, q1, r_min: float, side: int, length: float) -> Chain | None:
    """Whole orbits at the start (returning to the start pose), then direct Dubins."""
    direct = shortest(q0, q1, r_min)
    extra = length - direct.length
    n = math.floor(extra / (2 * math.pi * r_min))
    if n < 1:
        return None
    radius = extra / (2 * math.pi * n)
    orbit = DubinsPath(tuple(q0), radius, "LSS" if side > 0 else "RSS", (2 * math.pi * n, 0.0, 0.0))
    return Chain([orbit, direct])


def _terrain_floor(sc: Scenario, chain: Chain, q_ingress) -> float:
    """Lowest altitude that clears terrain along the horizontal track (chain + terminal leg)."""
    p = sc.params
    xy = chain.sample(np.arange(0.0, chain.length, p.dx))[:, :2]
    d = np.arange(0.0, p.ingress_radius - p.r_exempt + p.dx, p.dx)
    term = np.column_stack([q_ingress[0] + d * math.cos(q_ingress[2]), q_ingress[1] + d * math.sin(q_ingress[2])])
    xy = np.vstack([xy, term])
    return float(np.max(sc.terrain.height(xy[:, 0], xy[:, 1]))) + p.terrain_clearance


def _trajectory(
    chain: Chain, q_ingress, tgt: Target, z0: float, z1: float, speed: float, climb: float, sc: Scenario
) -> Trajectory:
    p = sc.params
    term = p.ingress_radius - p.r_exempt
    total = chain.length + term
    h = p.verify_dt
    t = np.arange(0.0, total / speed, h)
    t = np.append(t, total / speed)
    s = t * speed
    xy = np.empty((s.size, 2))
    m = s <= chain.length
    xy[m] = chain.sample(s[m])[:, :2]
    d = s[~m] - chain.length
    xy[~m, 0] = q_ingress[0] + d * math.cos(q_ingress[2])
    xy[~m, 1] = q_ingress[1] + d * math.sin(q_ingress[2])
    dz = z1 - z0
    z = z0 + math.copysign(1.0, dz) * np.minimum(abs(dz), s * climb / speed)
    return Trajectory(t, np.column_stack([xy, z]), chain.length / speed)


def build_candidates(sc: Scenario, ac: Aircraft, T: float) -> list[Candidate]:
    p = sc.params
    tgt = sc.target_of(ac.id)
    v_lo, v_hi = speed_band(sc, ac)
    q0 = (ac.x, ac.y, ac.heading)
    opts = ingress_options(sc, ac, tgt)
    best_direct = opts[0][2]
    out: list[Candidate] = []
    for rank, (ang, q1, d_len) in enumerate(opts):
        geoms: list[tuple[str, Chain]] = []
        if v_lo * T <= d_len <= v_hi * T:
            geoms.append(("direct", Chain([shortest(q0, q1, ac.r_min)])))
        pad_len = max(min(max(ac.v_nom * T, v_lo * T), v_hi * T), d_len + p.min_bulge_extra)
        if pad_len <= v_hi * T:
            for name, side in (("L", 1), ("R", -1)):
                c = _bulge(q0, q1, ac.r_min, side, pad_len, p.max_bulge_offset)
                if c is not None:
                    geoms.append((f"bulge_{name}", c))
                c = _loiter(q0, q1, ac.r_min, side, pad_len)
                if c is not None:
                    geoms.append((f"loiter_{name}", c))
        for geom, chain in geoms:
            speed = chain.length / T
            z_base = max(ac.z, _terrain_floor(sc, chain, q1) + p.terrain_margin)
            for layer in range(-p.alt_layers, p.alt_layers + 1):
                z1 = z_base + layer * p.alt_step
                if not ac.z_min <= z1 <= ac.z_max:
                    continue
                traj = _trajectory(chain, q1, tgt, ac.z, z1, speed, 0.8 * ac.climb_rate, sc)
                cost = tgt.value * (
                    GEOMETRY_PENALTY[geom]
                    + 0.5 * abs(layer)
                    + 0.3 * rank
                    + (chain.length - best_direct) / 1000.0
                    + 2.0 * abs(speed - ac.v_nom) / ac.v_nom
                )
                out.append(Candidate(ac.id, tgt.id, ang, (q1[0], q1[1]), geom, layer, speed, chain.length, traj, cost))
    out.sort(key=lambda c: (c.cost, c.label))
    if len(out) > 64:
        raise ValueError("at most 64 candidates per aircraft (bitmask conflict index)")
    return out
