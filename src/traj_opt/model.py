"""Core data types. Units: meters, seconds, radians. Frame: local ENU (x east, y north).

Headings use math convention: 0 = +x (east), counter-clockwise positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Params:
    # 4D voxel grid
    dx: float = 25.0
    dz: float = 10.0
    dt: float = 1.0
    # Separation minima (loss of separation needs BOTH horizontal and vertical to be violated)
    sep_h: float = 50.0
    sep_v: float = 30.0
    nav_unc_h: float = 5.0
    nav_unc_v: float = 5.0
    timing_jitter: float = 1.0
    terrain_clearance: float = 50.0
    # Ingress geometry
    ingress_radius: float = 800.0
    ingress_step_deg: float = 30.0
    ingress_per_aircraft: int = 4
    ingress_spread: bool = False  # keep angles evenly spaced around the ring instead of the nearest
    min_sector_deg: float = 45.0  # min approach-angle difference on the same target
    ingress_min_spacing: float = 180.0  # between ingress points of different targets
    r_exempt: float = 250.0  # trajectories end here; converging inside is intended
    # Timing
    schedule_margin: float = 5.0
    speed_margin: float = 0.15  # fraction of (v_max - v_min) kept free at each end of the band for timing control
    sync_tolerance: float = 0.1
    # Candidate families
    alt_step: float = 40.0
    terrain_margin: float = 25.0  # cruise floor above corridor terrain + clearance (covers voxel column conservatism)
    alt_layers: int = 1  # offsets -n..n
    loiter: bool = True
    min_bulge_extra: float = 300.0  # bulges must add at least this much length
    max_bulge_offset: float = 4000.0
    # Selection
    drop_cost: float = 1e4
    max_nodes: int = 200_000
    # Verification
    verify_dt: float = 0.1


@dataclass(frozen=True)
class Aircraft:
    id: int
    x: float
    y: float
    z: float
    heading: float
    v_min: float = 15.0
    v_nom: float = 20.0
    v_max: float = 30.0
    r_min: float = 100.0
    climb_rate: float = 5.0
    z_min: float = 50.0
    z_max: float = 1500.0


@dataclass(frozen=True)
class Target:
    id: str
    x: float
    y: float
    value: float
    aircraft: tuple[int, ...]


@dataclass(frozen=True)
class Hill:
    x: float
    y: float
    h: float
    sigma: float


@dataclass(frozen=True)
class Terrain:
    hills: tuple[Hill, ...] = ()

    def height(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.zeros(np.broadcast(x, y).shape)
        for h in self.hills:
            z = z + h.h * np.exp(-((x - h.x) ** 2 + (y - h.y) ** 2) / (2 * h.sigma**2))
        return z


@dataclass(frozen=True)
class Scenario:
    name: str
    aircraft: tuple[Aircraft, ...]
    targets: tuple[Target, ...]
    terrain: Terrain
    params: Params

    def aircraft_by_id(self) -> dict[int, Aircraft]:
        return {a.id: a for a in self.aircraft}

    def target_of(self, aircraft_id: int) -> Target:
        for t in self.targets:
            if aircraft_id in t.aircraft:
                return t
        raise KeyError(aircraft_id)


@dataclass
class Trajectory:
    t: np.ndarray  # (N,) seconds from plan epoch
    xyz: np.ndarray  # (N, 3)
    t_ingress: float  # time the ingress point is crossed


@dataclass
class Candidate:
    aircraft_id: int
    target_id: str
    ingress_angle: float  # approach direction: ingress point bearing from aimpoint
    ingress_xy: tuple[float, float]
    geometry: str  # direct | bulge_L | bulge_R | loiter_L | loiter_R
    alt_offset: int  # altitude layer index
    speed: float
    length: float  # horizontal length to ingress
    traj: Trajectory
    cost: float = 0.0
    idx: int = -1  # global index, set by the planner

    @property
    def label(self) -> str:
        return (
            f"ac{self.aircraft_id} {self.geometry} ing={np.degrees(self.ingress_angle):.0f}deg "
            f"alt{self.alt_offset:+d} v={self.speed:.1f}"
        )


@dataclass
class Plan:
    solver: str
    T: float
    chosen: dict[int, Candidate | None]  # aircraft id -> candidate (None = dropped)
    cost: float
    nodes: int = 0
    status: str = "CONFLICT_FREE"  # or NO_SAFE_PLAN
    reason: str = ""
    extra: dict = field(default_factory=dict)
