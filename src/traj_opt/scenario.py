"""Scenario JSON loading and seeded random generation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from .model import Aircraft, Hill, Params, Scenario, Target, Terrain


def load(path: str | Path) -> Scenario:
    d = json.loads(Path(path).read_text())
    return from_dict(d, name=d.get("name", Path(path).stem))


def from_dict(d: dict, name: str = "scenario") -> Scenario:
    aircraft = []
    for a in d["aircraft"]:
        a = dict(a)
        a["heading"] = math.radians(a.pop("heading_deg"))
        aircraft.append(Aircraft(**a))
    targets = [Target(t["id"], t["x"], t["y"], t["value"], tuple(t["aircraft"])) for t in d["targets"]]
    terrain = Terrain(tuple(Hill(**h) for h in d.get("terrain", {}).get("hills", [])))
    known = {f.name for f in fields(Params)}
    overrides = d.get("params", {})
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"unknown params: {sorted(unknown)}")
    return Scenario(name, tuple(aircraft), tuple(targets), terrain, Params(**overrides))


def to_dict(sc: Scenario) -> dict:
    aircraft = []
    for a in sc.aircraft:
        d = asdict(a)
        d["heading_deg"] = round(math.degrees(d.pop("heading")), 6)
        aircraft.append(d)
    defaults = Params()
    return {
        "name": sc.name,
        "aircraft": aircraft,
        "targets": [
            {"id": t.id, "x": t.x, "y": t.y, "value": t.value, "aircraft": list(t.aircraft)} for t in sc.targets
        ],
        "terrain": {"hills": [asdict(h) for h in sc.terrain.hills]},
        "params": {
            f.name: getattr(sc.params, f.name)
            for f in fields(Params)
            if getattr(sc.params, f.name) != getattr(defaults, f.name)
        },
    }


def save(sc: Scenario, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_dict(sc), indent=2) + "\n")


def generate(
    seed: int,
    n_targets: int = 2,
    per_target: int = 3,
    spawn_radius: tuple[float, float] = (3000.0, 6000.0),
    n_hills: int = 6,
    params: Params | None = None,
    target_spread: float = 1500.0,
) -> Scenario:
    """Random scenario: targets near the origin, aircraft spawned on an annulus, Gaussian hills."""
    rng = np.random.default_rng(seed)
    hills = tuple(
        Hill(
            float(rng.uniform(-6000, 6000)),
            float(rng.uniform(-6000, 6000)),
            float(rng.uniform(80, 280)),
            float(rng.uniform(400, 1200)),
        )
        for _ in range(n_hills)
    )
    terrain = Terrain(hills)
    p = params or Params()
    targets, aircraft = [], []
    ac_id = 1
    for k in range(n_targets):
        tx, ty = rng.uniform(-target_spread, target_spread, 2)
        ids = []
        for _ in range(per_target):
            r = rng.uniform(*spawn_radius)
            ang = rng.uniform(0, 2 * math.pi)
            x, y = tx + r * math.cos(ang), ty + r * math.sin(ang)
            heading = math.atan2(ty - y, tx - x) + rng.uniform(-1.5, 1.5)
            heading = math.radians(round(math.degrees(heading), 6))  # exactly representable in the JSON format
            z = float(rng.uniform(200, 400))
            z = max(z, float(terrain.height(x, y)) + p.terrain_clearance + 60.0)
            aircraft.append(Aircraft(ac_id, float(x), float(y), round(z, 1), float(heading)))
            ids.append(ac_id)
            ac_id += 1
        targets.append(Target(chr(ord("A") + k), float(tx), float(ty), float(rng.integers(1, 11)), tuple(ids)))
    return Scenario(f"random-{seed}", tuple(aircraft), tuple(targets), terrain, p)
