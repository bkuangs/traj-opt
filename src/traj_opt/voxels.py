"""Inflated 4D voxel occupancy, static terrain voxels, and the pairwise candidate conflict graph.

Each candidate's swept trajectory is rasterized into (ix, iy, iz, it) cells and inflated by half the
required separation plus its own uncertainty, as in the production SweptVolumeRasterizer. Two candidates
conflict when their inflated cell sets intersect. Because every cell is inflated by >= 1 in each axis,
crossings and adjacent-cell swaps within one time step are covered by the overlap test.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from .model import Candidate, Params, Scenario, Terrain

_OFF = 1 << 15
_MASK = (1 << 16) - 1


def inflation(p: Params) -> tuple[int, int, int]:
    nxy = math.ceil((p.sep_h / 2 + p.nav_unc_h) / p.dx)
    nz = math.ceil((p.sep_v / 2 + p.nav_unc_v) / p.dz)
    nt = math.ceil(p.timing_jitter / p.dt)
    return nxy, nz, nt


def base_cells(t: np.ndarray, xyz: np.ndarray, p: Params) -> np.ndarray:
    """Unique (ix, iy, iz, it) cells touched by a finely sampled trajectory."""
    cells = np.column_stack(
        [
            np.floor(xyz[:, 0] / p.dx),
            np.floor(xyz[:, 1] / p.dx),
            np.floor(xyz[:, 2] / p.dz),
            np.floor(t / p.dt),
        ]
    ).astype(np.int64)
    return np.unique(cells, axis=0)


def _encode(c: np.ndarray) -> np.ndarray:
    c = c + _OFF
    return (c[:, 0] << 48) | (c[:, 1] << 32) | (c[:, 2] << 16) | c[:, 3]


def decode(k: np.ndarray) -> np.ndarray:
    return np.column_stack([(k >> s) & _MASK for s in (48, 32, 16, 0)]) - _OFF


def inflated_keys(t: np.ndarray, xyz: np.ndarray, p: Params) -> np.ndarray:
    _check_sampling(t, xyz, p)
    nxy, nz, nt = inflation(p)
    keys = _encode(base_cells(t, xyz, p))
    # Box dilation is separable: dilate one axis at a time, deduplicating in between.
    for shift, n in ((0, nt), (16, nz), (32, nxy), (48, nxy)):
        step = np.int64(1) << shift
        keys = np.unique((keys[:, None] + np.arange(-n, n + 1, dtype=np.int64)[None, :] * step).ravel())
    return keys


def _check_sampling(t: np.ndarray, xyz: np.ndarray, p: Params) -> None:
    """Samples must not skip a cell, or the swept volume has holes."""
    if len(t) < 2:
        return
    step = np.abs(np.diff(xyz, axis=0)).max(axis=0)
    if step[0] > p.dx or step[1] > p.dx or step[2] > p.dz or np.diff(t).max() > p.dt:
        raise ValueError("trajectory sampled too coarsely for the voxel grid")


def terrain_floor(terrain: Terrain, p: Params, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Lowest free iz per (ix, iy): cells below max-terrain-in-cell + clearance are blocked."""
    xs = np.stack([ix * p.dx, (ix + 0.5) * p.dx, (ix + 1) * p.dx], axis=-1)
    ys = np.stack([iy * p.dx, (iy + 0.5) * p.dx, (iy + 1) * p.dx], axis=-1)
    hmax = terrain.height(xs[:, :, None], ys[:, None, :]).reshape(len(ix), -1).max(axis=1)
    return np.ceil((hmax + p.terrain_clearance) / p.dz).astype(np.int64)


def hits_terrain(keys: np.ndarray, terrain: Terrain, p: Params) -> bool:
    """Conservative: any inflated cell inside the terrain+clearance volume."""
    if not terrain.hills:
        return False
    cols, inv = np.unique(keys >> 32, return_inverse=True)
    ix = ((cols >> 16) & _MASK) - _OFF
    iy = (cols & _MASK) - _OFF
    iz = ((keys >> 16) & _MASK) - _OFF
    return bool((iz < terrain_floor(terrain, p, ix, iy)[inv]).any())


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def ingress_conflict(a: Candidate, b: Candidate, p: Params) -> bool:
    if a.target_id == b.target_id:
        return _angle_diff(a.ingress_angle, b.ingress_angle) < math.radians(p.min_sector_deg) - 1e-9
    return math.dist(a.ingress_xy, b.ingress_xy) < p.ingress_min_spacing


def conflict_graph(
    sc: Scenario, cands: dict[int, list[Candidate]], keys: dict[int, np.ndarray]
) -> tuple[dict[int, set[int]], dict]:
    """Global candidate index -> set of conflicting global indices (other aircraft only).

    `keys` maps candidate index -> inflated voxel keys. Index per aircraft: sorted unique keys with a
    uint64 bitmask of which of its candidates occupy each key.
    """
    p = sc.params
    index = {}
    for ac_id, cs in cands.items():
        if not cs:
            continue
        allk = np.concatenate([keys[c.idx] for c in cs])
        bits = np.concatenate([np.full(len(keys[c.idx]), 1 << i, dtype=np.uint64) for i, c in enumerate(cs)])
        order = np.argsort(allk, kind="stable")
        allk, bits = allk[order], bits[order]
        uniq, start = np.unique(allk, return_index=True)
        index[ac_id] = (uniq, np.bitwise_or.reduceat(bits, start))

    conflicts: dict[int, set[int]] = {c.idx: set() for cs in cands.values() for c in cs}
    n_pairs = 0
    for ai, aj in combinations(sorted(index), 2):
        uniq, masks = index[aj]
        for a in cands[ai]:
            pos = np.searchsorted(uniq, keys[a.idx])
            pos = np.minimum(pos, len(uniq) - 1)
            hit = uniq[pos] == keys[a.idx]
            mask = int(np.bitwise_or.reduce(masks[pos[hit]])) if hit.any() else 0
            for j, b in enumerate(cands[aj]):
                if (mask >> j) & 1 or ingress_conflict(a, b, p):
                    conflicts[a.idx].add(b.idx)
                    conflicts[b.idx].add(a.idx)
                    n_pairs += 1
    stats = {"voxels_total": int(sum(len(k) for k in keys.values())), "conflict_pairs": n_pairs}
    return conflicts, stats
