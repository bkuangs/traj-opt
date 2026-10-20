"""End-to-end pipeline: candidates -> terrain filter -> conflict graph -> solve -> verify."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import select, voxels
from .candidates import build_candidates, common_ingress_time
from .model import Candidate, Plan, Scenario
from .verify import Report, verify


@dataclass
class Problem:
    sc: Scenario
    T: float
    cands: dict[int, list[Candidate]]
    conflicts: dict[int, set[int]]
    stats: dict = field(default_factory=dict)


def build_problem(sc: Scenario) -> Problem:
    t0 = time.perf_counter()
    T = common_ingress_time(sc)
    cands: dict[int, list[Candidate]] = {}
    keys: dict[int, np.ndarray] = {}
    generated = terrain_rejected = 0
    idx = 0
    for ac in sc.aircraft:
        kept = []
        for c in build_candidates(sc, ac, T):
            generated += 1
            k = voxels.inflated_keys(c.traj.t, c.traj.xyz, sc.params)
            if voxels.hits_terrain(k, sc.terrain, sc.params):
                terrain_rejected += 1
                continue
            c.idx = idx
            keys[idx] = k
            idx += 1
            kept.append(c)
        cands[ac.id] = kept
    t1 = time.perf_counter()
    conflicts, info = voxels.conflict_graph(sc, cands, keys)
    t2 = time.perf_counter()
    stats = {
        "candidates_generated": generated,
        "terrain_rejected": terrain_rejected,
        "candidates": idx,
        "conflict_pairs": info["conflict_pairs"],
        "voxels_total": info["voxels_total"],
        "t_candidates_s": t1 - t0,
        "t_conflicts_s": t2 - t1,
    }
    return Problem(sc, T, cands, conflicts, stats)


SOLVERS = {"greedy": select.greedy, "bnb": select.bnb}


def solve(prob: Problem, solver: str) -> tuple[Plan, Report]:
    t0 = time.perf_counter()
    plan = SOLVERS[solver](prob.sc, prob.T, prob.cands, prob.conflicts)
    plan.extra["t_solve_s"] = time.perf_counter() - t0
    report = verify(prob.sc, plan)
    return plan, report
