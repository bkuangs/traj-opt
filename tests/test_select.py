import itertools

import numpy as np
import pytest

from traj_opt.model import Aircraft, Candidate, Params, Scenario, Target, Terrain
from traj_opt.select import bnb, brute_force, greedy, greedy_with_drops


def random_instance(seed, n_ac=4, max_cands=4, density=0.3):
    rng = np.random.default_rng(seed)
    acs = tuple(Aircraft(i, 0.0, 0.0, 300.0, 0.0) for i in range(1, n_ac + 1))
    tgts = (
        Target("A", 0, 0, 5.0, tuple(a.id for a in acs[: n_ac // 2])),
        Target("B", 0, 0, 2.0, tuple(a.id for a in acs[n_ac // 2 :])),
    )
    sc = Scenario("synthetic", acs, tgts, Terrain(), Params(drop_cost=50.0))
    cands, idx = {}, 0
    for a in acs:
        k = int(rng.integers(1, max_cands + 1))
        costs = sorted(rng.uniform(1, 20, k))
        cands[a.id] = []
        for c in costs:
            cands[a.id].append(
                Candidate(a.id, sc.target_of(a.id).id, 0.0, (0, 0), "direct", 0, 20.0, 0.0, None, float(c), idx)
            )
            idx += 1
    conflicts = {i: set() for i in range(idx)}
    owner = {c.idx: a for a, cs in cands.items() for c in cs}
    for i, j in itertools.combinations(range(idx), 2):
        if owner[i] != owner[j] and rng.random() < density:
            conflicts[i].add(j)
            conflicts[j].add(i)
    return sc, cands, conflicts


def chosen_is_feasible(plan, conflicts):
    picked = [c for c in plan.chosen.values() if c is not None]
    return all(b.idx not in conflicts[a.idx] for a, b in itertools.combinations(picked, 2))


@pytest.mark.parametrize("seed", range(300))
def test_bnb_matches_brute_force(seed):
    sc, cands, conflicts = random_instance(seed, n_ac=int(2 + seed % 4), density=0.15 + 0.5 * (seed % 3) / 2)
    plan = bnb(sc, 100.0, cands, conflicts)
    assert plan.cost == pytest.approx(brute_force(sc, cands, conflicts))
    assert chosen_is_feasible(plan, conflicts)
    assert set(plan.chosen) == {a.id for a in sc.aircraft}


@pytest.mark.parametrize("seed", range(300))
def test_bnb_never_worse_than_greedy(seed):
    sc, cands, conflicts = random_instance(seed, n_ac=5, density=0.3)
    g, b = greedy(sc, 100.0, cands, conflicts), bnb(sc, 100.0, cands, conflicts)
    if g.status == "CONFLICT_FREE":
        assert chosen_is_feasible(g, conflicts)
        assert b.status == "CONFLICT_FREE"
        assert b.cost <= g.cost + 1e-9


@pytest.mark.parametrize("seed", range(100))
def test_greedy_with_drops(seed):
    sc, cands, conflicts = random_instance(seed, n_ac=5, density=0.3)
    g, gd, b = (f(sc, 100.0, cands, conflicts) for f in (greedy, greedy_with_drops, bnb))
    assert chosen_is_feasible(gd, conflicts)
    assert set(gd.chosen) == {a.id for a in sc.aircraft}
    assert b.cost <= gd.cost + 1e-9
    if g.status == "CONFLICT_FREE":
        assert gd.chosen == g.chosen and gd.status == "CONFLICT_FREE"
