"""Choose one candidate (or drop) per aircraft so no two chosen candidates conflict.

greedy: the production approach. Fixed priority order, first conflict-free candidate, else NO_SAFE_PLAN.
bnb:    joint optimum. Depth-first branch-and-bound that branches on the most-constrained aircraft next
        (fewest unblocked candidates), cheapest candidates first, with an admissible bound (cost so far +
        each remaining aircraft's cheapest still-unblocked option or its drop cost). The incumbent starts
        as greedy-with-drops, so bnb never does worse than greedy.
"""

from __future__ import annotations

import itertools

from .model import Candidate, Plan, Scenario


def priority_order(sc: Scenario) -> list[int]:
    by_id = {t.id: t for t in sc.targets}
    return sorted((a.id for a in sc.aircraft), key=lambda i: (-by_id[sc.target_of(i).id].value, sc.target_of(i).id, i))


def drop_cost(sc: Scenario, ac_id: int) -> float:
    return sc.params.drop_cost * sc.target_of(ac_id).value


def greedy(sc: Scenario, T: float, cands: dict[int, list[Candidate]], conflicts: dict[int, set[int]]) -> Plan:
    chosen: dict[int, Candidate | None] = {}
    blocked: set[int] = set()
    for ac in priority_order(sc):
        pick = next((c for c in cands[ac] if c.idx not in blocked), None)
        if pick is None:
            return Plan(
                "greedy",
                T,
                chosen,
                float("inf"),
                status="NO_SAFE_PLAN",
                reason=f"no conflict-free candidate for ac{ac}",
            )
        chosen[ac] = pick
        blocked |= conflicts[pick.idx]
    return Plan("greedy", T, chosen, sum(c.cost for c in chosen.values()))


def bnb(sc: Scenario, T: float, cands: dict[int, list[Candidate]], conflicts: dict[int, set[int]]) -> Plan:
    order = priority_order(sc)
    drop = {ac: drop_cost(sc, ac) for ac in order}
    block_count: dict[int, int] = {}
    best: dict = {"cost": float("inf"), "chosen": None}
    rank = {ac: i for i, ac in enumerate(order)}
    nodes = 0
    limit = sc.params.max_nodes

    def cheapest_open(ac: int) -> float:
        for c in cands[ac]:  # sorted by cost
            if not block_count.get(c.idx):
                return min(c.cost, drop[ac])
        return drop[ac]

    def n_open(ac: int) -> int:
        return sum(1 for c in cands[ac] if not block_count.get(c.idx))

    def dfs(remaining: list[int], chosen: dict[int, Candidate | None], cost: float) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > limit:
            return
        if not remaining:
            if cost < best["cost"]:
                best["cost"], best["chosen"] = cost, dict(chosen)
            return
        if cost + sum(cheapest_open(ac) for ac in remaining) >= best["cost"]:
            return
        # Most-constrained aircraft first (ties: priority order) -> conflicts surface near the root.
        ac = min(remaining, key=lambda a: (n_open(a), rank[a]))
        rest = [a for a in remaining if a != ac]
        for c in cands[ac]:
            if block_count.get(c.idx):
                continue
            for j in conflicts[c.idx]:
                block_count[j] = block_count.get(j, 0) + 1
            chosen[ac] = c
            dfs(rest, chosen, cost + c.cost)
            for j in conflicts[c.idx]:
                block_count[j] -= 1
        chosen[ac] = None
        dfs(rest, chosen, cost + drop[ac])
        del chosen[ac]

    # Incumbent: greedy-with-drops, so pruning starts immediately.
    taken: set[int] = set()
    incumbent: dict[int, Candidate | None] = {}
    for ac in order:
        incumbent[ac] = next((c for c in cands[ac] if c.idx not in taken and c.cost < drop[ac]), None)
        if incumbent[ac] is not None:
            taken |= conflicts[incumbent[ac].idx]
    best["cost"] = sum(c.cost if c is not None else drop[ac] for ac, c in incumbent.items())
    best["chosen"] = incumbent

    dfs(list(order), {}, 0.0)
    plan = Plan("bnb", T, best["chosen"], best["cost"], nodes=nodes)
    if nodes > limit:
        plan.reason = f"node limit {limit} reached; best plan found so far"
    dropped = [ac for ac, c in plan.chosen.items() if c is None]
    if dropped:
        plan.status = "PARTIAL"
        plan.reason = (plan.reason + "; " if plan.reason else "") + f"dropped {dropped}"
    return plan


def brute_force(sc: Scenario, cands: dict[int, list[Candidate]], conflicts: dict[int, set[int]]) -> float:
    """Exact optimum by enumeration. Test use only."""
    order = priority_order(sc)
    best = float("inf")
    for combo in itertools.product(*[[*cands[ac], None] for ac in order]):
        picked = [c for c in combo if c is not None]
        if any(b.idx in conflicts[a.idx] for a, b in itertools.combinations(picked, 2)):
            continue
        cost = sum(c.cost if c is not None else drop_cost(sc, ac) for ac, c in zip(order, combo, strict=True))
        best = min(best, cost)
    return best
