"""Choose one candidate (or drop) per aircraft so no two chosen candidates conflict.

greedy: the production approach. Fixed priority order, first conflict-free candidate, else NO_SAFE_PLAN.
greedy_with_drops: same, but drops a stuck aircraft and continues (bnb's incumbent; a fair value baseline).
sector_greedy: per target, jointly assign approach angles (all pairs >= min_sector apart, minimum total
        best-candidate cost), then greedy restricted to the assigned angles. A cheap fix for the dominant
        greedy failure (teammates competing for the same approach arc).
rules:  procedural baseline, no search. Each aircraft gets an approach sector (as sector_greedy) and its
        target's altitude layer (one layer per target, by target priority: 0, +1, -1, ...), then the
        cheapest conflict-free route in exactly that sector and layer. Fails rather than relaxing a rule.
        Best with alt_bands=true (shared absolute bands) and alt_step >= 60 m.
rules_bnb: rules, and bnb only when the rules fail.
bnb:    joint optimum. Depth-first branch-and-bound that branches on the most-constrained aircraft next
        (fewest unblocked candidates), cheapest candidates first, with an admissible bound (cost so far +
        each remaining aircraft's cheapest still-unblocked option or its drop cost). The incumbent starts
        as greedy-with-drops, so bnb never does worse than greedy.
"""

from __future__ import annotations

import itertools
import math

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


def greedy_with_drops(
    sc: Scenario, T: float, cands: dict[int, list[Candidate]], conflicts: dict[int, set[int]]
) -> Plan:
    """Greedy that drops a stuck aircraft and keeps going instead of aborting."""
    chosen: dict[int, Candidate | None] = {}
    taken: set[int] = set()
    cost = 0.0
    for ac in priority_order(sc):
        drop = drop_cost(sc, ac)
        chosen[ac] = next((c for c in cands[ac] if c.idx not in taken and c.cost < drop), None)
        if chosen[ac] is None:
            cost += drop
        else:
            taken |= conflicts[chosen[ac].idx]
            cost += chosen[ac].cost
    status = "PARTIAL" if any(c is None for c in chosen.values()) else "CONFLICT_FREE"
    return Plan("greedy_drops", T, chosen, cost, status=status)


def assign_sectors(sc: Scenario, cands: dict[int, list[Candidate]]) -> dict[int, float] | None:
    """Approach angle per aircraft: cheapest per-target combination with all same-target pairs >= min_sector."""
    min_sep = math.radians(sc.params.min_sector_deg) - 1e-9
    out: dict[int, float] = {}
    for tgt in sc.targets:
        ids = list(tgt.aircraft)
        best_at = [{} for _ in ids]  # angle -> cheapest candidate cost at that angle
        for k, ac in enumerate(ids):
            for c in cands[ac]:
                best_at[k][c.ingress_angle] = min(best_at[k].get(c.ingress_angle, math.inf), c.cost)
        best = None
        for combo in itertools.product(*[sorted(b) for b in best_at]):
            if all(
                abs((x - y + math.pi) % (2 * math.pi) - math.pi) >= min_sep for x, y in itertools.combinations(combo, 2)
            ):
                cost = sum(b[a] for b, a in zip(best_at, combo, strict=True))
                if best is None or cost < best[0]:
                    best = (cost, combo)
        if best is None:
            return None
        out.update(zip(ids, best[1], strict=True))
    return out


def sector_greedy(sc: Scenario, T: float, cands: dict[int, list[Candidate]], conflicts: dict[int, set[int]]) -> Plan:
    sectors = assign_sectors(sc, cands)
    if sectors is None:
        return Plan("sector_greedy", T, {}, float("inf"), status="NO_SAFE_PLAN", reason="no valid sector assignment")
    plan = greedy(sc, T, {ac: [c for c in cs if c.ingress_angle == sectors[ac]] for ac, cs in cands.items()}, conflicts)
    plan.solver = "sector_greedy"
    return plan


def rules(sc: Scenario, T: float, cands: dict[int, list[Candidate]], conflicts: dict[int, set[int]]) -> Plan:
    sectors = assign_sectors(sc, cands)
    if sectors is None:
        return Plan("rules", T, {}, float("inf"), status="NO_SAFE_PLAN", reason="no valid sector assignment")
    layers = sorted({c.alt_offset for cs in cands.values() for c in cs}, key=lambda x: (abs(x), -x))
    tgt_rank = {t.id: i for i, t in enumerate(sorted(sc.targets, key=lambda t: (-t.value, t.id)))}
    layer = {a.id: layers[tgt_rank[sc.target_of(a.id).id] % len(layers)] for a in sc.aircraft}
    menu = {
        ac: [c for c in cs if c.ingress_angle == sectors[ac] and c.alt_offset == layer[ac]] for ac, cs in cands.items()
    }
    plan = greedy(sc, T, menu, conflicts)
    plan.solver = "rules"
    return plan


def rules_bnb(sc: Scenario, T: float, cands: dict[int, list[Candidate]], conflicts: dict[int, set[int]]) -> Plan:
    plan = rules(sc, T, cands, conflicts)
    if plan.status != "CONFLICT_FREE":
        plan = bnb(sc, T, cands, conflicts)
        plan.extra["fallback"] = True
    plan.solver = "rules_bnb"
    return plan


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
    incumbent = greedy_with_drops(sc, T, cands, conflicts)
    best["cost"], best["chosen"] = incumbent.cost, incumbent.chosen

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
