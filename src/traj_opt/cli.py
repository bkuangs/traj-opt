"""Command line: run a scenario, benchmark solvers over random seeds, or save a generated scenario."""

from __future__ import annotations

import argparse
import dataclasses
import statistics
from concurrent.futures import ProcessPoolExecutor

from . import scenario as scn
from .fuel import plan_fuel
from .model import Params
from .planner import build_problem, solve
from .robust import monte_carlo


def _load(args) -> scn.Scenario:
    if args.scenario:
        return scn.load(args.scenario)
    return scn.generate(args.seed, n_targets=args.targets, per_target=args.per_target, target_spread=args.spread)


def _summary(prob, plan, rep) -> str:
    lines = [f"[{plan.solver}] {plan.status} cost={plan.cost:.2f} nodes={plan.nodes} {plan.reason}".rstrip()]
    for ac_id, c in sorted((plan.chosen or {}).items()):
        lines.append(f"  ac{ac_id}: " + (c.label if c else "DROPPED"))
    lines.append("  verify: " + ("OK" if rep.ok else "FAILED " + "; ".join(rep.violations)))
    return "\n".join(lines)


def cmd_run(args) -> None:
    sc = _load(args)
    prob = build_problem(sc)
    print(f"{sc.name}: T={prob.T:.1f}s", {k: round(v, 2) if isinstance(v, float) else v for k, v in prob.stats.items()})
    solvers = ["greedy", "bnb"] if args.solver == "both" else [args.solver]
    results = [(s, *solve(prob, s)) for s in solvers]
    for _, plan, rep in results:
        print(_summary(prob, plan, rep))
        if args.mc and plan.status != "NO_SAFE_PLAN":
            print("  " + monte_carlo(sc, plan, trials=args.mc).summary())
    if args.no_show and not (args.gif or args.png):
        return
    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    from .viz import show

    for s, plan, rep in results:
        suffix = f"-{s}" if len(results) > 1 else ""
        fig = show(
            prob, plan, rep, gif=args.gif and args.gif.replace(".gif", f"{suffix}.gif"), interactive=not args.no_show
        )
        if args.png:
            fig.savefig(args.png.replace(".png", f"{suffix}.png"), dpi=110)


def _value(sc: scn.Scenario, plan, rob=None) -> float:
    """Fraction of total target value delivered. Each aircraft carries value / (aircraft on its target);
    with a Monte Carlo result, weighted by that aircraft's delivery probability."""
    got = 0.0
    for a, c in (plan.chosen or {}).items():
        if c is not None:
            tgt = sc.target_of(a)
            got += tgt.value / len(tgt.aircraft) * (rob.p_deliver[a] if rob else 1.0)
    return got / sum(t.value for t in sc.targets)


def _params(overrides: list[str] | None) -> Params:
    """Params from KEY=VALUE strings, each value parsed with the field's default type."""
    defaults = Params()
    kw = {}
    for item in overrides or []:
        key, val = item.split("=", 1)
        typ = type(getattr(defaults, key))
        kw[key] = val.lower() in ("1", "true", "yes") if typ is bool else typ(val)
    return dataclasses.replace(defaults, **kw)


def _bench_one(
    seed: int, mc: int = 0, targets: int = 2, per_target: int = 3, spread: float = 1500.0, params: Params | None = None
) -> dict:
    sc = scn.generate(seed, n_targets=targets, per_target=per_target, target_spread=spread, params=params)
    prob = build_problem(sc)
    row = {"seed": seed, "t_build": prob.stats["t_candidates_s"] + prob.stats["t_conflicts_s"]}
    for s in ("greedy", "sector_greedy", "bnb"):
        plan, rep = solve(prob, s)
        row[s] = (plan.status, plan.cost, rep.ok, plan.extra["t_solve_s"], plan.nodes, plan_fuel(sc, plan))
        if s == "bnb" and mc and plan.status == "CONFLICT_FREE":
            row["mc"] = monte_carlo(sc, plan, trials=mc)
    row["value"] = {}
    for s in ("greedy_drops", "bnb"):
        plan, rep = solve(prob, s)
        row.setdefault("verify_extra", []).append((s, rep.ok))
        rob = monte_carlo(sc, plan, trials=mc) if mc else None
        row["value"][s] = (_value(sc, plan), _value(sc, plan, rob) if rob else None)
    return row


def cmd_bench(args) -> None:
    params = _params(args.param)
    with ProcessPoolExecutor(args.jobs) as ex:
        seeds = range(args.start, args.start + args.seeds)
        k = len(seeds)
        rows = list(
            ex.map(
                _bench_one,
                seeds,
                [args.mc] * k,
                [args.targets] * k,
                [args.per_target] * k,
                [args.spread] * k,
                [params] * k,
            )
        )
    n = len(rows)
    g_ok = [r for r in rows if r["greedy"][0] == "CONFLICT_FREE"]
    b_full = [r for r in rows if r["bnb"][0] == "CONFLICT_FREE"]
    b_part = [r for r in rows if r["bnb"][0] == "PARTIAL"]
    rescued = [r["seed"] for r in rows if r["greedy"][0] != "CONFLICT_FREE" and r["bnb"][0] == "CONFLICT_FREE"]
    both = [r for r in g_ok if r["bnb"][0] == "CONFLICT_FREE"]
    better = [r for r in both if r["bnb"][1] < r["greedy"][1] - 1e-9]
    worse = [r["seed"] for r in both if r["bnb"][1] > r["greedy"][1] + 1e-9]
    bad_verify = [(r["seed"], s) for r in rows for s in ("greedy", "sector_greedy", "bnb") if not r[s][2]]
    bad_verify += [(r["seed"], s) for r in rows for s, ok in r["verify_extra"] if not ok]
    sg_ok = [r for r in rows if r["sector_greedy"][0] == "CONFLICT_FREE"]
    sg_gap = [(r["sector_greedy"][1] - r["bnb"][1]) / r["bnb"][1] for r in sg_ok if r["bnb"][0] == "CONFLICT_FREE"]
    print(
        f"seeds {args.start}..{args.start + n - 1}: {args.targets} targets x {args.per_target} aircraft, "
        f"target spread +-{args.spread:.0f} m" + (f", params {' '.join(args.param)}" if args.param else "")
    )
    print(f"greedy  all aircraft planned : {len(g_ok)}/{n}")
    print(
        f"sector_greedy all planned    : {len(sg_ok)}/{n}   "
        f"cost above bnb: mean {100 * statistics.mean(sg_gap or [0]):.1f}%, "
        f"equal to bnb in {sum(g < 1e-9 for g in sg_gap)}/{len(sg_gap)}"
    )
    print(f"bnb     all aircraft planned : {len(b_full)}/{n}   partial (some dropped): {len(b_part)}")
    print(f"bnb rescued greedy failures  : {len(rescued)}  e.g. seeds {rescued[:10]}")
    if both:
        saving = statistics.mean((r["greedy"][1] - r["bnb"][1]) / r["greedy"][1] for r in both)
        print(f"bnb cheaper than greedy      : {len(better)}/{len(both)}  mean cost reduction {100 * saving:.1f}%")
    print(f"bnb worse than greedy (bug)  : {worse}")
    sg_both = [r for r in sg_ok if r["bnb"][0] == "CONFLICT_FREE"]
    for s, rs in (("greedy", both), ("sector_greedy", sg_both)):
        if rs:
            d = [(r["bnb"][5] - r[s][5]) / r[s][5] for r in rs]
            print(
                f"fuel bnb vs {s:13s}: mean {100 * statistics.mean(d):+.2f}%  min {100 * min(d):+.2f}%  "
                f"max {100 * max(d):+.2f}%  bnb uses more in {sum(x > 1e-9 for x in d)}/{len(d)}"
            )
    print(f"verifier failures (bug)      : {bad_verify}")
    print(f"build time (shared) mean {statistics.mean(r['t_build'] for r in rows):.2f}s")
    for s in ("greedy", "sector_greedy", "bnb"):
        ts = sorted(r[s][3] * 1000 for r in rows)
        print(
            f"  {s:13s} solve ms: mean {statistics.mean(ts):8.2f}  "
            f"p95 {ts[int(0.95 * (n - 1))]:8.2f}  max {ts[-1]:8.2f}"
        )
    limit = params.max_nodes
    print(f"  bnb nodes max {max(r['bnb'][4] for r in rows)}, node limit hit: {sum(r['bnb'][4] > limit for r in rows)}")
    for k, label in ((0, "planned"), (1, f"expected after MC ({args.mc} trials)")):
        if rows[0]["value"]["bnb"][k] is None:
            continue
        g = [r["value"]["greedy_drops"][k] for r in rows]
        b = [r["value"]["bnb"][k] for r in rows]
        d = [y - x for x, y in zip(g, b, strict=True)]
        print(
            f"target value {label}: greedy_drops {100 * statistics.mean(g):.2f}%  bnb {100 * statistics.mean(b):.2f}%  "
            f"diff {100 * statistics.mean(d):+.2f} pts (min {100 * min(d):+.2f}, max {100 * max(d):+.2f}); "
            f"bnb more in {sum(x > 1e-9 for x in d)}, less in {sum(x < -1e-9 for x in d)}"
        )
    mcs = [r["mc"] for r in rows if "mc" in r]
    if mcs:
        print(f"robustness of bnb plans ({len(mcs)} plans x {args.mc} trials, mean over plans):")
        for name in ("p_sep_loss", "p_terrain", "p_arrival_miss"):
            vals = [getattr(m, name) for m in mcs]
            print(f"  {name:15s} mean {statistics.mean(vals):6.1%}   plans with any: {sum(v > 0 for v in vals)}")
        print(
            f"  arrival_err_p95 median {statistics.median(m.arrival_err_p95 for m in mcs):.1f}s, "
            f"min_sep_p05 median {statistics.median(m.min_sep_p05 for m in mcs):.2f}"
        )


def cmd_gen(args) -> None:
    sc = scn.generate(args.seed, n_targets=args.targets, per_target=args.per_target, target_spread=args.spread)
    scn.save(sc, args.out)
    print(f"wrote {args.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="traj-opt")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def scenario_args(p):
        p.add_argument("scenario", nargs="?", help="scenario JSON (omit to generate from --seed)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--targets", type=int, default=2)
        p.add_argument("--per-target", type=int, default=3)
        p.add_argument("--spread", type=float, default=1500.0, help="targets drawn in +-spread m")

    r = sub.add_parser("run", help="plan, verify, and visualize one scenario")
    scenario_args(r)
    r.add_argument("--solver", choices=["greedy", "sector_greedy", "bnb", "both"], default="both")
    r.add_argument("--gif", help="write an animation, e.g. out.gif")
    r.add_argument("--png", help="write a still, e.g. out.png")
    r.add_argument("--no-show", action="store_true", help="do not open a window")
    r.add_argument("--mc", type=int, default=200, help="Monte Carlo robustness trials (0 = off)")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("bench", help="greedy vs branch-and-bound over random seeds")
    b.add_argument("--seeds", type=int, default=200)
    b.add_argument("--start", type=int, default=0)
    b.add_argument("--jobs", type=int, default=None)
    b.add_argument("--targets", type=int, default=2)
    b.add_argument("--per-target", type=int, default=3)
    b.add_argument("--spread", type=float, default=1500.0, help="targets drawn in +-spread m")
    b.add_argument("--mc", type=int, default=0, help="Monte Carlo trials per bnb plan (0 = off)")
    b.add_argument("--param", action="append", metavar="KEY=VALUE", help="override a Params field (repeatable)")
    b.set_defaults(func=cmd_bench)

    g = sub.add_parser("gen", help="write a generated scenario to JSON")
    g.add_argument("out")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--targets", type=int, default=2)
    g.add_argument("--per-target", type=int, default=3)
    g.add_argument("--spread", type=float, default=1500.0)
    g.set_defaults(func=cmd_gen)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
