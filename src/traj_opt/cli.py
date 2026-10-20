"""Command line: run a scenario, benchmark solvers over random seeds, or save a generated scenario."""

from __future__ import annotations

import argparse
import statistics
from concurrent.futures import ProcessPoolExecutor

from . import scenario as scn
from .planner import build_problem, solve
from .robust import monte_carlo


def _load(args) -> scn.Scenario:
    if args.scenario:
        return scn.load(args.scenario)
    return scn.generate(args.seed, n_targets=args.targets, per_target=args.per_target)


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


def _bench_one(seed: int, mc: int = 0) -> dict:
    sc = scn.generate(seed)
    prob = build_problem(sc)
    row = {"seed": seed, "t_build": prob.stats["t_candidates_s"] + prob.stats["t_conflicts_s"]}
    for s in ("greedy", "bnb"):
        plan, rep = solve(prob, s)
        row[s] = (plan.status, plan.cost, rep.ok, plan.extra["t_solve_s"], plan.nodes)
        if s == "bnb" and mc and plan.status == "CONFLICT_FREE":
            row["mc"] = monte_carlo(sc, plan, trials=mc)
    return row


def cmd_bench(args) -> None:
    with ProcessPoolExecutor(args.jobs) as ex:
        seeds = range(args.start, args.start + args.seeds)
        rows = list(ex.map(_bench_one, seeds, [args.mc] * len(seeds)))
    n = len(rows)
    g_ok = [r for r in rows if r["greedy"][0] == "CONFLICT_FREE"]
    b_full = [r for r in rows if r["bnb"][0] == "CONFLICT_FREE"]
    b_part = [r for r in rows if r["bnb"][0] == "PARTIAL"]
    rescued = [r["seed"] for r in rows if r["greedy"][0] != "CONFLICT_FREE" and r["bnb"][0] == "CONFLICT_FREE"]
    both = [r for r in g_ok if r["bnb"][0] == "CONFLICT_FREE"]
    better = [r for r in both if r["bnb"][1] < r["greedy"][1] - 1e-9]
    worse = [r["seed"] for r in both if r["bnb"][1] > r["greedy"][1] + 1e-9]
    bad_verify = [(r["seed"], s) for r in rows for s in ("greedy", "bnb") if not r[s][2]]
    print(f"seeds {args.start}..{args.start + n - 1}")
    print(f"greedy  all aircraft planned : {len(g_ok)}/{n}")
    print(f"bnb     all aircraft planned : {len(b_full)}/{n}   partial (some dropped): {len(b_part)}")
    print(f"bnb rescued greedy failures  : {len(rescued)}  e.g. seeds {rescued[:10]}")
    if both:
        saving = statistics.mean((r["greedy"][1] - r["bnb"][1]) / r["greedy"][1] for r in both)
        print(f"bnb cheaper than greedy      : {len(better)}/{len(both)}  mean cost reduction {100 * saving:.1f}%")
    print(f"bnb worse than greedy (bug)  : {worse}")
    print(f"verifier failures (bug)      : {bad_verify}")
    print(
        f"build time mean {statistics.mean(r['t_build'] for r in rows):.2f}s, "
        f"bnb solve max {max(r['bnb'][3] for r in rows) * 1000:.1f}ms, nodes max {max(r['bnb'][4] for r in rows)}"
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
    sc = scn.generate(args.seed, n_targets=args.targets, per_target=args.per_target)
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

    r = sub.add_parser("run", help="plan, verify, and visualize one scenario")
    scenario_args(r)
    r.add_argument("--solver", choices=["greedy", "bnb", "both"], default="both")
    r.add_argument("--gif", help="write an animation, e.g. out.gif")
    r.add_argument("--png", help="write a still, e.g. out.png")
    r.add_argument("--no-show", action="store_true", help="do not open a window")
    r.add_argument("--mc", type=int, default=200, help="Monte Carlo robustness trials (0 = off)")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("bench", help="greedy vs branch-and-bound over random seeds")
    b.add_argument("--seeds", type=int, default=200)
    b.add_argument("--start", type=int, default=0)
    b.add_argument("--jobs", type=int, default=None)
    b.add_argument("--mc", type=int, default=0, help="Monte Carlo trials per bnb plan (0 = off)")
    b.set_defaults(func=cmd_bench)

    g = sub.add_parser("gen", help="write a generated scenario to JSON")
    g.add_argument("out")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--targets", type=int, default=2)
    g.add_argument("--per-target", type=int, default=3)
    g.set_defaults(func=cmd_gen)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
