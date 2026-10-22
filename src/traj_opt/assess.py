"""Judge a planner against requirements: plan each seed once, replay it under one or more error models,
pool the per-trial outcomes across missions, and compare against the budget.

Probabilities are per mission, averaged over the generated scenarios. Unplanned missions do not fly,
so they count against plan success, not against separation or arrival.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np

from . import scenario as scn
from .model import Params
from .planner import build_problem, solve
from .requirements import Budget
from .robust import Dispersions, monte_carlo

CHUNK = 200  # trials per monte_carlo call, bounds memory


def binom_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial probability after k events in n trials."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    alpha = 1.0 - conf

    def cdf(p: float) -> float:
        lp, lq = math.log(p), math.log1p(-p)
        lc = math.lgamma(n + 1)
        return sum(
            math.exp(lc - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq) for i in range(k + 1)
        )

    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


@dataclass
class Outcome:
    """Pooled per-trial outcomes for one error model."""

    weight: np.ndarray
    sep_loss: np.ndarray
    arrival_err: np.ndarray
    warning: np.ndarray

    @staticmethod
    def concat(parts: list[Outcome]) -> Outcome:
        return Outcome(*(np.concatenate([getattr(p, f) for p in parts]) for f in Outcome.__dataclass_fields__))


def evaluate(
    seed: int, gen: dict, params: Params, solver: str, disps: list[Dispersions], trials: int
) -> tuple[bool, float, bool, list[Outcome | None]]:
    """Plan one generated scenario, then replay the plan under each error model.
    Returns (planned, planning time, fell back to bnb, outcome per error model)."""
    sc = scn.generate(seed, params=params, **gen)
    t0 = time.perf_counter()
    prob = build_problem(sc)
    plan, rep = solve(prob, solver)
    t_plan = time.perf_counter() - t0
    ok = plan.status == "CONFLICT_FREE" and rep.ok
    fallback = bool(plan.extra.get("fallback"))
    if not ok:
        return False, t_plan, fallback, [None] * len(disps)
    outs = []
    for d in disps:
        parts = []
        for k in range(0, trials, CHUNK):
            r = monte_carlo(sc, plan, min(CHUNK, trials - k), d, seed=seed * 100_003 + k)
            parts.append(Outcome(r.weight, r.sep_loss, r.arrival_err, r.warning))
        outs.append(Outcome.concat(parts))
    return True, t_plan, fallback, outs


@dataclass
class Metrics:
    trials: int
    los_events: int
    p_los: float
    p_los_upper: float
    ess: float  # effective sample size; equals trials without importance sampling
    p_arrival_miss: float
    arrival_p95: float
    arrival_p99: float
    p_repair: float  # P(some aircraft can tell before arrival that it will miss): an in-flight repair is needed
    warning_median: float  # s before arrival at which that becomes known, median over such trials


def metrics(out: Outcome, disp: Dispersions, budget: Budget) -> Metrics:
    w = out.weight
    n = w.size
    weighted = disp.is_scale != 1.0
    k = int(out.sep_loss.sum())
    if weighted:
        x = w * out.sep_loss
        p = float(x.mean())
        upper = p + 1.645 * float(x.std(ddof=1)) / math.sqrt(n) if k else float("nan")
        ess = float(w.sum() ** 2 / (w * w).sum())
        p_miss = float((w * (out.arrival_err > disp.arrival_tol)).mean())
        p_rep = float((w * (out.warning > 0)).mean())
    else:
        p, upper, ess = k / n, binom_upper(k, n), float(n)
        p_miss = float((out.arrival_err > disp.arrival_tol).mean())
        p_rep = float((out.warning > 0).mean())
    return Metrics(
        n,
        k,
        p,
        upper,
        ess,
        p_miss,
        float(np.percentile(out.arrival_err, 95)),
        float(np.percentile(out.arrival_err, 99)),
        p_rep,
        float(np.median(out.warning[out.warning > 0])) if (out.warning > 0).any() else float("nan"),
    )


def run(
    seeds: range,
    gen: dict,
    params: Params,
    solver: str,
    disps: list[Dispersions],
    trials: int,
    jobs: int | None = None,
) -> tuple[float, list[float], int, list[Outcome | None]]:
    """Returns (plan success fraction, planning times, bnb fallbacks, pooled outcome per error model)."""
    from concurrent.futures import ProcessPoolExecutor

    k = len(seeds)
    with ProcessPoolExecutor(jobs) as ex:
        rows = list(ex.map(evaluate, seeds, [gen] * k, [params] * k, [solver] * k, [disps] * k, [trials] * k))
    planned = [r for r in rows if r[0]]
    pooled = [Outcome.concat([r[3][i] for r in planned]) if planned else None for i in range(len(disps))]
    return len(planned) / k, [r[1] for r in rows], sum(r[2] for r in rows), pooled


SWEEP_KEYS = (
    "wind_sd",
    "gust_sd",
    "start_delay_sd",
    "cross_track_sd",
    "xtrack_drift_sd",
    "turn_xtrack_sd",
    "alt_sd",
)


def sweep_disps(base: Dispersions, scales: list[float]) -> list[tuple[str, float, Dispersions]]:
    """One-at-a-time scaling of each error term; the unscaled base comes first."""
    out = [("base", 1.0, base)]
    for key in SWEEP_KEYS:
        for s in scales:
            if s != 1.0:
                out.append((key, s, replace(base, **{key: getattr(base, key) * s})))
    return out
