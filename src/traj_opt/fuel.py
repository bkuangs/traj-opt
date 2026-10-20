"""Fuel (energy) estimate for a plan, in seconds of level flight at nominal speed.

Model, per aircraft, up to the ingress point:
- Level-flight power P(v) = D(v) v with parabolic drag D = a v^2 + b / v^2. Assuming v_nom is the
  best-range (minimum-drag) speed gives P(v) / P(v_nom) = (v^4 + v_nom^4) / (2 v v_nom^3).
- Climb energy m g dh. With lift-to-drag ratio L/D at v_nom, P(v_nom) = m g v_nom / (L/D), so each
  metre climbed costs (L/D) / v_nom seconds of nominal cruise. Descents get no credit.

Because all aircraft arrive at the same T, every candidate flies for (nearly) the same time; fuel
differences come from speed (longer route -> faster) and climbs, not from distance directly.
"""

from __future__ import annotations

import numpy as np

from .model import Candidate, Plan, Scenario

LIFT_TO_DRAG = 10.0


def power_ratio(v: float, v_nom: float) -> float:
    return (v**4 + v_nom**4) / (2.0 * v * v_nom**3)


def candidate_fuel(c: Candidate, v_nom: float, lift_to_drag: float = LIFT_TO_DRAG) -> float:
    m = c.traj.t <= c.traj.t_ingress + 1e-9
    climb = float(np.clip(np.diff(c.traj.xyz[m, 2]), 0.0, None).sum())
    return c.traj.t_ingress * power_ratio(c.speed, v_nom) + climb * lift_to_drag / v_nom


def plan_fuel(sc: Scenario, plan: Plan) -> float:
    acs = sc.aircraft_by_id()
    return sum(candidate_fuel(c, acs[i].v_nom) for i, c in (plan.chosen or {}).items() if c is not None)
