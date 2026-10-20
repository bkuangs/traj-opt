"""Matplotlib view: map with a time slider, altitude vs time, separation vs time. Optional GIF export."""

from __future__ import annotations

import math
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider

from .model import Candidate, Plan
from .planner import Problem
from .verify import Report, separation_ratio


def _pos(c: Candidate, t: float) -> np.ndarray | None:
    if t > c.traj.t[-1]:
        return None
    return np.array([np.interp(t, c.traj.t, c.traj.xyz[:, i]) for i in range(3)])


def show(prob: Problem, plan: Plan, report: Report, gif: str | None = None, interactive: bool = True):
    sc, p = prob.sc, prob.sc.params
    chosen = {k: c for k, c in (plan.chosen or {}).items() if c is not None}
    colors = {ac.id: plt.cm.tab10(i % 10) for i, ac in enumerate(sc.aircraft)}
    t_max = max((c.traj.t[-1] for c in chosen.values()), default=prob.T)

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.35, 1], left=0.05, right=0.98, top=0.9, bottom=0.12, hspace=0.3, wspace=0.18
    )
    ax_map, ax_alt, ax_sep = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    # Map extent from every candidate so faint alternatives stay in view
    allxy = np.concatenate(
        [c.traj.xyz[:, :2] for cs in prob.cands.values() for c in cs] + [np.array([[a.x, a.y] for a in sc.aircraft])]
    )
    lo, hi = allxy.min(axis=0) - 500, allxy.max(axis=0) + 500
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
    hz = sc.terrain.height(gx, gy)
    if sc.terrain.hills:
        cf = ax_map.contourf(gx, gy, hz, levels=12, cmap="terrain", alpha=0.45)
        fig.colorbar(cf, ax=ax_map, shrink=0.6, label="terrain (m)")

    for cs in prob.cands.values():
        for c in cs:
            ax_map.plot(c.traj.xyz[:, 0], c.traj.xyz[:, 1], color="0.4", lw=0.4, alpha=0.15)
    for tgt in sc.targets:
        ax_map.plot(tgt.x, tgt.y, "k*", ms=14)
        ax_map.annotate(
            f"{tgt.id} (value {tgt.value:g})", (tgt.x, tgt.y), xytext=(8, 8), textcoords="offset points", fontsize=9
        )
        ring = np.linspace(0, 2 * math.pi, 100)
        ax_map.plot(tgt.x + p.ingress_radius * np.cos(ring), tgt.y + p.ingress_radius * np.sin(ring), "k:", lw=0.6)
        ax_map.plot(tgt.x + p.r_exempt * np.cos(ring), tgt.y + p.r_exempt * np.sin(ring), "r:", lw=0.6)
    for ac in sc.aircraft:
        col = colors[ac.id]
        ax_map.plot(ac.x, ac.y, "o", color=col, ms=7)
        ax_map.annotate(f"ac{ac.id}", (ac.x, ac.y), xytext=(6, -12), textcoords="offset points", color=col, fontsize=9)
        ax_map.arrow(
            ac.x,
            ac.y,
            250 * math.cos(ac.heading),
            250 * math.sin(ac.heading),
            color=col,
            width=25,
            length_includes_head=True,
        )
        c = chosen.get(ac.id)
        if c is None:
            continue
        ax_map.plot(c.traj.xyz[:, 0], c.traj.xyz[:, 1], color=col, lw=2, label=c.label)
        ax_map.plot(*c.ingress_xy, "s", color=col, ms=8, mec="k")
        t, z = c.traj.t, c.traj.xyz[:, 2]
        ax_alt.plot(t, z, color=col, lw=1.8)
        ax_alt.plot(
            t, sc.terrain.height(c.traj.xyz[:, 0], c.traj.xyz[:, 1]) + p.terrain_clearance, color=col, lw=0.8, ls="--"
        )
    ax_map.set_aspect("equal")
    ax_map.set_xlim(lo[0], hi[0])
    ax_map.set_ylim(lo[1], hi[1])
    ax_map.set_xlabel("east (m)")
    ax_map.set_ylabel("north (m)")
    if chosen:
        ax_map.legend(loc="lower left", fontsize=7)

    ax_alt.axvline(prob.T, color="k", lw=0.8, ls=":")
    ax_alt.set_title("altitude (solid) vs terrain + clearance (dashed)", fontsize=10)
    ax_alt.set_xlabel("time (s)")
    ax_alt.set_ylabel("m")

    shown = 0
    for (i, a), (j, b) in combinations(sorted(chosen.items()), 2):
        t, r = separation_ratio(a, b, sc, dt=0.5)
        if r.size and r.min() < 6:
            ax_sep.plot(t, r, lw=1, label=f"ac{i}-ac{j} (min {r.min():.2f})")
            shown += 1
    ax_sep.axhline(1.0, color="r", lw=1)
    ax_sep.axvline(prob.T, color="k", lw=0.8, ls=":")
    ax_sep.set_ylim(0, 6)
    ax_sep.set_title("separation  max(h/sep_h, v/sep_v)  (< 1 = loss)  pairs below 6 shown", fontsize=10)
    ax_sep.set_xlabel("time (s)")
    if shown:
        ax_sep.legend(fontsize=7, loc="upper right")

    verdict = "VERIFIED" if report.ok else f"VERIFY FAILED: {report.violations[0]}"
    fig.suptitle(
        f"{sc.name} | {plan.solver}: {plan.status} {plan.reason} | cost {plan.cost:.2f} | T={prob.T:.1f}s | {verdict}\n"
        f"{prob.stats['candidates']} candidates ({prob.stats['terrain_rejected']} terrain-rejected), "
        f"{prob.stats['conflict_pairs']} conflicting pairs",
        fontsize=10,
    )

    # Time-dependent artists
    dots = {k: ax_map.plot([], [], "o", color=colors[k], ms=9, mec="k")[0] for k in chosen}
    rings = {k: plt.Circle((0, 0), p.sep_h / 2, fill=False, color=colors[k], lw=1) for k in chosen}
    for r in rings.values():
        ax_map.add_patch(r)
    cursors = [ax.axvline(0, color="0.3", lw=1) for ax in (ax_alt, ax_sep)]
    clock = ax_map.text(0.02, 0.97, "", transform=ax_map.transAxes, va="top", fontsize=10, bbox=dict(fc="w", alpha=0.8))

    def update(t: float):
        for k, c in chosen.items():
            q = _pos(c, t)
            visible = q is not None
            dots[k].set_visible(visible)
            rings[k].set_visible(visible)
            if visible:
                dots[k].set_data([q[0]], [q[1]])
                rings[k].center = (q[0], q[1])
        for cur in cursors:
            cur.set_xdata([t, t])
        clock.set_text(f"t = {t:6.1f} s   (ingress at {prob.T:.1f} s)")
        return [*dots.values(), *rings.values(), *cursors, clock]

    update(0.0)
    if gif:
        frames = np.linspace(0, t_max, 120)
        FuncAnimation(fig, update, frames=frames, blit=False).save(gif, writer=PillowWriter(fps=15))
    if interactive:
        slider = Slider(fig.add_axes([0.08, 0.03, 0.5, 0.025]), "time (s)", 0.0, t_max, valinit=0.0)
        slider.on_changed(lambda v: (update(v), fig.canvas.draw_idle()))
        fig._slider = slider  # keep a reference alive
        plt.show()
    return fig
