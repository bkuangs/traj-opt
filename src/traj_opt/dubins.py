"""Shortest Dubins paths (forward-only, minimum turn radius) between 2D poses.

Pose = (x, y, heading). Formulas follow Shkel & Lumelsky (2001).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi
WORDS = ("LSL", "RSR", "LSR", "RSL", "RLR", "LRL")


def mod2pi(a: float) -> float:
    return a % TWO_PI


@dataclass(frozen=True)
class DubinsPath:
    q0: tuple[float, float, float]
    r: float
    word: str
    seg: tuple[float, float, float]  # normalized segment lengths (radians for arcs, units of r for S)

    @property
    def length(self) -> float:
        return self.r * sum(self.seg)

    def end(self) -> tuple[float, float, float]:
        p = self.sample(np.array([self.length]))
        return tuple(p[0])

    def sample(self, s: np.ndarray) -> np.ndarray:
        """Poses (N, 3) at arc lengths s (clipped to [0, length])."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length) / self.r
        out = np.empty((s.size, 3))
        x, y, th = self.q0
        start = 0.0
        for k, (kind, ln) in enumerate(zip(self.word, self.seg, strict=True)):
            m = (s >= start) if k == 2 else (s >= start) & (s < start + ln)
            if m.any():
                out[m] = np.stack(_step(kind, x, y, th, s[m] - start, self.r), axis=1)
            x, y, th = _step(kind, x, y, th, ln, self.r)
            start += ln
        return out


def _step(kind: str, x, y, th, ln, r: float):
    """Pose after moving ln (normalized) along segment kind; ln may be an array."""
    if kind == "L":
        return x + r * (np.sin(th + ln) - np.sin(th)), y + r * (np.cos(th) - np.cos(th + ln)), th + ln
    if kind == "R":
        return x + r * (np.sin(th) - np.sin(th - ln)), y + r * (np.cos(th - ln) - np.cos(th)), th - ln
    return x + ln * r * np.cos(th), y + ln * r * np.sin(th), th + 0.0 * ln


def _word(word: str, a: float, b: float, d: float):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    cab = math.cos(a - b)
    if word == "LSL":
        p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
        if p2 < 0:
            return None
        tmp = math.atan2(cb - ca, d + sa - sb)
        return mod2pi(-a + tmp), math.sqrt(p2), mod2pi(b - tmp)
    if word == "RSR":
        p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
        if p2 < 0:
            return None
        tmp = math.atan2(ca - cb, d - sa + sb)
        return mod2pi(a - tmp), math.sqrt(p2), mod2pi(-b + tmp)
    if word == "LSR":
        p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
        if p2 < 0:
            return None
        p = math.sqrt(p2)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        return mod2pi(-a + tmp), p, mod2pi(-b + tmp)
    if word == "RSL":
        p2 = -2 + d * d + 2 * cab - 2 * d * (sa + sb)
        if p2 < 0:
            return None
        p = math.sqrt(p2)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        return mod2pi(a - tmp), p, mod2pi(b - tmp)
    if word == "RLR":
        tmp = (6 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8
        if abs(tmp) > 1:
            return None
        p = mod2pi(TWO_PI - math.acos(tmp))
        t = mod2pi(a - math.atan2(ca - cb, d - sa + sb) + p / 2)
        return t, p, mod2pi(a - b - t + p)
    if word == "LRL":
        tmp = (6 - d * d + 2 * cab + 2 * d * (sb - sa)) / 8
        if abs(tmp) > 1:
            return None
        p = mod2pi(TWO_PI - math.acos(tmp))
        t = mod2pi(-a - math.atan2(ca - cb, d + sa - sb) + p / 2)
        return t, p, mod2pi(b - a - t + p)
    raise ValueError(word)


def shortest(q0, q1, r: float) -> DubinsPath:
    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    d = math.hypot(dx, dy) / r
    theta = mod2pi(math.atan2(dy, dx))
    a = mod2pi(q0[2] - theta)
    b = mod2pi(q1[2] - theta)
    best = None
    for w in WORDS:
        seg = _word(w, a, b, d)
        if seg is not None and (best is None or sum(seg) < sum(best[1])):
            best = (w, seg)
    assert best is not None
    return DubinsPath(tuple(map(float, q0)), r, best[0], tuple(best[1]))


class Chain:
    """A sequence of Dubins paths sampled as one curve."""

    def __init__(self, parts: list[DubinsPath]):
        self.parts = parts
        self.lengths = np.array([p.length for p in parts])
        self.cum = np.concatenate([[0.0], np.cumsum(self.lengths)])

    @property
    def length(self) -> float:
        return float(self.cum[-1])

    def sample(self, s: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        out = np.empty((s.size, 3))
        k = np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.parts) - 1)
        for i, part in enumerate(self.parts):
            m = k == i
            if m.any():
                out[m] = part.sample(s[m] - self.cum[i])
        return out
