"""Load requirements.toml: every number the safety and timing claims rest on, tagged with its source."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass

from .model import Params
from .robust import Dispersions

SOURCES = ("placeholder", "spec", "measured")


@dataclass(frozen=True)
class Budget:
    p_los_max: float = 1e-4
    p_arrival_miss_max: float = 0.01
    plan_success_min: float = 0.95
    plan_time_p99_max: float = 5.0


@dataclass(frozen=True)
class Requirements:
    params: Params
    disp: Dispersions
    budget: Budget
    sources: dict[str, str]  # key -> placeholder | spec | measured

    @property
    def placeholders(self) -> list[str]:
        return sorted(k for k, s in self.sources.items() if s == "placeholder")


def load(path: str, params: Params | None = None) -> Requirements:
    """Values override `params` (default Params()), Dispersions() and Budget() field by field."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    groups = {"params": {}, "disp": {}, "budget": {}}
    fields = {
        name: {f.name for f in dataclasses.fields(cls)}
        for name, cls in (("params", Params), ("disp", Dispersions), ("budget", Budget))
    }
    sources = {}
    for section in raw.values():
        for key, entry in section.items():
            if entry.get("source") not in SOURCES:
                raise ValueError(f"{key}: source must be one of {SOURCES}")
            dest = next((g for g, names in fields.items() if key in names), None)
            if dest is None:
                raise ValueError(f"{key}: not a Params, Dispersions or Budget field")
            groups[dest][key] = entry["value"]
            sources[key] = entry["source"]
    return Requirements(
        dataclasses.replace(params or Params(), **groups["params"]),
        Dispersions(**groups["disp"]),
        Budget(**groups["budget"]),
        sources,
    )
