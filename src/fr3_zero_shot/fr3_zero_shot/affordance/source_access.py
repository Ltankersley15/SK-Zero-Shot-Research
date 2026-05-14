from __future__ import annotations

from dataclasses import dataclass

from fr3_zero_shot.core.geometry import footprint_radius, xy_distance
from fr3_zero_shot.core.types import ObjectHypothesis


@dataclass(frozen=True)
class SourceAccess:
    top_down_clear: bool
    side_clear: bool
    nearest_keepout_clearance_m: float
    reason: str


def evaluate_source_access(
    source: ObjectHypothesis,
    keepouts: tuple[ObjectHypothesis, ...],
    *,
    min_top_down_clearance_m: float = 0.05,
    min_side_clearance_m: float = 0.015,
) -> SourceAccess:
    if not keepouts:
        return SourceAccess(True, True, float("inf"), "no keepouts")
    source_xy = (source.xyz[0], source.xyz[1])
    clearances = []
    for keepout in keepouts:
        clearance = xy_distance(source_xy, (keepout.xyz[0], keepout.xyz[1]))
        clearance -= footprint_radius(source.footprint_xy) + footprint_radius(keepout.footprint_xy)
        clearances.append(clearance)
    nearest = min(clearances)
    top_down_clear = nearest >= float(min_top_down_clearance_m)
    side_clear = nearest >= float(min_side_clearance_m)
    reason = "top-down clear" if top_down_clear else "near keepout; prefer side access"
    return SourceAccess(top_down_clear, side_clear, float(nearest), reason)

