from __future__ import annotations

from fr3_zero_shot.core.types import GroundedPlan, ObjectHypothesis


def relative_place_xy(
    grounded: GroundedPlan,
    target: ObjectHypothesis,
    *,
    offset_m: float = 0.10,
) -> tuple[float, float] | None:
    relation = grounded.symbolic.relation
    if relation == "left_of":
        return (float(target.xyz[0]), float(target.xyz[1] + offset_m))
    if relation == "right_of":
        return (float(target.xyz[0]), float(target.xyz[1] - offset_m))
    if relation == "near":
        return (float(target.xyz[0] + offset_m), float(target.xyz[1]))
    if relation == "on":
        return (float(target.xyz[0]), float(target.xyz[1]))
    return None

