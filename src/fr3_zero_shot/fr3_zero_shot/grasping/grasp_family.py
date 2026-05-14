from __future__ import annotations

from fr3_zero_shot.core.types import ApproachFamily


SIDE_FAMILIES = (
    ApproachFamily.SIDE_LEFT,
    ApproachFamily.SIDE_RIGHT,
    ApproachFamily.SIDE_FRONT,
    ApproachFamily.SIDE_BACK,
)

ALL_FAMILIES = (
    ApproachFamily.TOP_DOWN,
    ApproachFamily.TOP_DOWN_YAW_45,
    *SIDE_FAMILIES,
)


def approach_vector_xy(family: ApproachFamily) -> tuple[float, float]:
    if family == ApproachFamily.SIDE_LEFT:
        return (0.0, -1.0)
    if family == ApproachFamily.SIDE_RIGHT:
        return (0.0, 1.0)
    if family == ApproachFamily.SIDE_FRONT:
        return (-1.0, 0.0)
    if family == ApproachFamily.SIDE_BACK:
        return (1.0, 0.0)
    return (0.0, 0.0)

