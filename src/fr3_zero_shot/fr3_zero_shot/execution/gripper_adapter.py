from __future__ import annotations


GRIPPER_OPEN_M = 0.04
GRIPPER_CLOSE_M = 0.014


def contact_detected_from_width(
    sensed_width_m: float | None,
    *,
    commanded_width_m: float = GRIPPER_CLOSE_M,
    margin_m: float = 0.002,
) -> bool:
    if sensed_width_m is None:
        return False
    return bool(float(sensed_width_m) > float(commanded_width_m) + float(margin_m))
