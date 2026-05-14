from __future__ import annotations

from dataclasses import dataclass

from fr3_zero_shot.core.types import ObjectHypothesis


@dataclass(frozen=True)
class ContainerAccess:
    opening_center_xy: tuple[float, float]
    rim_z_m: float
    safe_release_z_m: float
    release_hover_z_m: float
    valid: bool
    reason: str


def evaluate_container_access(
    target: ObjectHypothesis,
    *,
    rim_margin_m: float = 0.004,
    release_drop_m: float = 0.035,
    hover_clearance_m: float = 0.12,
) -> ContainerAccess:
    if not target.container_like:
        return ContainerAccess((target.xyz[0], target.xyz[1]), target.xyz[2], target.xyz[2], target.xyz[2], False, "target is not container-like")
    rim_z = float(target.xyz[2] + max(0.0, target.height_m * 0.5))
    safe_release_z = float(rim_z + max(0.0, rim_margin_m) + max(0.0, release_drop_m))
    hover_z = float(rim_z + max(0.02, hover_clearance_m))
    return ContainerAccess(
        opening_center_xy=(float(target.xyz[0]), float(target.xyz[1])),
        rim_z_m=rim_z,
        safe_release_z_m=safe_release_z,
        release_hover_z_m=hover_z,
        valid=True,
        reason="container opening estimated from object geometry",
    )

