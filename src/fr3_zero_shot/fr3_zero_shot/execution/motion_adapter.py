from __future__ import annotations

from dataclasses import dataclass

from fr3_zero_shot.execution.gripper_adapter import contact_detected_from_width
from fr3_zero_shot.core.transforms import top_down_quaternion
from fr3_zero_shot.core.types import ObjectHypothesis


@dataclass(frozen=True)
class AdapterResult:
    success: bool
    message: str
    planning_time: float = 0.0


class DryRunMotionAdapter:
    def __init__(self) -> None:
        self.commands: list[dict] = []
        self._tcp_pose: tuple[tuple[float, float, float], tuple[float, float, float, float]] | None = None
        self._holding = False
        self._gripper_width = 0.04

    def move_to_pose(
        self,
        *,
        xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float],
    ) -> AdapterResult:
        self.commands.append({"type": "move_to_pose", "xyz": xyz, "quat_xyzw": quat_xyzw})
        self._tcp_pose = (tuple(float(v) for v in xyz), tuple(float(v) for v in quat_xyzw))
        return AdapterResult(True, "dry-run pose accepted")

    def open_gripper(self) -> AdapterResult:
        self.commands.append({"type": "open_gripper"})
        self._holding = False
        self._gripper_width = 0.04
        return AdapterResult(True, "dry-run gripper opened")

    def close_gripper(
        self,
        *,
        position_m: float = 0.014,
        contact_margin_m: float = 0.002,
    ) -> AdapterResult:
        self.commands.append(
            {"type": "close_gripper", "position_m": position_m, "contact_margin_m": contact_margin_m}
        )
        self._holding = True
        self._gripper_width = 0.02
        return AdapterResult(True, "dry-run gripper closed")

    def verify_carry(self, source: ObjectHypothesis) -> bool:
        self.commands.append({"type": "verify_carry", "object_id": source.object_id})
        return self._holding

    def verify_place_in(self, held_object: ObjectHypothesis, container: ObjectHypothesis) -> bool:
        self.commands.append(
            {
                "type": "verify_place_in",
                "held_object_id": held_object.object_id,
                "container_id": container.object_id,
            }
        )
        return not self._holding

    def current_tcp_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
        return self._tcp_pose

    def default_place_quat_xyzw(self) -> tuple[float, float, float, float]:
        return top_down_quaternion(0.0)

    def current_gripper_width(self) -> float | None:
        return float(self._gripper_width)

    def current_gripper_effort(self) -> float | None:
        return None

    def gripper_contact_detected(self, *, commanded_width: float = 0.014, margin_m: float = 0.002) -> bool:
        return contact_detected_from_width(
            self._gripper_width,
            commanded_width_m=commanded_width,
            margin_m=margin_m,
        )

    def gripper_is_open(self, *, open_threshold_m: float = 0.030) -> bool:
        return bool(float(self._gripper_width) >= float(open_threshold_m))
