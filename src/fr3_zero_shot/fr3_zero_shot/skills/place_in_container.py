from __future__ import annotations

from fr3_zero_shot.affordance.container_access import ContainerAccess, evaluate_container_access
from fr3_zero_shot.core.geometry import bounded_xy_offset
from fr3_zero_shot.core.types import FailureCode, ObjectHypothesis, ReleaseGateResult, SkillOutcome


def strict_release_gate(
    *,
    actual_tcp_xyz: tuple[float, float, float],
    commanded_tcp_xyz: tuple[float, float, float],
    target_center_xy: tuple[float, float],
    tcp_to_object_center_xy: tuple[float, float] = (0.0, 0.0),
    center_tolerance_m: float,
    safe_release_z_m: float,
    rim_z_m: float,
    z_slack_m: float = 0.005,
    tcp_xy_tolerance_m: float | None = None,
) -> ReleaseGateResult:
    target_xy = (float(target_center_xy[0]), float(target_center_xy[1]))
    actual_xy = (float(actual_tcp_xyz[0]), float(actual_tcp_xyz[1]))
    commanded_xy = (float(commanded_tcp_xyz[0]), float(commanded_tcp_xyz[1]))
    object_center_xy = (
        float(actual_xy[0] + float(tcp_to_object_center_xy[0])),
        float(actual_xy[1] + float(tcp_to_object_center_xy[1])),
    )
    actual_center_err = ((object_center_xy[0] - target_xy[0]) ** 2 + (object_center_xy[1] - target_xy[1]) ** 2) ** 0.5
    actual_target_xy_err = ((actual_xy[0] - commanded_xy[0]) ** 2 + (actual_xy[1] - commanded_xy[1]) ** 2) ** 0.5
    tcp_xy_tolerance = max(0.0, float(tcp_xy_tolerance_m if tcp_xy_tolerance_m is not None else center_tolerance_m))
    z_limit = float(safe_release_z_m) + max(0.0, float(z_slack_m))
    z_above_limit = max(0.0, float(actual_tcp_xyz[2]) - z_limit)
    rim_drop = float(actual_tcp_xyz[2] - float(rim_z_m))
    return ReleaseGateResult(
        ready=bool(
            actual_center_err <= float(center_tolerance_m)
            and actual_target_xy_err <= tcp_xy_tolerance
            and z_above_limit <= 0.0
        ),
        actual_center_err_m=float(actual_center_err),
        actual_target_xy_err_m=float(actual_target_xy_err),
        z_above_limit_m=float(z_above_limit),
        rim_drop_m=rim_drop,
        actual_object_center_xy=object_center_xy,
        target_center_xy=target_xy,
    )


class PlaceInContainerSkill:
    def __init__(
        self,
        *,
        motion,
        center_tolerance_m: float = 0.015,
        tcp_xy_tolerance_m: float = 0.015,
        max_release_offset_m: float = 0.008,
        release_drop_m: float = 0.035,
        release_z_slack_m: float = 0.005,
    ) -> None:
        self._motion = motion
        self._center_tolerance_m = max(0.0, float(center_tolerance_m))
        self._tcp_xy_tolerance_m = max(0.0, float(tcp_xy_tolerance_m))
        self._max_release_offset_m = max(0.0, float(max_release_offset_m))
        self._release_drop_m = max(0.0, float(release_drop_m))
        self._release_z_slack_m = max(0.0, float(release_z_slack_m))

    def run(
        self,
        *,
        held_object: ObjectHypothesis,
        container: ObjectHypothesis,
        tcp_to_object_center_xy: tuple[float, float] = (0.0, 0.0),
        access: ContainerAccess | None = None,
    ) -> SkillOutcome:
        container_access = (
            evaluate_container_access(container, release_drop_m=self._release_drop_m)
            if access is None
            else access
        )
        if not container_access.valid:
            return SkillOutcome(False, FailureCode.TARGET_NOT_VISIBLE, container_access.reason)

        release_offset = bounded_xy_offset(tcp_to_object_center_xy, max_norm_m=self._max_release_offset_m)
        center_x, center_y = container_access.opening_center_xy
        release_xyz = (
            float(center_x - release_offset[0]),
            float(center_y - release_offset[1]),
            float(container_access.safe_release_z_m),
        )
        hover_xyz = (release_xyz[0], release_xyz[1], float(container_access.release_hover_z_m))

        for label, xyz in (("hover", hover_xyz), ("release", release_xyz)):
            result = self._motion.move_to_pose(xyz=xyz, quat_xyzw=self._motion.default_place_quat_xyzw())
            if not result.success:
                return SkillOutcome(False, FailureCode.MOTION_FAILED, f"{label} motion failed: {result.message}")

        actual_pose = self._motion.current_tcp_pose()
        if actual_pose is None:
            return SkillOutcome(False, FailureCode.RELEASE_GATE_FAILED, "no TCP pose for release gate")
        gate = strict_release_gate(
            actual_tcp_xyz=actual_pose[0],
            commanded_tcp_xyz=release_xyz,
            target_center_xy=container_access.opening_center_xy,
            tcp_to_object_center_xy=tcp_to_object_center_xy,
            center_tolerance_m=self._center_tolerance_m,
            safe_release_z_m=container_access.safe_release_z_m,
            rim_z_m=container_access.rim_z_m,
            z_slack_m=self._release_z_slack_m,
            tcp_xy_tolerance_m=self._tcp_xy_tolerance_m,
        )
        if (
            not gate.ready
            and gate.actual_center_err_m <= self._center_tolerance_m
            and gate.actual_target_xy_err_m <= self._tcp_xy_tolerance_m
            and gate.z_above_limit_m > 0.0
        ):
            corrected_release_z = max(
                float(container_access.rim_z_m + 0.050),
                float(release_xyz[2] - min(0.050, gate.z_above_limit_m + 0.015)),
            )
            corrected_release_xyz = (release_xyz[0], release_xyz[1], corrected_release_z)
            result = self._motion.move_to_pose(
                xyz=corrected_release_xyz,
                quat_xyzw=self._motion.default_place_quat_xyzw(),
            )
            if not result.success:
                return SkillOutcome(
                    False,
                    FailureCode.MOTION_FAILED,
                    f"corrective release motion failed: {result.message}",
                )
            actual_pose = self._motion.current_tcp_pose()
            if actual_pose is None:
                return SkillOutcome(False, FailureCode.RELEASE_GATE_FAILED, "no TCP pose for release gate")
            gate = strict_release_gate(
                actual_tcp_xyz=actual_pose[0],
                commanded_tcp_xyz=corrected_release_xyz,
                target_center_xy=container_access.opening_center_xy,
                tcp_to_object_center_xy=tcp_to_object_center_xy,
                center_tolerance_m=self._center_tolerance_m,
                safe_release_z_m=container_access.safe_release_z_m,
                rim_z_m=container_access.rim_z_m,
                z_slack_m=self._release_z_slack_m,
                tcp_xy_tolerance_m=self._tcp_xy_tolerance_m,
            )
        if not gate.ready:
            return SkillOutcome(
                False,
                FailureCode.RELEASE_GATE_FAILED,
                "strict release gate failed",
                data={"gate": gate},
            )

        opened = self._motion.open_gripper()
        if not opened.success:
            return SkillOutcome(False, FailureCode.MOTION_FAILED, opened.message)
        retreat = self._motion.move_to_pose(xyz=hover_xyz, quat_xyzw=self._motion.default_place_quat_xyzw())
        if not retreat.success:
            return SkillOutcome(False, FailureCode.MOTION_FAILED, f"retreat failed: {retreat.message}")
        if not self._motion.verify_place_in(held_object, container):
            return SkillOutcome(
                False,
                FailureCode.PLACE_VERIFY_FAILED,
                "place verification failed",
                data={"gate": gate},
            )
        return SkillOutcome(True, FailureCode.NONE, "place-in complete", data={"gate": gate})
