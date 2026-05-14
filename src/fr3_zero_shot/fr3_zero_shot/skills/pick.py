from __future__ import annotations

from fr3_zero_shot.core.types import ApproachFamily, FailureCode, GraspCandidate, ObjectHypothesis, SkillOutcome
from fr3_zero_shot.grasping.grasp_planner import GraspPlanner


class PickSkill:
    def __init__(self, *, grasp_planner: GraspPlanner, motion) -> None:
        self._grasp_planner = grasp_planner
        self._motion = motion

    def run(
        self,
        *,
        source: ObjectHypothesis,
        avoid_objects: tuple[ObjectHypothesis, ...],
        approach_preference: ApproachFamily,
        candidate: GraspCandidate | None = None,
        close_position_m: float | None = None,
        contact_margin_m: float = 0.002,
    ) -> SkillOutcome:
        if not source.visible:
            return SkillOutcome(False, FailureCode.SOURCE_NOT_VISIBLE, "source is not visible")
        if candidate is None:
            candidate = self._grasp_planner.plan(
                source=source,
                avoid_objects=avoid_objects,
                approach_preference=approach_preference,
            )
        if candidate is None:
            return SkillOutcome(False, FailureCode.NO_VALID_GRASP, "no valid grasp candidate")

        opened = self._motion.open_gripper()
        if not opened.success:
            return SkillOutcome(False, FailureCode.MOTION_FAILED, opened.message)
        for label, xyz in (
            ("pregrasp", candidate.pregrasp_xyz),
            ("grasp", candidate.grasp_xyz),
        ):
            result = self._motion.move_to_pose(xyz=xyz, quat_xyzw=candidate.quat_xyzw)
            if not result.success:
                return SkillOutcome(False, FailureCode.MOTION_FAILED, f"{label} motion failed: {result.message}")
        close_kwargs = {"contact_margin_m": float(contact_margin_m)}
        if close_position_m is not None:
            close_kwargs["position_m"] = float(close_position_m)
        close = self._motion.close_gripper(**close_kwargs)
        if not close.success:
            retreat = self._motion.move_to_pose(xyz=candidate.retreat_xyz, quat_xyzw=candidate.quat_xyzw)
            return SkillOutcome(
                False,
                FailureCode.MOTION_FAILED,
                close.message,
                data={"grasp": candidate, "retreat_after_close_failure": retreat},
            )
        lift = self._motion.move_to_pose(xyz=candidate.retreat_xyz, quat_xyzw=candidate.quat_xyzw)
        if not lift.success:
            return SkillOutcome(False, FailureCode.MOTION_FAILED, f"retreat failed: {lift.message}")
        carry_ok = self._motion.verify_carry(source)
        if not carry_ok and hasattr(self._motion, "gripper_contact_detected"):
            commanded_width = float(close_position_m) if close_position_m is not None else 0.014
            carry_ok = bool(
                self._motion.gripper_contact_detected(
                    commanded_width=commanded_width,
                    margin_m=float(contact_margin_m),
                )
            )
        if not carry_ok:
            return SkillOutcome(False, FailureCode.CARRY_VERIFY_FAILED, "carry verification failed")
        return SkillOutcome(True, FailureCode.NONE, "pick complete", data={"grasp": candidate})
