from __future__ import annotations

from fr3_zero_shot.core.types import FailureCode, ObjectHypothesis, SkillOutcome
from fr3_zero_shot.grounding.relation_grounder import relative_place_xy


class PlaceRelativeSkill:
    def __init__(self, *, motion, release_z_m: float = 0.045, hover_z_m: float = 0.14) -> None:
        self._motion = motion
        self._release_z_m = float(release_z_m)
        self._hover_z_m = float(hover_z_m)

    def run(self, *, held_object: ObjectHypothesis, target: ObjectHypothesis, grounded_plan) -> SkillOutcome:
        xy = relative_place_xy(grounded_plan, target)
        if xy is None:
            return SkillOutcome(False, FailureCode.UNSUPPORTED_COMMAND, "unsupported relative placement relation")
        release_xyz = (xy[0], xy[1], self._release_z_m)
        hover_xyz = (xy[0], xy[1], self._hover_z_m)
        for xyz in (hover_xyz, release_xyz):
            result = self._motion.move_to_pose(xyz=xyz, quat_xyzw=self._motion.default_place_quat_xyzw())
            if not result.success:
                return SkillOutcome(False, FailureCode.MOTION_FAILED, result.message)
        opened = self._motion.open_gripper()
        if not opened.success:
            return SkillOutcome(False, FailureCode.MOTION_FAILED, opened.message)
        self._motion.move_to_pose(xyz=hover_xyz, quat_xyzw=self._motion.default_place_quat_xyzw())
        return SkillOutcome(True, FailureCode.NONE, "relative place complete", data={"held_object": held_object.object_id})

