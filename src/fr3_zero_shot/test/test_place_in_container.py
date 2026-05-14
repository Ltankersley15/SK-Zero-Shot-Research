from fr3_zero_shot.core.types import ObjectHypothesis
from fr3_zero_shot.execution.motion_adapter import AdapterResult, DryRunMotionAdapter
from fr3_zero_shot.skills.place_in_container import PlaceInContainerSkill, strict_release_gate


def _obj(object_id, xyz, footprint=(0.04, 0.04), container=False):
    return ObjectHypothesis(
        object_id=object_id,
        label=object_id,
        color=None,
        shape="cup" if container else "cube",
        xyz=xyz,
        footprint_xy=footprint,
        height_m=0.08 if container else 0.04,
        bbox_xyxy=(0, 0, 10, 10),
        confidence=0.9,
        visible=True,
        pickable=not container,
        container_like=container,
    )


def test_strict_release_gate_rejects_high_release():
    gate = strict_release_gate(
        actual_tcp_xyz=(0.5, 0.0, 0.24),
        commanded_tcp_xyz=(0.5, 0.0, 0.18),
        target_center_xy=(0.5, 0.0),
        center_tolerance_m=0.015,
        safe_release_z_m=0.18,
        rim_z_m=0.14,
    )

    assert not gate.ready
    assert gate.z_above_limit_m > 0.0


def test_place_in_container_opens_only_after_gate_passes():
    motion = DryRunMotionAdapter()
    motion.close_gripper()
    outcome = PlaceInContainerSkill(motion=motion).run(
        held_object=_obj("cube", (0.4, 0.0, 0.02)),
        container=_obj("cup", (0.5, 0.0, 0.04), footprint=(0.08, 0.08), container=True),
    )

    assert outcome.success
    assert any(command["type"] == "open_gripper" for command in motion.commands)


def test_place_in_container_refuses_release_when_corrective_gate_still_fails():
    class HighTcpMotion(DryRunMotionAdapter):
        def move_to_pose(self, *, xyz, quat_xyzw):
            self.commands.append({"type": "move_to_pose", "xyz": xyz, "quat_xyzw": quat_xyzw})
            self._tcp_pose = ((float(xyz[0]), float(xyz[1]), float(xyz[2] + 0.08)), tuple(quat_xyzw))
            return AdapterResult(True, "high tcp pose accepted")

    motion = HighTcpMotion()
    motion.close_gripper()
    outcome = PlaceInContainerSkill(motion=motion).run(
        held_object=_obj("cube", (0.4, 0.0, 0.02)),
        container=_obj("cup", (0.5, 0.0, 0.04), footprint=(0.08, 0.08), container=True),
    )

    assert not outcome.success
    assert not any(command["type"] == "open_gripper" for command in motion.commands)
