import pytest

from fr3_zero_shot.core.types import GraspIntent, MotionSketch, ObjectHypothesis, ReobserveDirective, RetreatIntent

pytest.importorskip("rclpy")

from fr3_zero_shot.live_green_occluded_pick import (
    GREEN_SIDE_FACE_INSET_M,
    GREEN_SIDE_GRASP_Z_M,
    GREEN_SEPARATED_TOP_DOWN_RECOVERY_CLEARANCE_M,
    GREEN_SIDE_EXEC_MAX_X_M,
    GREEN_VERTICAL_DESCENT_BAN_DISTANCE_M,
    GREEN_YAWED_APPROACH_AWAY_OFFSET_M,
    GREEN_YAWED_EDGE_AWAY_OFFSET_M,
    GREEN_YAWED_ESCAPE_AWAY_OFFSET_M,
    _candidate_for_motion_plan,
    _green_motion_recovery_allowed,
    _green_reach_in_yaw_rad,
    _green_separated_top_down_recovery_allowed,
    _green_yawed_top_down_candidate,
)
from fr3_zero_shot.planning.motion_sketch_compiler import MotionSketchCompiler
from fr3_zero_shot.planning.pick_motion import PickMotionPlan


def _obj(object_id, *, xyz, color=None, label="object", container_like=False):
    return ObjectHypothesis(
        object_id=object_id,
        label=label,
        color=color,
        shape="cup" if container_like else "cube",
        xyz=xyz,
        footprint_xy=(0.08, 0.08) if container_like else (0.025, 0.025),
        height_m=0.08 if container_like else 0.04,
        bbox_xyxy=(0, 0, 20, 20),
        confidence=1.0,
        visible=True,
        pickable=not container_like,
        container_like=container_like,
    )


def test_green_yawed_candidate_biases_exposed_edge_away_from_cup():
    green = _obj("green", xyz=(0.62, 0.0, 0.04), color="green", label="green cube")
    cup = _obj("cup", xyz=(0.50, -0.10, 0.06), label="cup", container_like=True)

    candidate = _green_yawed_top_down_candidate(green, cup)

    assert candidate.grasp_xyz[0] > green.xyz[0]
    assert candidate.grasp_xyz[1] > green.xyz[1]
    assert _xy_distance(candidate.grasp_xyz, green.xyz) == pytest.approx(GREEN_YAWED_EDGE_AWAY_OFFSET_M)
    assert GREEN_YAWED_APPROACH_AWAY_OFFSET_M > GREEN_YAWED_EDGE_AWAY_OFFSET_M
    assert GREEN_YAWED_ESCAPE_AWAY_OFFSET_M >= GREEN_YAWED_APPROACH_AWAY_OFFSET_M
    assert _green_reach_in_yaw_rad(green, cup) == pytest.approx(2.2655346029916)


def test_green_candidate_honors_valid_preferred_side_family():
    green = _obj("green", xyz=(0.57, 0.0, 0.04), color="green", label="green cube")
    cup = _obj("cup", xyz=(0.50, -0.01, 0.06), label="cup", container_like=True)
    sketch = _motion_sketch(
        reason="mock side plan",
        approach_direction="exposed_side",
        grasp_region="outer_edge",
        wrist_intent="yaw_45_away_from_keepout",
        constraints=["avoid_cup", "pull_away_before_lift", "require_side_grasp"],
    )
    compiled = MotionSketchCompiler().compile(sketch=sketch, source=green, avoid_objects=(cup,))
    plan = PickMotionPlan(
        sketch=sketch,
        compile_result=compiled,
        candidate=compiled.candidate,
        prompt="",
        raw_response="",
        rejection_reason=None,
    )

    candidate = _candidate_for_motion_plan(green, cup, plan)

    assert candidate is not None
    assert candidate.family == compiled.preferred_family
    assert candidate.family.value.startswith("side")
    assert candidate.grasp_xyz[2] == pytest.approx(GREEN_SIDE_GRASP_Z_M)
    assert GREEN_SIDE_FACE_INSET_M >= 0.0


def test_green_candidate_uses_yawed_reach_in_when_true_away_side_is_low_ik_risk():
    green = _obj("green", xyz=(0.62, 0.0, 0.04), color="green", label="green cube")
    cup = _obj("cup", xyz=(0.50, -0.10, 0.06), label="cup", container_like=True)
    sketch = _motion_sketch(
        reason="mock yawed plan",
        approach_direction="top_edge",
        grasp_region="top_edge",
        wrist_intent="yaw_45_away_from_keepout",
        constraints=["avoid_cup", "pull_away_before_lift"],
    )
    compiled = MotionSketchCompiler().compile(sketch=sketch, source=green, avoid_objects=(cup,))
    assert compiled.candidate is not None
    assert compiled.candidate.family.value.startswith("top_down")
    plan = PickMotionPlan(sketch, compiled, compiled.candidate, "", "", None)

    candidate = _candidate_for_motion_plan(green, cup, plan)

    assert candidate is not None
    assert candidate.family.value == "top_down"
    assert candidate.pregrasp_xyz[0] > green.xyz[0]
    assert GREEN_SIDE_EXEC_MAX_X_M < compiled.candidates[-1].pregrasp_xyz[0]
    assert _xy_distance(green.xyz, cup.xyz) < GREEN_VERTICAL_DESCENT_BAN_DISTANCE_M


def test_green_recovery_is_side_entry_authorized_near_cup():
    green = _obj("green", xyz=(0.59, -0.04, 0.04), color="green", label="green cube")
    cup = _obj("cup", xyz=(0.50, -0.10, 0.06), label="cup", container_like=True)
    sketch = _motion_sketch(
        reason="mock recovery plan",
        approach_direction="top_edge",
        grasp_region="top_edge",
        wrist_intent="yaw_45_away_from_keepout",
        constraints=["avoid_cup", "pull_away_before_lift"],
    )
    compiled = MotionSketchCompiler().compile(sketch=sketch, source=green, avoid_objects=(cup,))
    plan = PickMotionPlan(sketch, compiled, compiled.candidate, "", "", None)

    assert _green_motion_recovery_allowed(plan, green, cup)
    assert not _green_separated_top_down_recovery_allowed(plan, green, cup)


def test_green_top_down_recovery_is_disabled_for_occluded_cup_path():
    green = _obj("green", xyz=(0.65, 0.04, 0.04), color="green", label="green cube")
    cup = _obj("cup", xyz=(0.50, -0.10, 0.06), label="cup", container_like=True)
    sketch = _motion_sketch(
        reason="mock separated recovery plan",
        approach_direction="exposed_side",
        grasp_region="outer_edge",
        wrist_intent="yaw_45_away_from_keepout",
        constraints=["avoid_cup", "low_lateral_entry", "pull_away_before_lift"],
    )
    compiled = MotionSketchCompiler().compile(sketch=sketch, source=green, avoid_objects=(cup,))
    plan = PickMotionPlan(sketch, compiled, compiled.candidate, "", "", None)

    assert _xy_distance(green.xyz, cup.xyz) > GREEN_SEPARATED_TOP_DOWN_RECOVERY_CLEARANCE_M
    assert not _green_separated_top_down_recovery_allowed(plan, green, cup)


def _xy_distance(a, b):
    return ((float(a[0] - b[0]) ** 2) + (float(a[1] - b[1]) ** 2)) ** 0.5


def _motion_sketch(
    *,
    reason,
    approach_direction,
    grasp_region,
    wrist_intent,
    constraints,
):
    return MotionSketch(
        reobserve=ReobserveDirective(False, None, None, "already observed"),
        grasp_intent=GraspIntent(approach_direction, grasp_region, wrist_intent, "edge constrained"),
        retreat_intent=RetreatIntent("pull_away_then_lift", "away from cup", True),
        constraints=constraints,
        avoid_refs=["cup"],
        allow_recovery=True,
        reason=reason,
        planner_source="mock",
    )
