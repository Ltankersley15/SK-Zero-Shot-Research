from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import json
import math
from typing import Any

import rclpy

from fr3_zero_shot.core.transforms import top_down_quaternion
from fr3_zero_shot.core.types import (
    ApproachFamily,
    FailureCode,
    GraspCandidate,
    ObjectHypothesis,
    SkillOutcome,
)
from fr3_zero_shot.execution.planning_scene_adapter import PlanningSceneAdapter
from fr3_zero_shot.execution.ros_motion_adapter import RosMotionAdapter
from fr3_zero_shot.grasping.grasp_planner import GraspPlanner
from fr3_zero_shot.grasping.grasp_candidate import is_valid
from fr3_zero_shot.grasping.grasp_family import approach_vector_xy
from fr3_zero_shot.live_pick_place import CUP_TOP_VIEW
from fr3_zero_shot.live_pick_place_utils import (
    colored_pickables_by_area,
    select_cup,
    side_family_away_from_keepout,
    source_still_on_table,
)
from fr3_zero_shot.perception.live_scene_census import LiveSceneCensusConfig, LiveSceneCensusNode
from fr3_zero_shot.perception.green_source_selector import select_green_from_captures
from fr3_zero_shot.planning.motion_sketch_planner import MotionSketchPlanner
from fr3_zero_shot.planning.pick_motion import PickMotionPlan, plan_pick_candidate


INITIAL_CUP_SIDE_VIEW = (0.56, -0.02, 0.42)
GREEN_REOBSERVE_VIEW = (0.62, 0.08, 0.42)
GREEN_WIDE_REOBSERVE_VIEW = (0.66, 0.08, 0.44)
GREEN_VERIFY_VIEW = (0.66, 0.12, 0.44)
GREEN_SIDE_STAGE_Z_M = 0.300
GREEN_SIDE_PREGRASP_Z_M = 0.095
GREEN_SIDE_PREGRASP_OFFSET_M = 0.140
GREEN_SIDE_GRASP_Z_M = 0.043
GREEN_SIDE_FACE_INSET_M = 0.004
GREEN_SIDE_ESCAPE_Z_M = 0.060
GREEN_SIDE_LIFT_Z_M = 0.245
GREEN_VERTICAL_DESCENT_BAN_DISTANCE_M = 0.22
GREEN_SIDE_RECOVERY_MIN_CLEARANCE_M = 0.04
GREEN_SEPARATED_TOP_DOWN_RECOVERY_CLEARANCE_M = 0.13
GREEN_RECOVERY_TOP_DOWN_X_BIAS_M = 0.008
GREEN_RECOVERY_TOP_DOWN_Y_BIAS_M = 0.008
GREEN_SIDE_MAX_PREGRASP_X_M = 0.775
GREEN_SIDE_EXEC_MAX_X_M = 0.690
GREEN_OCCLUSION_CENTER_CORRECTION_M = 0.0
GREEN_CLOSE_WIDTH_M = 0.014
GREEN_CONTACT_MARGIN_M = 0.001
GREEN_SIDE_RIGHT_DOWN_TILT_DEG = 40.0
GREEN_SIDE_BACK_DOWN_TILT_DEG = 35.0
GREEN_YAWED_EDGE_AWAY_OFFSET_M = 0.008
GREEN_YAWED_APPROACH_AWAY_OFFSET_M = 0.055
GREEN_YAWED_ESCAPE_AWAY_OFFSET_M = 0.055
GREEN_YAWED_ESCAPE_Z_M = GREEN_YAWED_GRASP_Z_M = 0.055
GREEN_YAWED_LOW_APPROACH_Z_M = 0.125
GREEN_YAWED_PREGRASP_Z_M = 0.140
GREEN_YAWED_LIFT_Z_M = 0.245
GREEN_REACH_IN_YAW_OFFSET_RAD = math.pi / 2.0


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Pick the partially occluded green cube while avoiding the cup.')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--llm-strategy', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--no-llm', dest='llm_strategy', action='store_false')
    parser.add_argument('--strategy-model', default='gpt-5-mini')
    parser.add_argument('--strategy-timeout-sec', type=float, default=20.0)
    parser.add_argument('--timeout-sec', type=float, default=8.0)
    parser.add_argument('--table-z', type=float, default=0.02)
    parser.add_argument('--position-tolerance', type=float, default=0.012)
    parser.add_argument('--orientation-tolerance', type=float, default=0.35)
    args = parser.parse_args(argv)

    rclpy.init()
    node = LiveSceneCensusNode(LiveSceneCensusConfig(table_z_m=float(args.table_z)))
    try:
        motion = RosMotionAdapter(
            node,
            position_tolerance_m=float(args.position_tolerance),
            orientation_tolerance_rad=float(args.orientation_tolerance),
        )
        motion_sketch_planner = MotionSketchPlanner(
            model=str(args.strategy_model),
            timeout_sec=float(args.strategy_timeout_sec),
        )
        captures = [
            ('initial_cup_side', _capture_from_pose(node=node, motion=motion, xyz=INITIAL_CUP_SIDE_VIEW, timeout_sec=float(args.timeout_sec))),
            ('cup_top', _capture_from_pose(node=node, motion=motion, xyz=CUP_TOP_VIEW, timeout_sec=float(args.timeout_sec))),
        ]
        green = select_green_from_captures(captures)
        cup, cup_reason = select_cup(captures)
        if green is None or cup is None:
            return _finish(
                False,
                captures=captures,
                outcome=SkillOutcome(
                    False,
                    FailureCode.SOURCE_NOT_VISIBLE,
                    f'need green source and cup keepout; green={green is not None} cup={cup is not None} reason={cup_reason}',
                ),
            )

        initial_motion = _plan_green_motion(
            planner=motion_sketch_planner,
            command='pick up the green cube while avoiding the cup',
            phase='initial_observation',
            green=green,
            cup=cup,
            use_llm=bool(args.llm_strategy),
        )
        if initial_motion.sketch.planner_source == 'rejected':
            return _finish(
                False,
                captures=captures,
                green=green,
                cup=cup,
                cup_reason=cup_reason,
                motion_sketch={'initial': initial_motion},
                outcome=SkillOutcome(False, FailureCode.NO_VALID_GRASP, 'LLM motion sketch rejected before reobserve'),
            )
        if initial_motion.sketch.reobserve.required:
            captures.extend(
                [
                    ('green_reobserve', _capture_from_pose(node=node, motion=motion, xyz=GREEN_REOBSERVE_VIEW, timeout_sec=float(args.timeout_sec))),
                    ('green_wide_reobserve', _capture_from_pose(node=node, motion=motion, xyz=GREEN_WIDE_REOBSERVE_VIEW, timeout_sec=float(args.timeout_sec))),
                ]
            )
            green = select_green_from_captures(captures) or green
            cup, cup_reason = select_cup(captures)
            if cup is None:
                return _finish(
                    False,
                    captures=captures,
                    green=green,
                    cup_reason=cup_reason,
                    motion_sketch={'initial': initial_motion},
                    outcome=SkillOutcome(False, FailureCode.TARGET_NOT_VISIBLE, 'cup lost after reobserve'),
                )

        green = _green_occlusion_corrected(green, cup)
        execution_motion = _plan_green_motion(
            planner=motion_sketch_planner,
            command='pick up the green cube while avoiding the cup',
            phase='post_reobserve_strategy',
            green=green,
            cup=cup,
            use_llm=bool(args.llm_strategy),
        )
        if execution_motion.sketch.planner_source == 'rejected':
            return _finish(
                False,
                captures=captures,
                green=green,
                cup=cup,
                cup_reason=cup_reason,
                motion_sketch={'initial': initial_motion, 'execution': execution_motion},
                outcome=SkillOutcome(False, FailureCode.NO_VALID_GRASP, 'LLM motion sketch rejected after reobserve'),
            )

        scene = PlanningSceneAdapter(node)
        if args.execute and not scene.apply_cup_keepout(cup, table_z=float(args.table_z)):
            return _finish(
                False,
                captures=captures,
                green=green,
                cup=cup,
                cup_reason=cup_reason,
                motion_sketch={'initial': initial_motion, 'execution': execution_motion},
                outcome=SkillOutcome(False, FailureCode.MOTION_FAILED, 'failed to apply cup planning-scene keepout'),
            )

        approach = execution_motion.compile_result.preferred_family
        candidate = _candidate_for_motion_plan(green, cup, execution_motion)
        if not args.execute:
            return _finish(
                candidate is not None,
                execute=False,
                captures=captures,
                green=green,
                cup=cup,
                cup_reason=cup_reason,
                approach=approach,
                candidate=candidate,
                motion_sketch={'initial': initial_motion, 'execution': execution_motion},
            )
        if candidate is None:
            return _finish(
                False,
                captures=captures,
                green=green,
                cup=cup,
                cup_reason=cup_reason,
                approach=approach,
                motion_sketch={'initial': initial_motion, 'execution': execution_motion},
                outcome=SkillOutcome(False, FailureCode.NO_VALID_GRASP, 'motion sketch compiled no valid green grasp'),
            )

        outcome = _run_motion_sketch_pick(motion=motion, candidate=candidate, cup=cup)
        # Stay at the lifted carry pose for the immediate carry check. A separate
        # verification sweep can cross the cup area and is not needed when the
        # positive gripper-contact signal is available.
        post_objects = node.capture(timeout_sec=float(args.timeout_sec))
        source_absent = not source_still_on_table(green, post_objects, table_z=float(args.table_z))
        contact = _green_contact(motion)
        green_on_table = _green_visible_on_table(post_objects, table_z=float(args.table_z))
        success = bool(outcome.code != FailureCode.MOTION_FAILED and source_absent and contact and not green_on_table)
        recovery = None
        ambiguous_carry = bool(outcome.success and source_absent and not green_on_table and not contact)
        if not success and not ambiguous_carry:
            recovery_view = _capture_from_pose(
                node=node,
                motion=motion,
                xyz=CUP_TOP_VIEW,
                timeout_sec=float(args.timeout_sec),
            )
            captures.append(('green_recovery_reobserve', recovery_view))
            post_objects = recovery_view
            recovery_green = _select_green(post_objects)
            recovery_clearance = _xy_distance(recovery_green, cup) if recovery_green is not None else None
            recovery = {
                'attempted': False,
                'clearance_m': recovery_clearance,
                'llm_authorized': bool(execution_motion.sketch.allow_recovery),
                'mode': 'not_selected',
            }
            if _green_motion_recovery_allowed(execution_motion, recovery_green, cup):
                recovery_green = _green_occlusion_corrected(recovery_green, cup)
                recovery_candidate = _side_away_candidate_for_objects(recovery_green, cup)
                recovery['mode'] = 'side_entry_only'
                recovery['attempted'] = recovery_candidate is not None
                recovery['candidate'] = recovery_candidate
                if recovery_candidate is not None:
                    outcome = _run_side_entry_pick(motion=motion, candidate=recovery_candidate)
                    post_objects = node.capture(timeout_sec=float(args.timeout_sec))
                    source_absent = not source_still_on_table(green, post_objects, table_z=float(args.table_z))
                    contact = _green_contact(motion)
                    green_on_table = _green_visible_on_table(post_objects, table_z=float(args.table_z))
                    success = bool(outcome.code != FailureCode.MOTION_FAILED and source_absent and contact and not green_on_table)
        proof_checks = None
        if success:
            proof_objects = _capture_from_pose(
                node=node,
                motion=motion,
                xyz=CUP_TOP_VIEW,
                timeout_sec=float(args.timeout_sec),
            )
            captures.append(('green_final_proof', proof_objects))
            proof_scene_valid = bool(proof_objects)
            proof_source_absent = not source_still_on_table(green, proof_objects, table_z=float(args.table_z))
            proof_contact = _green_contact(motion)
            proof_green_on_table = _green_visible_on_table(proof_objects, table_z=float(args.table_z))
            proof_checks = {
                'final_proof_scene_valid': proof_scene_valid,
                'final_proof_source_absent': proof_source_absent,
                'final_proof_gripper_contact': proof_contact,
                'final_proof_green_visible_on_table': proof_green_on_table,
            }
            post_objects = proof_objects
            source_absent = proof_source_absent
            contact = proof_contact
            green_on_table = proof_green_on_table
            success = bool(proof_scene_valid and proof_source_absent and proof_contact and not proof_green_on_table)
        if not success:
            original_outcome = outcome
            outcome = SkillOutcome(
                False,
                FailureCode.CARRY_VERIFY_FAILED,
                'green carry verification failed',
                data={
                    'ambiguous_carry_held_for_visual_check': ambiguous_carry,
                    'pick_success': original_outcome.success,
                    'original_outcome_code': original_outcome.code.value,
                    'original_outcome_message': original_outcome.message,
                    'source_absent_after_pick': source_absent,
                    'gripper_contact_after_pick': contact,
                    'green_visible_on_table_after_pick': green_on_table,
                    'proof_checks': proof_checks,
                },
            )
        return _finish(
            success,
            execute=True,
            captures=captures,
            green=green,
            cup=cup,
            cup_reason=cup_reason,
            approach=approach,
            candidate=candidate,
            outcome=outcome,
            recovery=recovery,
            planner_source=execution_motion.sketch.planner_source,
            motion_sketch={'initial': initial_motion, 'execution': execution_motion},
            post_objects=post_objects,
            checks={
                'source_absent_after_pick': source_absent,
                'gripper_contact_after_pick': contact,
                'green_visible_on_table_after_pick': green_on_table,
                'proof': proof_checks,
            },
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _select_green(objects: list[ObjectHypothesis]) -> ObjectHypothesis | None:
    green = colored_pickables_by_area(objects, 'green')
    if not green:
        return None
    return green[0]


def _plan_green_motion(
    *,
    planner: MotionSketchPlanner,
    command: str,
    phase: str,
    green: ObjectHypothesis,
    cup: ObjectHypothesis,
    use_llm: bool,
) -> PickMotionPlan:
    return plan_pick_candidate(
        planner=planner,
        command=command,
        phase=phase,
        source=green,
        avoid_objects=(cup,),
        grasp_planner=_green_motion_grasp_planner(),
        use_llm=use_llm,
        side_retention_risk=True,
    )


def _candidate_for_motion_plan(
    green: ObjectHypothesis,
    cup: ObjectHypothesis,
    motion_plan: PickMotionPlan,
) -> GraspCandidate | None:
    if _xy_distance(green, cup) is not None and _xy_distance(green, cup) < GREEN_VERTICAL_DESCENT_BAN_DISTANCE_M:
        side = (
            _side_away_candidate(motion_plan, green, cup)
            or _side_right_candidate(motion_plan)
            or _preferred_side_candidate(motion_plan)
            or _best_valid_side_candidate(motion_plan)
        )
        if side is not None and side.family == ApproachFamily.SIDE_BACK:
            return _green_yawed_top_down_candidate(green, cup)
        return None if side is None else _tuned_green_side_candidate(side)
    candidate = _preferred_side_candidate(motion_plan) or motion_plan.candidate
    if candidate is None:
        return None
    if candidate.family.value.startswith('side'):
        if not _side_candidate_live_reachable(candidate):
            candidate = _side_right_candidate(motion_plan) or _best_valid_side_candidate(motion_plan)
        if candidate is None:
            return None
        return None if candidate is None else _tuned_green_side_candidate(candidate)
    if candidate.family in {ApproachFamily.TOP_DOWN, ApproachFamily.TOP_DOWN_YAW_45}:
        return _green_yawed_top_down_candidate(green, cup)
    return candidate


def _motion_sketch_requires_side(motion_plan: PickMotionPlan) -> bool:
    text = ' '.join(
        [
            motion_plan.sketch.grasp_intent.approach_direction,
            motion_plan.sketch.grasp_intent.grasp_region,
            motion_plan.sketch.grasp_intent.contact_style,
            ' '.join(motion_plan.sketch.constraints),
        ]
    ).lower()
    return 'require_side_grasp' in text or 'side_only' in text


def _preferred_side_candidate(motion_plan: PickMotionPlan) -> GraspCandidate | None:
    preferred = motion_plan.compile_result.preferred_family
    if not preferred.value.startswith('side'):
        return None
    for candidate in motion_plan.compile_result.candidates:
        if candidate.family == preferred and is_valid(candidate, min_keepout_clearance_m=0.006):
            return candidate
    return None


def _best_valid_side_candidate(motion_plan: PickMotionPlan) -> GraspCandidate | None:
    valid = [
        candidate
        for candidate in motion_plan.compile_result.candidates
        if (
            candidate.family.value.startswith('side')
            and _side_candidate_live_reachable(candidate)
            and is_valid(candidate, min_keepout_clearance_m=0.006)
        )
    ]
    if not valid:
        return None
    return max(valid, key=lambda candidate: (candidate.keepout_clearance_m, candidate.clearance_score))


def _side_away_candidate(
    motion_plan: PickMotionPlan,
    green: ObjectHypothesis,
    cup: ObjectHypothesis,
) -> GraspCandidate | None:
    side = side_family_away_from_keepout(green, cup)
    for candidate in motion_plan.compile_result.candidates:
        if (
            candidate.family == side
            and _side_candidate_live_reachable(candidate)
            and is_valid(candidate, min_keepout_clearance_m=0.006)
        ):
            return candidate
    return None


def _side_right_candidate(motion_plan: PickMotionPlan) -> GraspCandidate | None:
    for candidate in motion_plan.compile_result.candidates:
        if (
            candidate.family == ApproachFamily.SIDE_RIGHT
            and _side_candidate_live_reachable(candidate)
            and is_valid(candidate, min_keepout_clearance_m=0.006)
        ):
            return candidate
    return None


def _side_candidate_live_reachable(candidate: GraspCandidate) -> bool:
    return float(candidate.pregrasp_xyz[0]) <= GREEN_SIDE_MAX_PREGRASP_X_M


def _green_motion_recovery_allowed(
    motion_plan: PickMotionPlan,
    recovery_green: ObjectHypothesis | None,
    cup: ObjectHypothesis,
) -> bool:
    # Recovery for this live path is side-entry only. A vertical top-down
    # recovery near the cup is the observed collision failure mode.
    recovery_clearance = _xy_distance(recovery_green, cup) if recovery_green is not None else None
    return bool(
        motion_plan.sketch.allow_recovery
        and recovery_clearance is not None
        and recovery_clearance >= GREEN_SIDE_RECOVERY_MIN_CLEARANCE_M
    )


def _green_separated_top_down_recovery_allowed(
    motion_plan: PickMotionPlan,
    recovery_green: ObjectHypothesis | None,
    cup: ObjectHypothesis,
) -> bool:
    _ = (motion_plan, recovery_green, cup)
    return False


def _side_away_candidate_for_objects(
    green: ObjectHypothesis,
    cup: ObjectHypothesis,
) -> GraspCandidate | None:
    side = side_family_away_from_keepout(green, cup)
    candidates = _green_motion_grasp_planner().generate_candidates(source=green, avoid_objects=(cup,))
    for candidate in candidates:
        if (
            candidate.family == side
            and _side_candidate_live_reachable(candidate)
            and is_valid(candidate, min_keepout_clearance_m=0.006)
        ):
            return _tuned_green_side_candidate(candidate)
    for candidate in candidates:
        if (
            candidate.family == ApproachFamily.SIDE_RIGHT
            and _side_candidate_live_reachable(candidate)
            and is_valid(candidate, min_keepout_clearance_m=0.006)
        ):
            return _tuned_green_side_candidate(candidate)
    valid_sides = [
        candidate
        for candidate in candidates
        if (
            candidate.family.value.startswith('side')
            and _side_candidate_live_reachable(candidate)
            and is_valid(candidate, min_keepout_clearance_m=0.006)
        )
    ]
    if not valid_sides:
        return None
    return _tuned_green_side_candidate(max(valid_sides, key=lambda candidate: (candidate.keepout_clearance_m, candidate.clearance_score)))


def _run_motion_sketch_pick(
    *,
    motion: RosMotionAdapter,
    candidate: GraspCandidate,
    cup: ObjectHypothesis,
) -> SkillOutcome:
    if candidate.family.value.startswith('side'):
        return _run_side_entry_pick(motion=motion, candidate=candidate)
    return _run_yawed_top_down_pick(motion=motion, candidate=candidate, cup=cup)


def _green_occlusion_corrected(green: ObjectHypothesis, cup: ObjectHypothesis) -> ObjectHypothesis:
    dx = float(cup.xyz[0] - green.xyz[0])
    dy = float(cup.xyz[1] - green.xyz[1])
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1e-6:
        return green
    scale = GREEN_OCCLUSION_CENTER_CORRECTION_M / norm
    xyz = (
        float(green.xyz[0] + dx * scale),
        float(green.xyz[1] + dy * scale),
        float(green.xyz[2]),
    )
    metadata = dict(green.metadata)
    metadata['occlusion_center_correction_m'] = GREEN_OCCLUSION_CENTER_CORRECTION_M
    return replace(green, xyz=xyz, metadata=metadata)


def _green_top_down_recovery_biased(green: ObjectHypothesis) -> ObjectHypothesis:
    metadata = dict(green.metadata)
    metadata['top_down_recovery_xy_bias_m'] = (
        GREEN_RECOVERY_TOP_DOWN_X_BIAS_M,
        GREEN_RECOVERY_TOP_DOWN_Y_BIAS_M,
    )
    return replace(
        green,
        xyz=(
            float(green.xyz[0] + GREEN_RECOVERY_TOP_DOWN_X_BIAS_M),
            float(green.xyz[1] + GREEN_RECOVERY_TOP_DOWN_Y_BIAS_M),
            float(green.xyz[2]),
        ),
        metadata=metadata,
    )


def _green_side_grasp_planner() -> GraspPlanner:
    return GraspPlanner(
        side_pregrasp_offset_m=GREEN_SIDE_PREGRASP_OFFSET_M,
        retreat_lift_m=0.22,
        min_keepout_clearance_m=0.006,
    )


def _green_top_down_grasp_planner() -> GraspPlanner:
    return GraspPlanner(
        retreat_lift_m=0.22,
        top_down_x_bias_m=0.0,
        top_down_surface_penetration_m=0.015,
        min_keepout_clearance_m=0.006,
    )


def _green_motion_grasp_planner() -> GraspPlanner:
    return GraspPlanner(
        side_pregrasp_offset_m=GREEN_SIDE_PREGRASP_OFFSET_M,
        retreat_lift_m=0.22,
        top_down_x_bias_m=0.0,
        top_down_surface_penetration_m=0.015,
        min_keepout_clearance_m=0.006,
    )


def _green_yawed_top_down_candidate(green: ObjectHypothesis, cup: ObjectHypothesis) -> GraspCandidate:
    away_x, away_y = _away_from_cup_xy(green, cup)
    x = float(green.xyz[0] + away_x * GREEN_YAWED_EDGE_AWAY_OFFSET_M)
    y = float(green.xyz[1] + away_y * GREEN_YAWED_EDGE_AWAY_OFFSET_M)
    yaw_rad = _green_reach_in_yaw_rad(green, cup)
    return GraspCandidate(
        family=ApproachFamily.TOP_DOWN,
        pregrasp_xyz=(x, y, GREEN_YAWED_PREGRASP_Z_M),
        grasp_xyz=(x, y, GREEN_YAWED_GRASP_Z_M),
        retreat_xyz=(x, y, GREEN_YAWED_LIFT_Z_M),
        quat_xyzw=top_down_quaternion(yaw_rad),
        clearance_score=1.0,
        ik_feasible=True,
        collision_free=True,
        keepout_clearance_m=float('inf'),
        reason='green occlusion overreach grasp keeps the wrist outside the cup line and rotates the wrist so fingers close back toward the cube',
    )


def _away_from_cup_xy(source: ObjectHypothesis, cup: ObjectHypothesis) -> tuple[float, float]:
    return _away_from_cup_point_xy(source.xyz, cup)


def _away_from_cup_point_xy(
    xyz: tuple[float, float, float],
    cup: ObjectHypothesis,
) -> tuple[float, float]:
    dx = float(xyz[0] - cup.xyz[0])
    dy = float(xyz[1] - cup.xyz[1])
    norm = max(1e-6, math.sqrt(dx * dx + dy * dy))
    return (dx / norm, dy / norm)


def _green_reach_in_yaw_rad(source: ObjectHypothesis, cup: ObjectHypothesis) -> float:
    return _reach_in_yaw_from_point_rad(source.xyz, cup)


def _reach_in_yaw_from_point_rad(
    xyz: tuple[float, float, float],
    cup: ObjectHypothesis,
) -> float:
    away_x, away_y = _away_from_cup_point_xy(xyz, cup)
    return math.atan2(away_y, away_x) + GREEN_REACH_IN_YAW_OFFSET_RAD


def _tuned_green_side_candidate(candidate):
    ax, ay = approach_vector_xy(candidate.family)
    pregrasp_x = float(candidate.pregrasp_xyz[0])
    if candidate.family == ApproachFamily.SIDE_BACK:
        pregrasp_x = min(pregrasp_x, GREEN_SIDE_EXEC_MAX_X_M)
    grasp = (
        float(candidate.grasp_xyz[0] + ax * GREEN_SIDE_FACE_INSET_M),
        float(candidate.grasp_xyz[1] + ay * GREEN_SIDE_FACE_INSET_M),
        GREEN_SIDE_GRASP_Z_M,
    )
    pregrasp = (
        pregrasp_x,
        float(candidate.pregrasp_xyz[1]),
        GREEN_SIDE_PREGRASP_Z_M,
    )
    retreat = (
        pregrasp_x if candidate.family == ApproachFamily.SIDE_BACK else float(candidate.retreat_xyz[0]),
        float(candidate.retreat_xyz[1]),
        max(float(candidate.retreat_xyz[2]), GREEN_SIDE_GRASP_Z_M + 0.11),
    )
    return replace(
        candidate,
        pregrasp_xyz=pregrasp,
        grasp_xyz=grasp,
        retreat_xyz=retreat,
        quat_xyzw=_green_side_quaternion(candidate.family, candidate.quat_xyzw),
        reason=f'{candidate.reason}; corrected side wrist and target near cube face',
    )


def _green_side_quaternion(
    family: ApproachFamily,
    fallback: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if family == ApproachFamily.SIDE_RIGHT:
        return _quat_from_columns(
            x_axis=_normalized((0.0, math.sin(math.radians(GREEN_SIDE_RIGHT_DOWN_TILT_DEG)), -math.cos(math.radians(GREEN_SIDE_RIGHT_DOWN_TILT_DEG)))),
            y_axis=(1.0, 0.0, 0.0),
            z_axis=_normalized((0.0, -math.cos(math.radians(GREEN_SIDE_RIGHT_DOWN_TILT_DEG)), -math.sin(math.radians(GREEN_SIDE_RIGHT_DOWN_TILT_DEG)))),
        )
    if family == ApproachFamily.SIDE_BACK:
        return _quat_from_columns(
            x_axis=_normalized((math.sin(math.radians(GREEN_SIDE_BACK_DOWN_TILT_DEG)), 0.0, -math.cos(math.radians(GREEN_SIDE_BACK_DOWN_TILT_DEG)))),
            y_axis=(0.0, -1.0, 0.0),
            z_axis=_normalized((-math.cos(math.radians(GREEN_SIDE_BACK_DOWN_TILT_DEG)), 0.0, -math.sin(math.radians(GREEN_SIDE_BACK_DOWN_TILT_DEG)))),
        )
    return fallback


def _normalized(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vec))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0)
    return tuple(float(v) / norm for v in vec)  # type: ignore[return-value]


def _quat_from_columns(
    *,
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    norm = math.sqrt((qx * qx) + (qy * qy) + (qz * qz) + (qw * qw))
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def _run_side_entry_pick(
    *,
    motion: RosMotionAdapter,
    candidate,
) -> SkillOutcome:
    pregrasp = candidate.pregrasp_xyz
    grasp = candidate.grasp_xyz
    low_pregrasp = (float(pregrasp[0]), float(pregrasp[1]), float(grasp[2]))
    lateral_escape = low_pregrasp
    pull = (float(pregrasp[0]), float(pregrasp[1]), GREEN_SIDE_ESCAPE_Z_M)
    lift = (pull[0], pull[1], max(float(candidate.retreat_xyz[2]), GREEN_SIDE_LIFT_Z_M))
    stage = (float(pregrasp[0]), float(pregrasp[1]), GREEN_SIDE_STAGE_Z_M)

    opened = motion.open_gripper()
    if not opened.success:
        return SkillOutcome(False, FailureCode.MOTION_FAILED, opened.message)
    for label, xyz, quat in (
        ('stage', stage, motion.default_place_quat_xyzw()),
        ('pregrasp', pregrasp, candidate.quat_xyzw),
        ('low_pregrasp_outside_keepout', low_pregrasp, candidate.quat_xyzw),
        ('grasp', grasp, candidate.quat_xyzw),
    ):
        result = motion.move_to_pose(xyz=xyz, quat_xyzw=quat)
        if not result.success:
            return SkillOutcome(False, FailureCode.MOTION_FAILED, f'{label} motion failed: {result.message}')
    closed = motion.close_gripper(position_m=GREEN_CLOSE_WIDTH_M, contact_margin_m=GREEN_CONTACT_MARGIN_M)
    if not closed.success:
        return SkillOutcome(False, FailureCode.MOTION_FAILED, closed.message)
    for label, xyz in (('lateral_escape', lateral_escape), ('pull_away', pull), ('lift', lift)):
        result = motion.move_to_pose(xyz=xyz, quat_xyzw=candidate.quat_xyzw)
        if not result.success:
            return SkillOutcome(False, FailureCode.MOTION_FAILED, f'{label} motion failed: {result.message}')
    return SkillOutcome(
        True,
        FailureCode.NONE,
        'green side-entry pick complete',
        data={
            'pregrasp': pregrasp,
            'low_pregrasp_outside_keepout': low_pregrasp,
            'grasp': grasp,
            'lateral_escape': lateral_escape,
            'pull_away': pull,
            'lift': lift,
        },
    )


def _run_top_down_recovery_pick(*, motion: RosMotionAdapter, candidate) -> SkillOutcome:
    opened = motion.open_gripper()
    if not opened.success:
        return SkillOutcome(False, FailureCode.MOTION_FAILED, opened.message)
    for label, xyz in (('recovery_pregrasp', candidate.pregrasp_xyz), ('recovery_grasp', candidate.grasp_xyz)):
        result = motion.move_to_pose(xyz=xyz, quat_xyzw=candidate.quat_xyzw)
        if not result.success:
            return SkillOutcome(False, FailureCode.MOTION_FAILED, f'{label} motion failed: {result.message}')
    closed = motion.close_gripper(position_m=GREEN_CLOSE_WIDTH_M, contact_margin_m=GREEN_CONTACT_MARGIN_M)
    if not closed.success:
        return SkillOutcome(False, FailureCode.MOTION_FAILED, closed.message)
    lift = motion.move_to_pose(xyz=candidate.retreat_xyz, quat_xyzw=candidate.quat_xyzw)
    if not lift.success:
        return SkillOutcome(False, FailureCode.MOTION_FAILED, f'recovery lift failed: {lift.message}')
    if not _green_contact(motion):
        return SkillOutcome(False, FailureCode.CARRY_VERIFY_FAILED, 'recovery gripper lost contact after lift')
    return SkillOutcome(True, FailureCode.NONE, 'green top-down recovery pick complete', data={'grasp': candidate})


def _run_yawed_top_down_pick(
    *,
    motion: RosMotionAdapter,
    candidate: GraspCandidate,
    cup: ObjectHypothesis,
) -> SkillOutcome:
    pseudo_source = replace(
        cup,
        xyz=(
            float(candidate.grasp_xyz[0]),
            float(candidate.grasp_xyz[1]),
            float(candidate.grasp_xyz[2]),
        ),
    )
    away_x, away_y = _away_from_cup_xy(pseudo_source, cup)
    low_approach = (
        float(candidate.grasp_xyz[0] + away_x * GREEN_YAWED_APPROACH_AWAY_OFFSET_M),
        float(candidate.grasp_xyz[1] + away_y * GREEN_YAWED_APPROACH_AWAY_OFFSET_M),
        GREEN_YAWED_LOW_APPROACH_Z_M,
    )
    high_approach = (
        low_approach[0],
        low_approach[1],
        max(float(candidate.pregrasp_xyz[2]), GREEN_YAWED_PREGRASP_Z_M),
    )
    low_entry = (
        low_approach[0],
        low_approach[1],
        GREEN_YAWED_GRASP_Z_M,
    )
    escape = (
        float(candidate.grasp_xyz[0] + away_x * GREEN_YAWED_ESCAPE_AWAY_OFFSET_M),
        float(candidate.grasp_xyz[1] + away_y * GREEN_YAWED_ESCAPE_AWAY_OFFSET_M),
        GREEN_YAWED_ESCAPE_Z_M,
    )
    pull_away = (
        escape[0],
        escape[1],
        GREEN_YAWED_LOW_APPROACH_Z_M,
    )
    lift = (
        escape[0],
        escape[1],
        max(float(candidate.retreat_xyz[2]), GREEN_YAWED_LIFT_Z_M),
    )
    opened = motion.open_gripper()
    if not opened.success:
        return SkillOutcome(False, FailureCode.MOTION_FAILED, opened.message)
    for label, xyz in (
        ('high_approach', high_approach),
        ('low_approach', low_approach),
        ('low_entry_outside_keepout', low_entry),
        ('grasp', candidate.grasp_xyz),
    ):
        result = motion.move_to_pose(xyz=xyz, quat_xyzw=candidate.quat_xyzw)
        if not result.success:
            return SkillOutcome(False, FailureCode.MOTION_FAILED, f'{label} motion failed: {result.message}')
    closed = motion.close_gripper(position_m=GREEN_CLOSE_WIDTH_M, contact_margin_m=GREEN_CONTACT_MARGIN_M)
    if not closed.success:
        return SkillOutcome(False, FailureCode.MOTION_FAILED, closed.message)
    for label, xyz in (('lateral_escape', escape), ('pull_away', pull_away), ('lift', lift)):
        result = motion.move_to_pose(xyz=xyz, quat_xyzw=candidate.quat_xyzw)
        if not result.success:
            return SkillOutcome(False, FailureCode.MOTION_FAILED, f'{label} failed: {result.message}')
    return SkillOutcome(
        True,
        FailureCode.NONE,
        'green yawed reach-in exposed-edge pick complete',
        data={
            'high_approach': high_approach,
            'low_approach': low_approach,
            'low_entry_outside_keepout': low_entry,
            'grasp': candidate,
            'lateral_escape': escape,
            'pull_away': pull_away,
            'lift': lift,
            'edge_away_offset_m': GREEN_YAWED_EDGE_AWAY_OFFSET_M,
            'approach_away_offset_m': GREEN_YAWED_APPROACH_AWAY_OFFSET_M,
            'escape_away_offset_m': GREEN_YAWED_ESCAPE_AWAY_OFFSET_M,
            'escape_z_m': GREEN_YAWED_ESCAPE_Z_M,
            'reach_in_yaw_rad': _reach_in_yaw_from_point_rad(candidate.grasp_xyz, cup),
        },
    )


def _green_contact(motion: RosMotionAdapter) -> bool:
    return motion.gripper_contact_detected(
        commanded_width=GREEN_CLOSE_WIDTH_M,
        margin_m=GREEN_CONTACT_MARGIN_M,
    )


def _green_visible_on_table(objects: list[ObjectHypothesis], *, table_z: float) -> bool:
    for obj in objects:
        if obj.color != 'green' or obj.container_like:
            continue
        raw_xyz = obj.metadata.get('raw_depth_base_xyz')
        if raw_xyz is None:
            return True
        if float(raw_xyz[2]) <= float(table_z) + 0.12:
            return True
    return False


def _xy_distance(a: ObjectHypothesis | None, b: ObjectHypothesis | None) -> float | None:
    if a is None or b is None:
        return None
    dx = float(a.xyz[0] - b.xyz[0])
    dy = float(a.xyz[1] - b.xyz[1])
    return (dx * dx + dy * dy) ** 0.5


def _capture_from_pose(
    *,
    node: LiveSceneCensusNode,
    motion: RosMotionAdapter,
    xyz: tuple[float, float, float],
    timeout_sec: float,
) -> list[ObjectHypothesis]:
    moved = motion.move_to_pose(xyz=xyz, quat_xyzw=motion.default_place_quat_xyzw())
    if not moved.success:
        return node.capture(timeout_sec=timeout_sec)
    return node.capture(timeout_sec=timeout_sec)


def _finish(success: bool, **payload: Any) -> int:
    payload['success'] = bool(success)
    print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    return 0 if success else 2


if __name__ == '__main__':
    raise SystemExit(main())
