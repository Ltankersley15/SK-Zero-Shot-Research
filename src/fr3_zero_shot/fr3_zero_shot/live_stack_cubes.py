from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from typing import Any

import rclpy

from fr3_zero_shot.core.types import ApproachFamily, FailureCode, GraspCandidate, ObjectHypothesis, SkillOutcome
from fr3_zero_shot.execution.ros_motion_adapter import RosMotionAdapter
from fr3_zero_shot.grasping.grasp_planner import GraspPlanner
from fr3_zero_shot.live_pick_place import CUP_TOP_VIEW
from fr3_zero_shot.live_pick_place_sequence import SOURCE_VIEW
from fr3_zero_shot.live_pick_place_utils import colored_pickables_by_area
from fr3_zero_shot.perception.live_scene_census import LiveSceneCensusConfig, LiveSceneCensusNode
from fr3_zero_shot.planning.motion_sketch_planner import MotionSketchPlanner
from fr3_zero_shot.planning.pick_motion import plan_pick_candidate
from fr3_zero_shot.skills.pick import PickSkill


STACK_VIEW = (0.56, 0.12, 0.46)
SMALL_RED_VIEW = (0.52, -0.02, 0.42)
FAR_SMALL_RED_VIEW = (0.48, -0.22, 0.38)
STACK_RELEASE_CLEARANCE_M = 0.055
STACK_HOVER_CLEARANCE_M = 0.180
STACK_MIN_HOVER_Z_M = 0.320
STACK_SOURCE_STAGE_Z_M = 0.320
STACK_TCP_BOTTOM_CLEARANCE_M = 0.035
STACK_CLOSE_WIDTH_M = 0.014
STACK_CONTACT_MARGIN_M = 0.001
STACK_GRASP_NUDGE_OFFSETS_M = (
    (0.0, 0.0),
    (-0.012, 0.0),
    (0.012, 0.0),
    (0.0, -0.012),
    (0.0, 0.012),
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Stack two cubes on a base cube in Isaac.')
    parser.add_argument('--base-color', default='blue', choices=['blue'])
    parser.add_argument(
        '--stack-sources',
        nargs=2,
        default=['large_red', 'small_red'],
        metavar=('FIRST', 'SECOND'),
        help='Two stack sources, e.g. --stack-sources yellow small_red.',
    )
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--timeout-sec', type=float, default=8.0)
    parser.add_argument('--table-z', type=float, default=0.02)
    parser.add_argument('--cube-height', type=float, default=0.04)
    parser.add_argument('--position-tolerance', type=float, default=0.025)
    parser.add_argument('--orientation-tolerance', type=float, default=0.45)
    parser.add_argument('--llm-motion-sketch', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sketch-model', default='gpt-5-mini')
    parser.add_argument('--sketch-timeout-sec', type=float, default=20.0)
    args = parser.parse_args(argv)

    rclpy.init()
    node = LiveSceneCensusNode(LiveSceneCensusConfig(table_z_m=float(args.table_z)))
    try:
        motion = RosMotionAdapter(
            node,
            position_tolerance_m=float(args.position_tolerance),
            orientation_tolerance_rad=float(args.orientation_tolerance),
        )
        objects = []
        for xyz in (SOURCE_VIEW, SMALL_RED_VIEW, FAR_SMALL_RED_VIEW, CUP_TOP_VIEW):
            objects.extend(_capture_from_pose(node=node, motion=motion, xyz=xyz, timeout_sec=float(args.timeout_sec)))

        blue = _select_base_blue(objects)
        first_source = _select_stack_source(objects, args.stack_sources[0])
        second_source = _select_stack_source(objects, args.stack_sources[1])
        if blue is None or first_source is None or second_source is None:
            return _finish(
                False,
                objects=objects,
                outcome=SkillOutcome(
                    False,
                    FailureCode.SOURCE_NOT_VISIBLE,
                    (
                        'need base blue and requested stack sources; '
                        f'blue={blue is not None} first={first_source is not None} second={second_source is not None}'
                    ),
                ),
            )
        if not args.execute:
            return _finish(
                True,
                execute=False,
                objects=objects,
                blue=blue,
                first_source=first_source,
                second_source=second_source,
                stack_sources=args.stack_sources,
            )

        stack_xy = (float(blue.xyz[0]), float(blue.xyz[1]))
        cube_height = max(0.02, float(args.cube_height))
        table_z = float(args.table_z)
        steps = []

        first = _pick_and_place_on_stack(
            color_name=str(args.stack_sources[0]),
            source=first_source,
            stack_xy=stack_xy,
            target_top_z=table_z + cube_height,
            node=node,
            motion=motion,
            timeout_sec=float(args.timeout_sec),
            table_z=table_z,
            use_llm_motion_sketch=bool(args.llm_motion_sketch),
            sketch_model=str(args.sketch_model),
            sketch_timeout_sec=float(args.sketch_timeout_sec),
        )
        steps.append(first)
        if not first['success']:
            return _finish(
                False,
                execute=True,
                blue=blue,
                first_source=first_source,
                second_source=second_source,
                stack_sources=args.stack_sources,
                steps=steps,
            )

        second = _pick_and_place_on_stack(
            color_name=str(args.stack_sources[1]),
            source=second_source,
            stack_xy=stack_xy,
            target_top_z=table_z + cube_height * 2.0,
            node=node,
            motion=motion,
            timeout_sec=float(args.timeout_sec),
            table_z=table_z,
            use_llm_motion_sketch=bool(args.llm_motion_sketch),
            sketch_model=str(args.sketch_model),
            sketch_timeout_sec=float(args.sketch_timeout_sec),
        )
        steps.append(second)
        final_objects = _capture_from_pose(node=node, motion=motion, xyz=STACK_VIEW, timeout_sec=float(args.timeout_sec))
        success = bool(all(step['success'] for step in steps))
        return _finish(
            success,
            execute=True,
            blue=blue,
            first_source=first_source,
            second_source=second_source,
            stack_sources=args.stack_sources,
            steps=steps,
            final_objects=final_objects,
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _select_base_blue(objects: list[ObjectHypothesis]) -> ObjectHypothesis | None:
    blue = colored_pickables_by_area(objects, 'blue')
    if not blue:
        return None
    return blue[0]


def _select_red_stack_pair(reds: list[ObjectHypothesis]) -> tuple[ObjectHypothesis, ObjectHypothesis]:
    if len(reds) < 2:
        raise ValueError('need at least two red candidates')
    if abs(float(reds[0].xyz[1]) - float(reds[-1].xyz[1])) > 0.12:
        return max(reds, key=lambda obj: obj.xyz[1]), min(reds, key=lambda obj: obj.xyz[1])
    return reds[0], reds[-1]


def _select_stack_source(objects: list[ObjectHypothesis], source_name: str) -> ObjectHypothesis | None:
    normalized = source_name.strip().lower().replace('-', '_')
    if normalized in {'yellow', 'yellow_cube'}:
        yellow = colored_pickables_by_area(objects, 'yellow')
        return yellow[0] if yellow else None
    if normalized in {'large_red', 'large_red_cube'}:
        reds = colored_pickables_by_area(objects, 'red')
        return _select_red_stack_pair(reds)[0] if len(reds) >= 2 else (reds[0] if reds else None)
    if normalized in {'small_red', 'small_red_cube'}:
        reds = colored_pickables_by_area(objects, 'red')
        return _select_red_stack_pair(reds)[1] if len(reds) >= 2 else (reds[0] if reds else None)
    if normalized in {'red', 'red_cube'}:
        reds = colored_pickables_by_area(objects, 'red')
        return reds[0] if reds else None
    if normalized in {'blue', 'blue_cube'}:
        blue = colored_pickables_by_area(objects, 'blue')
        return blue[0] if blue else None
    return None


def _pick_and_place_on_stack(
    *,
    color_name: str,
    source: ObjectHypothesis,
    stack_xy: tuple[float, float],
    target_top_z: float,
    node: LiveSceneCensusNode,
    motion: RosMotionAdapter,
    timeout_sec: float,
    table_z: float,
    use_llm_motion_sketch: bool,
    sketch_model: str,
    sketch_timeout_sec: float,
) -> dict[str, Any]:
    grasp_planner = _stack_grasp_planner()
    motion_sketch = plan_pick_candidate(
        planner=MotionSketchPlanner(model=sketch_model, timeout_sec=sketch_timeout_sec),
        command=f'pick up the {color_name} cube for stacking',
        phase=f'stack_pick_{color_name}',
        source=source,
        avoid_objects=(),
        grasp_planner=grasp_planner,
        use_llm=use_llm_motion_sketch,
    )
    candidate = motion_sketch.candidate
    candidate = _prefer_yawed_top_down_for_stack(candidate, motion_sketch.compile_result.candidates)
    if candidate is None:
        return {
            'color_name': color_name,
            'success': False,
            'source': source,
            'motion_sketch': motion_sketch,
            'outcome': SkillOutcome(False, FailureCode.NO_VALID_GRASP, 'motion sketch compiled no stack grasp'),
        }
    staged = motion.move_to_pose(
        xyz=(float(source.xyz[0]), float(source.xyz[1]), STACK_SOURCE_STAGE_Z_M),
        quat_xyzw=candidate.quat_xyzw,
    )
    if not staged.success:
        return {
            'color_name': color_name,
            'success': False,
            'source': source,
            'candidate': candidate,
            'motion_sketch': motion_sketch,
            'outcome': SkillOutcome(
                False,
                FailureCode.MOTION_FAILED,
                f'source staging failed: {staged.message}',
            ),
        }
    pick_result = _pick_with_stack_centering(
        grasp_planner=grasp_planner,
        motion=motion,
        source=source,
        candidate=candidate,
        approach_preference=motion_sketch.compile_result.preferred_family,
    )
    pick = pick_result['outcome']
    candidate = pick_result['candidate']
    contact = motion.gripper_contact_detected(commanded_width=STACK_CLOSE_WIDTH_M, margin_m=STACK_CONTACT_MARGIN_M)
    if not (pick.success and contact):
        safe_retreat = _safe_source_retreat(motion=motion, source=source, candidate=candidate)
        return {
            'color_name': color_name,
            'success': False,
            'source': source,
            'candidate': candidate,
            'motion_sketch': motion_sketch,
            'pick_outcome': pick,
            'pick_attempts': pick_result['attempts'],
            'safe_retreat_after_pick_failure': safe_retreat,
            'outcome': SkillOutcome(
                False,
                FailureCode.CARRY_VERIFY_FAILED,
                'stack pick failed or carry contact missing; skipped broad verification sweep',
                data={'pick_success': pick.success, 'gripper_contact': contact},
            ),
        }

    place = _place_on_stack(
        motion=motion,
        stack_xy=stack_xy,
        target_top_z=target_top_z,
    )
    return {
        'color_name': color_name,
        'success': bool(place.success and motion.gripper_is_open()),
        'source': source,
        'candidate': candidate,
        'motion_sketch': motion_sketch,
        'pick_outcome': pick,
        'pick_attempts': pick_result['attempts'],
        'place_outcome': place,
        'checks': {
            'gripper_contact_after_pick': contact,
            'gripper_open_after_place': motion.gripper_is_open(),
        },
    }


def _pick_with_stack_centering(
    *,
    grasp_planner: GraspPlanner,
    motion: RosMotionAdapter,
    source: ObjectHypothesis,
    candidate: GraspCandidate,
    approach_preference: ApproachFamily,
) -> dict[str, Any]:
    attempts = []
    for dx, dy in STACK_GRASP_NUDGE_OFFSETS_M:
        trial = _candidate_with_xy_offset(candidate, dx=dx, dy=dy)
        outcome = PickSkill(grasp_planner=grasp_planner, motion=motion).run(
            source=source,
            avoid_objects=(),
            approach_preference=approach_preference,
            candidate=trial,
            close_position_m=STACK_CLOSE_WIDTH_M,
            contact_margin_m=STACK_CONTACT_MARGIN_M,
        )
        contact = motion.gripper_contact_detected(commanded_width=STACK_CLOSE_WIDTH_M, margin_m=STACK_CONTACT_MARGIN_M)
        attempts.append({'dx': dx, 'dy': dy, 'outcome': outcome, 'contact': contact})
        if outcome.success and contact:
            return {'outcome': outcome, 'candidate': trial, 'attempts': attempts}
    return {'outcome': attempts[-1]['outcome'], 'candidate': trial, 'attempts': attempts}


def _candidate_with_xy_offset(candidate: GraspCandidate, *, dx: float, dy: float) -> GraspCandidate:
    def shifted(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        return (float(xyz[0] + dx), float(xyz[1] + dy), float(xyz[2]))

    return GraspCandidate(
        family=candidate.family,
        pregrasp_xyz=shifted(candidate.pregrasp_xyz),
        grasp_xyz=shifted(candidate.grasp_xyz),
        retreat_xyz=shifted(candidate.retreat_xyz),
        quat_xyzw=candidate.quat_xyzw,
        clearance_score=candidate.clearance_score,
        ik_feasible=candidate.ik_feasible,
        collision_free=candidate.collision_free,
        keepout_clearance_m=candidate.keepout_clearance_m,
        reason=f'{candidate.reason}; stack center nudge dx={dx:.3f} dy={dy:.3f}',
    )


def _safe_source_retreat(
    *,
    motion: RosMotionAdapter,
    source: ObjectHypothesis,
    candidate,
) -> SkillOutcome:
    retreat_xyz = (
        float(source.xyz[0]),
        float(source.xyz[1]),
        max(float(candidate.retreat_xyz[2]), STACK_SOURCE_STAGE_Z_M),
    )
    result = motion.move_to_pose(xyz=retreat_xyz, quat_xyzw=candidate.quat_xyzw)
    if not result.success:
        return SkillOutcome(False, FailureCode.MOTION_FAILED, f'safe source retreat failed: {result.message}')
    return SkillOutcome(True, FailureCode.NONE, 'safe source retreat complete', data={'retreat_xyz': retreat_xyz})


def _prefer_yawed_top_down_for_stack(
    candidate: GraspCandidate | None,
    candidates: list[GraspCandidate],
) -> GraspCandidate | None:
    if candidate is None or candidate.family != ApproachFamily.TOP_DOWN:
        return candidate
    for option in candidates:
        if (
            option.family == ApproachFamily.TOP_DOWN_YAW_45
            and option.ik_feasible
            and option.collision_free
            and option.keepout_clearance_m >= 0.0
        ):
            return option
    return candidate


def _place_on_stack(
    *,
    motion: RosMotionAdapter,
    stack_xy: tuple[float, float],
    target_top_z: float,
) -> SkillOutcome:
    release_xyz = (
        float(stack_xy[0]),
        float(stack_xy[1]),
        float(target_top_z + STACK_RELEASE_CLEARANCE_M),
    )
    hover_z = max(float(release_xyz[2] + STACK_HOVER_CLEARANCE_M), STACK_MIN_HOVER_Z_M)
    hover_xyz = (release_xyz[0], release_xyz[1], hover_z)
    descend_xyz = (release_xyz[0], release_xyz[1], float((hover_xyz[2] + release_xyz[2]) * 0.5))
    for label, xyz in (('hover', hover_xyz), ('descend', descend_xyz), ('release', release_xyz)):
        result = motion.move_to_pose(xyz=xyz, quat_xyzw=motion.default_place_quat_xyzw())
        if not result.success:
            return SkillOutcome(False, FailureCode.MOTION_FAILED, f'{label} motion failed: {result.message}')
    actual_pose = motion.current_tcp_pose()
    if actual_pose is None:
        return SkillOutcome(False, FailureCode.RELEASE_GATE_FAILED, 'no TCP pose for stack release')
    actual_xyz = actual_pose[0]
    xy_err = ((actual_xyz[0] - release_xyz[0]) ** 2 + (actual_xyz[1] - release_xyz[1]) ** 2) ** 0.5
    min_tcp_z = float(target_top_z + STACK_TCP_BOTTOM_CLEARANCE_M)
    if xy_err > 0.025 or float(actual_xyz[2]) < min_tcp_z:
        return SkillOutcome(
            False,
            FailureCode.RELEASE_GATE_FAILED,
            'stack release gate failed',
            data={'actual_xyz': actual_xyz, 'release_xyz': release_xyz, 'xy_err': xy_err, 'min_tcp_z': min_tcp_z},
        )
    opened = motion.open_gripper()
    if not opened.success:
        return SkillOutcome(False, FailureCode.MOTION_FAILED, opened.message)
    retreat = motion.move_to_pose(xyz=hover_xyz, quat_xyzw=motion.default_place_quat_xyzw())
    if not retreat.success:
        return SkillOutcome(False, FailureCode.MOTION_FAILED, f'retreat failed: {retreat.message}')
    return SkillOutcome(
        True,
        FailureCode.NONE,
        'stack place complete',
        data={'actual_xyz': actual_xyz, 'release_xyz': release_xyz, 'xy_err': xy_err, 'target_top_z': target_top_z},
    )


def _stack_grasp_planner() -> GraspPlanner:
    return GraspPlanner(
        pregrasp_lift_m=0.11,
        retreat_lift_m=0.22,
        top_down_x_bias_m=0.0,
        top_down_surface_penetration_m=0.015,
    )


def _capture_from_pose(
    *,
    node: LiveSceneCensusNode,
    motion: RosMotionAdapter,
    xyz: tuple[float, float, float],
    timeout_sec: float,
) -> list[ObjectHypothesis]:
    moved = motion.move_to_pose(xyz=xyz, quat_xyzw=motion.default_place_quat_xyzw())
    if not moved.success:
        return []
    return node.capture(timeout_sec=timeout_sec)


def _finish(success: bool, **payload: Any) -> int:
    payload['success'] = bool(success)
    print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    return 0 if success else 2


if __name__ == '__main__':
    raise SystemExit(main())
