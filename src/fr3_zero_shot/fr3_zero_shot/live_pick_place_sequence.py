from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from typing import Any

import rclpy

from fr3_zero_shot.core.types import FailureCode, ObjectHypothesis, SkillOutcome
from fr3_zero_shot.execution.ros_motion_adapter import RosMotionAdapter
from fr3_zero_shot.grasping.grasp_planner import GraspPlanner
from fr3_zero_shot.live_pick_place import CUP_OBLIQUE_VIEW, CUP_TOP_VIEW
from fr3_zero_shot.live_pick_place_utils import (
    best_colored_pickable,
    color_visible_outside_anchor,
    normalize_color_sequence,
    release_drop_for_sequence_step,
    select_cup,
    source_still_on_table,
)
from fr3_zero_shot.perception.live_scene_census import LiveSceneCensusConfig, LiveSceneCensusNode
from fr3_zero_shot.planning.motion_sketch_planner import MotionSketchPlanner
from fr3_zero_shot.planning.pick_motion import plan_pick_candidate
from fr3_zero_shot.skills.pick import PickSkill
from fr3_zero_shot.skills.place_in_container import PlaceInContainerSkill


SOURCE_VIEW = (0.48, 0.22, 0.38)
YELLOW_SOURCE_VIEW = (0.52, -0.16, 0.38)
YELLOW_POST_PICK_VIEW = (0.64, -0.34, 0.42)


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run a fixed multi-cube cup sequence in Isaac.')
    parser.add_argument(
        '--colors',
        nargs='+',
        default=['red', 'blue'],
        help='Ordered cube colors to place in the cup, e.g. --colors red blue yellow.',
    )
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--pick-only', action='store_true')
    parser.add_argument('--timeout-sec', type=float, default=8.0)
    parser.add_argument('--table-z', type=float, default=0.02)
    parser.add_argument('--position-tolerance', type=float, default=0.025)
    parser.add_argument('--orientation-tolerance', type=float, default=0.45)
    parser.add_argument('--llm-motion-sketch', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sketch-model', default='gpt-5-mini')
    parser.add_argument('--sketch-timeout-sec', type=float, default=20.0)
    args = parser.parse_args(argv)

    try:
        colors = normalize_color_sequence(list(args.colors))
    except ValueError as exc:
        return _finish(False, outcome=SkillOutcome(False, FailureCode.UNSUPPORTED_COMMAND, str(exc)))

    rclpy.init()
    node = LiveSceneCensusNode(LiveSceneCensusConfig(table_z_m=float(args.table_z)))
    try:
        motion = RosMotionAdapter(
            node,
            position_tolerance_m=float(args.position_tolerance),
            orientation_tolerance_rad=float(args.orientation_tolerance),
        )
        cup, cup_payload = _observe_cup(node=node, motion=motion, timeout_sec=float(args.timeout_sec))
        if cup is None:
            return _finish(False, colors=colors, **cup_payload)

        steps = []
        for step_index, color in enumerate(colors):
            step = _run_color_step(
                color=color,
                step_index=step_index,
                cup=cup,
                node=node,
                motion=motion,
                execute=bool(args.execute),
                pick_only=bool(args.pick_only),
                timeout_sec=float(args.timeout_sec),
                table_z=float(args.table_z),
                use_llm_motion_sketch=bool(args.llm_motion_sketch),
                sketch_model=str(args.sketch_model),
                sketch_timeout_sec=float(args.sketch_timeout_sec),
            )
            steps.append(step)
            if not step['success'] or not args.execute:
                return _finish(bool(step['success']), colors=colors, cup=cup, steps=steps, execute=bool(args.execute))

        final_objects = _capture_from_pose(
            node=node,
            motion=motion,
            xyz=CUP_TOP_VIEW,
            timeout_sec=float(args.timeout_sec),
        )
        final_checks = {
            color: not color_visible_outside_anchor(
                final_objects,
                color=color,
                anchor=cup,
                table_z=float(args.table_z),
            )
            for color in colors
        }
        success = all(step['success'] for step in steps) and all(final_checks.values())
        return _finish(
            success,
            colors=colors,
            cup=cup,
            steps=steps,
            final_objects=final_objects,
            final_checks=final_checks,
            execute=bool(args.execute),
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _observe_cup(
    *,
    node: LiveSceneCensusNode,
    motion: RosMotionAdapter,
    timeout_sec: float,
) -> tuple[ObjectHypothesis | None, dict[str, Any]]:
    cup_views = []
    for view_name, xyz in (
        ('cup_oblique', CUP_OBLIQUE_VIEW),
        ('cup_top', CUP_TOP_VIEW),
    ):
        moved = motion.move_to_pose(xyz=xyz, quat_xyzw=motion.default_place_quat_xyzw())
        if not moved.success:
            return None, {
                'outcome': SkillOutcome(
                    False,
                    FailureCode.MOTION_FAILED,
                    f'{view_name} motion failed: {moved.message}',
                )
            }
        cup_views.append((view_name, node.capture(timeout_sec=timeout_sec)))
    cup, reason = select_cup(cup_views)
    if cup is None:
        return None, {
            'cup_views': cup_views,
            'outcome': SkillOutcome(False, FailureCode.TARGET_NOT_VISIBLE, reason),
        }
    return cup, {'cup_reason': reason, 'cup_views': cup_views}


def _run_color_step(
    *,
    color: str,
    step_index: int,
    cup: ObjectHypothesis,
    node: LiveSceneCensusNode,
    motion: RosMotionAdapter,
    execute: bool,
    pick_only: bool,
    timeout_sec: float,
    table_z: float,
    use_llm_motion_sketch: bool,
    sketch_model: str,
    sketch_timeout_sec: float,
) -> dict[str, Any]:
    source_objects = _capture_from_pose(
        node=node,
        motion=motion,
        xyz=_source_view_for_color(color),
        timeout_sec=timeout_sec,
    )
    source = best_colored_pickable(source_objects, color)
    if source is None:
        return {
            'color': color,
            'success': False,
            'objects': source_objects,
            'outcome': SkillOutcome(
                False,
                FailureCode.SOURCE_NOT_VISIBLE,
                f'{color} cube not visible from source view',
            ),
        }

    grasp_planner = _sequence_grasp_planner(color)
    motion_sketch = plan_pick_candidate(
        planner=MotionSketchPlanner(model=sketch_model, timeout_sec=sketch_timeout_sec),
        command=f'pick up the {color} cube and place it in the cup',
        phase=f'sequence_pick_{color}_{step_index}',
        source=source,
        avoid_objects=(cup,),
        grasp_planner=grasp_planner,
        use_llm=use_llm_motion_sketch,
    )
    candidate = motion_sketch.candidate
    if not execute:
        return {
            'color': color,
            'success': candidate is not None,
            'source': source,
            'candidate': candidate,
            'motion_sketch': motion_sketch,
        }
    if candidate is None:
        return {
            'color': color,
            'success': False,
            'source': source,
            'motion_sketch': motion_sketch,
            'outcome': SkillOutcome(False, FailureCode.NO_VALID_GRASP, 'motion sketch compiled no sequence grasp'),
        }

    pick = PickSkill(grasp_planner=grasp_planner, motion=motion).run(
        source=source,
        avoid_objects=(cup,),
        approach_preference=motion_sketch.compile_result.preferred_family,
        candidate=candidate,
    )
    post_pick_objects = _capture_from_pose(
        node=node,
        motion=motion,
        xyz=_post_pick_view_for_color(color),
        timeout_sec=timeout_sec,
    )
    source_absent = not source_still_on_table(source, post_pick_objects, table_z=table_z)
    contact = motion.gripper_contact_detected()
    if not (pick.success and source_absent and contact):
        return {
            'color': color,
            'success': False,
            'source': source,
            'candidate': candidate,
            'motion_sketch': motion_sketch,
            'pick_outcome': pick,
            'post_pick_objects': post_pick_objects,
            'outcome': SkillOutcome(
                False,
                FailureCode.CARRY_VERIFY_FAILED,
                'carry verification failed',
                data={'source_absent': source_absent, 'gripper_contact': contact},
            ),
        }
    if pick_only:
        return {
            'color': color,
            'success': True,
            'source': source,
            'candidate': candidate,
            'motion_sketch': motion_sketch,
            'pick_outcome': pick,
            'post_pick_objects': post_pick_objects,
            'checks': {
                'source_absent_after_pick': source_absent,
                'gripper_contact_after_pick': contact,
            },
        }

    release_drop_m = release_drop_for_sequence_step(step_index)
    place = PlaceInContainerSkill(
        motion=motion,
        release_drop_m=release_drop_m,
        release_z_slack_m=0.015,
    ).run(held_object=source, container=cup)
    final_objects = []
    final_objects.extend(
        _capture_from_pose(
            node=node,
            motion=motion,
            xyz=CUP_TOP_VIEW,
            timeout_sec=timeout_sec,
        )
    )
    final_objects.extend(
        _capture_from_pose(
            node=node,
            motion=motion,
            xyz=CUP_OBLIQUE_VIEW,
            timeout_sec=timeout_sec,
        )
    )
    gripper_open = motion.gripper_is_open()
    color_outside = color_visible_outside_anchor(
        final_objects,
        color=color,
        anchor=cup,
        table_z=table_z,
    )
    return {
        'color': color,
        'success': bool(place.success and gripper_open and not color_outside),
        'source': source,
        'candidate': candidate,
        'motion_sketch': motion_sketch,
        'release_drop_m': release_drop_m,
        'pick_outcome': pick,
        'place_outcome': place,
        'post_pick_objects': post_pick_objects,
        'final_objects': final_objects,
        'checks': {
            'source_absent_after_pick': source_absent,
            'gripper_contact_after_pick': contact,
            'gripper_open_after_place': gripper_open,
            f'{color}_visible_outside_cup': color_outside,
        },
    }


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


def _sequence_grasp_planner(color: str = '') -> GraspPlanner:
    if color == 'yellow':
        return GraspPlanner(
            pregrasp_lift_m=0.13,
            retreat_lift_m=0.24,
            top_down_x_bias_m=0.0,
            top_down_surface_penetration_m=0.030,
        )
    return GraspPlanner(
        top_down_x_bias_m=0.0,
        top_down_surface_penetration_m=0.015,
    )


def _source_view_for_color(color: str) -> tuple[float, float, float]:
    if color == 'yellow':
        return YELLOW_SOURCE_VIEW
    return SOURCE_VIEW


def _post_pick_view_for_color(color: str) -> tuple[float, float, float]:
    if color == 'yellow':
        return YELLOW_POST_PICK_VIEW
    return SOURCE_VIEW


def _finish(success: bool, **payload: Any) -> int:
    payload['success'] = bool(success)
    print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    return 0 if success else 2


if __name__ == '__main__':
    raise SystemExit(main())
