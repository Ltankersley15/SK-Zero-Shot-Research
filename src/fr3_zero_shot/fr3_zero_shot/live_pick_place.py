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
from fr3_zero_shot.live_pick_place_utils import select_cup, xy_dist
from fr3_zero_shot.perception.live_scene_census import LiveSceneCensusConfig, LiveSceneCensusNode
from fr3_zero_shot.planning.motion_sketch_planner import MotionSketchPlanner
from fr3_zero_shot.planning.pick_motion import plan_pick_candidate
from fr3_zero_shot.skills.pick import PickSkill
from fr3_zero_shot.skills.place_in_container import PlaceInContainerSkill


CUP_OBLIQUE_VIEW = (0.48, 0.00, 0.38)
CUP_TOP_VIEW = (0.56, 0.00, 0.42)
BLUE_SOURCE_VIEW = (0.48, 0.22, 0.38)


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run live blue-cube-in-cup in Isaac.')
    parser.add_argument(
        'command',
        nargs='?',
        default='pick up the blue cube and place it in the cup',
    )
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--timeout-sec', type=float, default=8.0)
    parser.add_argument('--table-z', type=float, default=0.02)
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

        cup_views = []
        for view_name, xyz in (
            ('cup_oblique', CUP_OBLIQUE_VIEW),
            ('cup_top', CUP_TOP_VIEW),
        ):
            moved = motion.move_to_pose(xyz=xyz, quat_xyzw=motion.default_place_quat_xyzw())
            if not moved.success:
                return _finish(
                    False,
                    outcome=SkillOutcome(
                        False,
                        FailureCode.MOTION_FAILED,
                        f'{view_name} motion failed: {moved.message}',
                    ),
                    view=view_name,
                )
            objects = node.capture(timeout_sec=float(args.timeout_sec))
            cup_views.append((view_name, objects))

        cup, cup_reason = select_cup(cup_views)
        if cup is None:
            return _finish(
                False,
                cup_views=cup_views,
                outcome=SkillOutcome(False, FailureCode.TARGET_NOT_VISIBLE, cup_reason),
            )

        moved = motion.move_to_pose(xyz=BLUE_SOURCE_VIEW, quat_xyzw=motion.default_place_quat_xyzw())
        if not moved.success:
            return _finish(
                False,
                cup=cup,
                outcome=SkillOutcome(
                    False,
                    FailureCode.MOTION_FAILED,
                    f'blue source view motion failed: {moved.message}',
                ),
            )
        source_objects = node.capture(timeout_sec=float(args.timeout_sec))
        blue = _best_blue_cube(source_objects)
        if blue is None:
            return _finish(
                False,
                cup=cup,
                objects=source_objects,
                outcome=SkillOutcome(False, FailureCode.SOURCE_NOT_VISIBLE, 'blue cube not visible from source view'),
            )

        grasp_planner = _blue_cup_grasp_planner()
        motion_sketch = plan_pick_candidate(
            planner=MotionSketchPlanner(
                model=str(args.sketch_model),
                timeout_sec=float(args.sketch_timeout_sec),
            ),
            command=str(args.command),
            phase='blue_cube_pick_for_cup',
            source=blue,
            avoid_objects=(cup,),
            grasp_planner=grasp_planner,
            use_llm=bool(args.llm_motion_sketch),
        )
        candidate = motion_sketch.candidate
        if not args.execute:
            return _finish(
                candidate is not None,
                command=str(args.command),
                cup=cup,
                blue=blue,
                candidate=candidate,
                motion_sketch=motion_sketch,
                cup_reason=cup_reason,
                execute=False,
            )
        if candidate is None:
            return _finish(
                False,
                command=str(args.command),
                cup=cup,
                blue=blue,
                motion_sketch=motion_sketch,
                outcome=SkillOutcome(False, FailureCode.NO_VALID_GRASP, 'motion sketch compiled no valid blue grasp'),
                execute=True,
            )

        pick = PickSkill(grasp_planner=grasp_planner, motion=motion).run(
            source=blue,
            avoid_objects=(cup,),
            approach_preference=motion_sketch.compile_result.preferred_family,
            candidate=candidate,
        )
        moved = motion.move_to_pose(xyz=BLUE_SOURCE_VIEW, quat_xyzw=motion.default_place_quat_xyzw())
        post_pick_objects = node.capture(timeout_sec=float(args.timeout_sec)) if moved.success else []
        source_absent = not _source_still_on_table(blue, post_pick_objects, table_z=float(args.table_z))
        contact = motion.gripper_contact_detected()
        if not (pick.success and source_absent and contact):
            return _finish(
                False,
                command=str(args.command),
                cup=cup,
                blue=blue,
                candidate=candidate,
                motion_sketch=motion_sketch,
                pick_outcome=pick,
                post_pick_objects=post_pick_objects,
                outcome=SkillOutcome(
                    False,
                    FailureCode.CARRY_VERIFY_FAILED,
                    'carry verification failed',
                    data={'source_absent': source_absent, 'gripper_contact': contact},
                ),
                execute=True,
            )

        place = PlaceInContainerSkill(
            motion=motion,
            release_drop_m=0.060,
            release_z_slack_m=0.020,
        ).run(held_object=blue, container=cup)
        moved = motion.move_to_pose(xyz=CUP_TOP_VIEW, quat_xyzw=motion.default_place_quat_xyzw())
        final_top = node.capture(timeout_sec=float(args.timeout_sec)) if moved.success else []
        moved = motion.move_to_pose(xyz=CUP_OBLIQUE_VIEW, quat_xyzw=motion.default_place_quat_xyzw())
        final_oblique = node.capture(timeout_sec=float(args.timeout_sec)) if moved.success else []
        gripper_open = motion.gripper_is_open()
        blue_outside = _blue_visible_outside_cup(final_top + final_oblique, cup, table_z=float(args.table_z))
        success = bool(place.success and gripper_open and not blue_outside)
        return _finish(
            success,
            command=str(args.command),
            cup=cup,
            blue=blue,
            candidate=candidate,
            motion_sketch=motion_sketch,
            pick_outcome=pick,
            place_outcome=place,
            post_pick_objects=post_pick_objects,
            final_top_objects=final_top,
            final_oblique_objects=final_oblique,
            final_checks={
                'source_absent_after_pick': source_absent,
                'gripper_contact_after_pick': contact,
                'gripper_open_after_place': gripper_open,
                'blue_visible_outside_cup': blue_outside,
            },
            execute=True,
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _finish(success: bool, **payload: Any) -> int:
    payload['success'] = bool(success)
    print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    return 0 if success else 2


def _blue_cup_grasp_planner() -> GraspPlanner:
    return GraspPlanner(
        top_down_x_bias_m=0.0,
        top_down_surface_penetration_m=0.025,
    )


def _best_blue_cube(objects: list[ObjectHypothesis]) -> ObjectHypothesis | None:
    blue = [obj for obj in objects if obj.color == 'blue' and obj.pickable and not obj.container_like]
    if not blue:
        return None
    return sorted(blue, key=lambda obj: obj.confidence, reverse=True)[0]


def _source_still_on_table(source: ObjectHypothesis, objects: list[ObjectHypothesis], *, table_z: float) -> bool:
    for obj in objects:
        if obj.color != source.color:
            continue
        raw_xyz = obj.metadata.get('raw_depth_base_xyz')
        if not raw_xyz:
            continue
        if float(raw_xyz[2]) <= float(table_z) + 0.10:
            return True
    return False


def _blue_visible_outside_cup(
    objects: list[ObjectHypothesis],
    cup: ObjectHypothesis,
    *,
    table_z: float,
) -> bool:
    radius = max(float(cup.footprint_xy[0]), float(cup.footprint_xy[1])) * 0.5 + 0.035
    for obj in objects:
        if obj.color != 'blue':
            continue
        raw_xyz = obj.metadata.get('raw_depth_base_xyz')
        on_table = raw_xyz is None or float(raw_xyz[2]) <= float(table_z) + 0.10
        if on_table and xy_dist(obj.xyz, cup.xyz) > radius:
            return True
    return False


if __name__ == '__main__':
    raise SystemExit(main())
