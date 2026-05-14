from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from typing import Any

import rclpy

from fr3_zero_shot.core.types import FailureCode, SkillOutcome
from fr3_zero_shot.execution.ros_motion_adapter import RosMotionAdapter
from fr3_zero_shot.grasping.grasp_planner import GraspPlanner
from fr3_zero_shot.grounding.reference_grounder import ReferenceGrounder
from fr3_zero_shot.perception.live_scene_census import (
    LiveSceneCensusConfig,
    LiveSceneCensusNode,
)
from fr3_zero_shot.planning.motion_sketch_planner import MotionSketchPlanner
from fr3_zero_shot.planning.pick_motion import plan_pick_candidate
from fr3_zero_shot.planning.plan_validator import PlanValidator
from fr3_zero_shot.planning.symbolic_planner import SymbolicPlanner
from fr3_zero_shot.skills.pick import PickSkill


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run one live pick in Isaac.')
    parser.add_argument('command')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--timeout-sec', type=float, default=5.0)
    parser.add_argument('--table-z', type=float, default=0.02)
    parser.add_argument('--observe-first', action='store_true', default=True)
    parser.add_argument('--observation-x', type=float, default=0.48)
    parser.add_argument('--observation-y', type=float, default=0.0)
    parser.add_argument('--observation-z', type=float, default=0.38)
    parser.add_argument('--position-tolerance', type=float, default=0.025)
    parser.add_argument('--orientation-tolerance', type=float, default=0.45)
    parser.add_argument('--llm-motion-sketch', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sketch-model', default='gpt-5-mini')
    parser.add_argument('--sketch-timeout-sec', type=float, default=20.0)
    args = parser.parse_args(argv)

    rclpy.init()
    node = LiveSceneCensusNode(
        LiveSceneCensusConfig(table_z_m=float(args.table_z))
    )
    try:
        motion = RosMotionAdapter(
            node,
            position_tolerance_m=float(args.position_tolerance),
            orientation_tolerance_rad=float(args.orientation_tolerance),
        )
        if args.observe_first:
            observed = motion.move_to_pose(
                xyz=(
                    float(args.observation_x),
                    float(args.observation_y),
                    float(args.observation_z),
                ),
                quat_xyzw=motion.default_place_quat_xyzw(),
            )
            if not observed.success:
                return _finish(
                    False,
                    outcome=SkillOutcome(
                        False,
                        FailureCode.MOTION_FAILED,
                        f'observation motion failed: {observed.message}',
                    ),
                )
        scene_objects = node.capture(timeout_sec=float(args.timeout_sec))
        plan = SymbolicPlanner().plan(str(args.command))
        validation = PlanValidator().validate(plan)
        if not validation.success:
            return _finish(False, plan=plan, objects=scene_objects, outcome=validation)
        grounded, ground_outcome = ReferenceGrounder().ground(
            plan,
            tuple(scene_objects),
        )
        if grounded is None:
            return _finish(
                False,
                plan=plan,
                objects=scene_objects,
                outcome=ground_outcome,
            )
        source = next(
            (obj for obj in scene_objects if obj.object_id == grounded.source_id),
            None,
        )
        if source is None:
            return _finish(
                False,
                plan=plan,
                objects=scene_objects,
                outcome=SkillOutcome(
                    False,
                    FailureCode.SOURCE_NOT_VISIBLE,
                    'grounded source missing',
                ),
            )
        avoid = tuple(
            obj for obj in scene_objects if obj.object_id in set(grounded.avoid_ids)
        )
        grasp_planner = GraspPlanner()
        motion_sketch = plan_pick_candidate(
            planner=MotionSketchPlanner(
                model=str(args.sketch_model),
                timeout_sec=float(args.sketch_timeout_sec),
            ),
            command=str(args.command),
            phase='live_pick',
            source=source,
            avoid_objects=avoid,
            grasp_planner=grasp_planner,
            use_llm=bool(args.llm_motion_sketch),
        )
        candidate = motion_sketch.candidate
        if not args.execute:
            return _finish(
                candidate is not None,
                plan=plan,
                grounded=grounded,
                objects=scene_objects,
                candidate=candidate,
                motion_sketch=motion_sketch,
                execute=False,
            )
        if candidate is None:
            return _finish(
                False,
                plan=plan,
                grounded=grounded,
                objects=scene_objects,
                motion_sketch=motion_sketch,
                outcome=SkillOutcome(False, FailureCode.NO_VALID_GRASP, 'motion sketch compiled no valid grasp'),
                execute=True,
            )
        outcome = PickSkill(grasp_planner=grasp_planner, motion=motion).run(
            source=source,
            avoid_objects=avoid,
            approach_preference=motion_sketch.compile_result.preferred_family,
            candidate=candidate,
        )
        post_objects = []
        if outcome.success:
            motion.move_to_pose(
                xyz=(
                    float(args.observation_x),
                    float(args.observation_y),
                    float(args.observation_z),
                ),
                quat_xyzw=motion.default_place_quat_xyzw(),
            )
            post_objects = node.capture(timeout_sec=float(args.timeout_sec))
            source_absent = not _source_still_on_table(source, post_objects, table_z=float(args.table_z))
            contact = motion.gripper_contact_detected()
            if not source_absent:
                outcome = SkillOutcome(
                    False,
                    FailureCode.CARRY_VERIFY_FAILED,
                    'post-pick observation still sees source on tabletop',
                    data={'post_objects': post_objects, 'source_absent_after_pick': source_absent, 'gripper_contact_after_pick': contact},
                )
            elif not contact:
                outcome = SkillOutcome(
                    False,
                    FailureCode.CARRY_VERIFY_FAILED,
                    'source absent from tabletop but gripper contact is false',
                    data={'post_objects': post_objects, 'source_absent_after_pick': source_absent, 'gripper_contact_after_pick': contact},
                )
            else:
                outcome = SkillOutcome(
                    True,
                    FailureCode.NONE,
                    'pick verified by source absence and gripper contact',
                    data={'post_objects': post_objects, 'source_absent_after_pick': source_absent, 'gripper_contact_after_pick': contact},
                )
        return _finish(
            outcome.success,
            plan=plan,
            grounded=grounded,
            objects=scene_objects,
            post_objects=post_objects,
            candidate=candidate,
            motion_sketch=motion_sketch,
            outcome=outcome,
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


def _source_still_on_table(source, objects, *, table_z: float) -> bool:
    for obj in objects:
        if obj.color != source.color:
            continue
        raw_xyz = obj.metadata.get('raw_depth_base_xyz')
        if not raw_xyz:
            continue
        if float(raw_xyz[2]) <= float(table_z) + 0.10:
            return True
    return False


if __name__ == '__main__':
    raise SystemExit(main())
