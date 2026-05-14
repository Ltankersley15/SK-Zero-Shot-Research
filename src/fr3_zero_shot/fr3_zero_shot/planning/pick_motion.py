from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fr3_zero_shot.core.types import GraspCandidate, MotionSketch, ObjectHypothesis
from fr3_zero_shot.grasping.grasp_planner import GraspPlanner
from fr3_zero_shot.planning.motion_sketch_compiler import MotionSketchCompileResult, MotionSketchCompiler
from fr3_zero_shot.planning.motion_sketch_planner import MotionSketchPlanner


@dataclass(frozen=True)
class PickMotionPlan:
    sketch: MotionSketch
    compile_result: MotionSketchCompileResult
    candidate: GraspCandidate | None
    prompt: str
    raw_response: str
    rejection_reason: str | None


def plan_pick_candidate(
    *,
    planner: MotionSketchPlanner,
    command: str,
    phase: str,
    source: ObjectHypothesis,
    avoid_objects: tuple[ObjectHypothesis, ...],
    grasp_planner: GraspPlanner,
    use_llm: bool,
    side_retention_risk: bool = True,
) -> PickMotionPlan:
    sketch = planner.plan(
        command=command,
        phase=phase,
        scene_facts=pick_scene_facts(source=source, avoid_objects=avoid_objects, phase=phase),
        use_llm=use_llm,
    )
    compiled = MotionSketchCompiler(
        grasp_planner=grasp_planner,
        side_retention_risk=side_retention_risk,
    ).compile(sketch=sketch, source=source, avoid_objects=avoid_objects)
    return PickMotionPlan(
        sketch=sketch,
        compile_result=compiled,
        candidate=compiled.candidate,
        prompt=planner.last_prompt,
        raw_response=planner.last_raw_response,
        rejection_reason=planner.last_rejection_reason,
    )


def pick_scene_facts(
    *,
    source: ObjectHypothesis,
    avoid_objects: tuple[ObjectHypothesis, ...],
    phase: str | None = None,
) -> dict[str, Any]:
    relative = [_relative_summary(source, obj) for obj in avoid_objects]
    near_keepout = any(item["near_keepout"] for item in relative)
    return {
        "source": _object_summary(source),
        "avoid_objects": [_object_summary(obj) for obj in avoid_objects],
        "relative_keepouts": relative,
        "phase": str(phase or ""),
        "motion_sketch_headroom": {
            "llm_may_choose": [
                "whether to reobserve",
                "which object-relative edge or side to use",
                "whether to pull away from the keepout before lifting",
                "whether bounded recovery is authorized",
            ],
            "llm_must_not_choose": [
                "metric coordinates",
                "quaternions",
                "joint angles",
                "gripper widths",
                "trajectories",
            ],
            "near_keepout": near_keepout,
            "top_down_center_risk": near_keepout,
            "recommended_relative_sketches_when_near_keepout": [
                "outer_edge plus yaw_45_away_from_keepout",
                "top_edge plus pull_away_before_lift",
                "exposed_side only when deterministic clearance can pass",
            ],
        },
        "available_reobserve_views": ["current_source_view", "side_view", "wide_view", "top_view"],
        "robot_authority_boundary": (
            "LLM writes object-relative motion sketch only; deterministic code owns exact poses, "
            "quaternions, IK, collision keepouts, gripper commands, and final verification."
        ),
    }


def _object_summary(obj: ObjectHypothesis) -> dict[str, Any]:
    return {
        "id": obj.object_id,
        "label": obj.label,
        "color": obj.color,
        "shape": obj.shape,
        "visible": obj.visible,
        "pickable": obj.pickable,
        "container_like": obj.container_like,
        "footprint_xy": obj.footprint_xy,
        "height_m": obj.height_m,
    }


def _relative_summary(source: ObjectHypothesis, avoid: ObjectHypothesis) -> dict[str, Any]:
    dx = float(source.xyz[0] - avoid.xyz[0])
    dy = float(source.xyz[1] - avoid.xyz[1])
    return {
        "avoid_id": avoid.object_id,
        "avoid_label": avoid.label,
        "source_direction_from_avoid": {
            "x_sign": "positive" if dx >= 0.0 else "negative",
            "y_sign": "positive" if dy >= 0.0 else "negative",
            "dominant_axis": "x" if abs(dx) >= abs(dy) else "y",
        },
        "near_keepout": ((dx * dx + dy * dy) ** 0.5) < 0.18,
    }
