from __future__ import annotations

from dataclasses import dataclass
import math

from fr3_zero_shot.core.transforms import top_down_quaternion
from fr3_zero_shot.core.types import ApproachFamily, GraspCandidate, MotionSketch, ObjectHypothesis
from fr3_zero_shot.grasping.grasp_candidate import is_valid
from fr3_zero_shot.grasping.grasp_family import SIDE_FAMILIES
from fr3_zero_shot.grasping.grasp_planner import GraspPlanner
from fr3_zero_shot.live_pick_place_utils import side_family_away_from_keepout


@dataclass(frozen=True)
class CandidateTrace:
    family: ApproachFamily
    score: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class MotionSketchCompileResult:
    candidate: GraspCandidate | None
    candidates: list[GraspCandidate]
    trace: list[CandidateTrace]
    preferred_family: ApproachFamily


class MotionSketchCompiler:
    def __init__(
        self,
        *,
        grasp_planner: GraspPlanner | None = None,
        min_keepout_clearance_m: float = 0.006,
        side_retention_risk: bool = True,
    ) -> None:
        self._grasp_planner = grasp_planner or GraspPlanner(min_keepout_clearance_m=min_keepout_clearance_m)
        self._min_keepout_clearance_m = float(min_keepout_clearance_m)
        self._side_retention_risk = bool(side_retention_risk)

    def compile(
        self,
        *,
        sketch: MotionSketch,
        source: ObjectHypothesis,
        avoid_objects: tuple[ObjectHypothesis, ...] = (),
    ) -> MotionSketchCompileResult:
        preferred = preferred_family_from_sketch(sketch, source=source, avoid_objects=avoid_objects)
        raw = self._grasp_planner.generate_candidates(source=source, avoid_objects=avoid_objects)
        candidates = [_apply_sketch_pose_adjustments(cand, sketch) for cand in raw]
        trace: list[CandidateTrace] = []
        scored: list[tuple[float, GraspCandidate]] = []
        for candidate in candidates:
            valid = is_valid(candidate, min_keepout_clearance_m=self._min_keepout_clearance_m)
            score, why = self._score(candidate, preferred, sketch, avoid_objects)
            trace.append(CandidateTrace(candidate.family, score, valid, why if valid else f"rejected: {why}"))
            if valid:
                scored.append((score, candidate))
        if not scored or sketch.grasp_intent.approach_direction == "fail_closed":
            return MotionSketchCompileResult(None, candidates, trace, preferred)
        scored.sort(key=lambda item: item[0], reverse=True)
        return MotionSketchCompileResult(scored[0][1], candidates, trace, preferred)

    def _score(
        self,
        candidate: GraspCandidate,
        preferred: ApproachFamily,
        sketch: MotionSketch,
        avoid_objects: tuple[ObjectHypothesis, ...],
    ) -> tuple[float, str]:
        score = float(candidate.clearance_score) + min(0.5, max(0.0, candidate.keepout_clearance_m))
        reasons: list[str] = ["clearance"]
        if candidate.family == preferred:
            score += 2.0
            reasons.append("matches_sketch")
        elif preferred.value.startswith("side") and candidate.family.value.startswith("side"):
            score += 0.8
            reasons.append("matches_side_family")
        if _wants_yawed_edge(sketch) and candidate.family == ApproachFamily.TOP_DOWN_YAW_45:
            score += 2.2
            reasons.append("yawed_edge_intent")
        if avoid_objects and candidate.family == ApproachFamily.TOP_DOWN:
            score -= 0.4
            reasons.append("plain_top_down_near_keepout")
        if self._side_retention_risk and candidate.family in SIDE_FAMILIES and not _requires_side(sketch):
            score -= 2.4
            reasons.append("side_retention_risk")
        if _requires_side(sketch) and candidate.family in SIDE_FAMILIES:
            score += 1.4
            reasons.append("explicit_side_required")
        if sketch.retreat_intent.lift_after_clearance and candidate.retreat_xyz[2] <= candidate.grasp_xyz[2]:
            score -= 1.0
            reasons.append("no_lift_after_contact")
        return score, ",".join(reasons)


def preferred_family_from_sketch(
    sketch: MotionSketch,
    *,
    source: ObjectHypothesis,
    avoid_objects: tuple[ObjectHypothesis, ...] = (),
) -> ApproachFamily:
    text = " ".join(
        [
            sketch.grasp_intent.approach_direction,
            sketch.grasp_intent.grasp_region,
            sketch.grasp_intent.wrist_intent,
            sketch.grasp_intent.contact_style,
            " ".join(sketch.constraints),
        ]
    ).lower()
    if "fail_closed" in text:
        return ApproachFamily.AUTO
    if "side_right" in text or "right" == sketch.grasp_intent.approach_direction:
        return ApproachFamily.SIDE_RIGHT
    if "side_left" in text or "left" == sketch.grasp_intent.approach_direction:
        return ApproachFamily.SIDE_LEFT
    if "side_front" in text or "front" == sketch.grasp_intent.approach_direction:
        return ApproachFamily.SIDE_FRONT
    if "side_back" in text or "back" == sketch.grasp_intent.approach_direction:
        return ApproachFamily.SIDE_BACK
    if ("away_from_keepout" in text or "exposed_side" in text) and avoid_objects:
        return side_family_away_from_keepout(source, avoid_objects[0])
    if _wants_yawed_edge(sketch):
        return ApproachFamily.TOP_DOWN_YAW_45
    if "top" in text:
        return ApproachFamily.TOP_DOWN_YAW_45
    return ApproachFamily.AUTO


def _apply_sketch_pose_adjustments(candidate: GraspCandidate, sketch: MotionSketch) -> GraspCandidate:
    if _wants_yawed_edge(sketch) and candidate.family == ApproachFamily.TOP_DOWN_YAW_45:
        return GraspCandidate(
            family=candidate.family,
            pregrasp_xyz=candidate.pregrasp_xyz,
            grasp_xyz=candidate.grasp_xyz,
            retreat_xyz=candidate.retreat_xyz,
            quat_xyzw=top_down_quaternion(math.radians(-45.0)),
            clearance_score=candidate.clearance_score,
            ik_feasible=candidate.ik_feasible,
            collision_free=candidate.collision_free,
            keepout_clearance_m=candidate.keepout_clearance_m,
            reason=f"{candidate.reason}; motion sketch requested yawed edge wrist",
        )
    return candidate


def _wants_yawed_edge(sketch: MotionSketch) -> bool:
    text = " ".join(
        [
            sketch.grasp_intent.approach_direction,
            sketch.grasp_intent.grasp_region,
            sketch.grasp_intent.wrist_intent,
            sketch.grasp_intent.contact_style,
        ]
    ).lower()
    return "edge" in text or "yaw" in text or "align_to_cube_edge" in text


def _requires_side(sketch: MotionSketch) -> bool:
    text = " ".join(
        [
            sketch.grasp_intent.approach_direction,
            sketch.grasp_intent.grasp_region,
            sketch.grasp_intent.contact_style,
            " ".join(sketch.constraints),
        ]
    ).lower()
    return "require_side_grasp" in text or "side_only" in text
