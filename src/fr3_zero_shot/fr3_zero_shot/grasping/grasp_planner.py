from __future__ import annotations

from collections.abc import Callable
import math

from fr3_zero_shot.core.geometry import footprint_radius, segment_point_clearance_xy
from fr3_zero_shot.core.transforms import grasp_quaternion
from fr3_zero_shot.core.types import ApproachFamily, GraspCandidate, ObjectHypothesis

from .grasp_candidate import is_valid
from .grasp_family import ALL_FAMILIES, approach_vector_xy
from .grasp_ranker import rank_candidates


IkChecker = Callable[[GraspCandidate], bool]
CollisionChecker = Callable[[GraspCandidate], bool]


class GraspPlanner:
    def __init__(
        self,
        *,
        pregrasp_lift_m: float = 0.09,
        side_pregrasp_offset_m: float = 0.10,
        retreat_lift_m: float = 0.10,
        top_down_x_bias_m: float = 0.015,
        top_down_surface_penetration_m: float = 0.015,
        top_down_min_support_clearance_m: float = 0.004,
        min_keepout_clearance_m: float = 0.010,
        ik_checker: IkChecker | None = None,
        collision_checker: CollisionChecker | None = None,
    ) -> None:
        self._pregrasp_lift_m = max(0.01, float(pregrasp_lift_m))
        self._side_pregrasp_offset_m = max(0.02, float(side_pregrasp_offset_m))
        self._retreat_lift_m = max(0.02, float(retreat_lift_m))
        self._top_down_x_bias_m = float(top_down_x_bias_m)
        self._top_down_surface_penetration_m = max(
            0.0,
            float(top_down_surface_penetration_m),
        )
        self._top_down_min_support_clearance_m = max(
            0.0,
            float(top_down_min_support_clearance_m),
        )
        self._min_keepout_clearance_m = max(0.0, float(min_keepout_clearance_m))
        self._ik_checker = ik_checker
        self._collision_checker = collision_checker

    def plan(
        self,
        *,
        source: ObjectHypothesis,
        avoid_objects: tuple[ObjectHypothesis, ...],
        approach_preference: ApproachFamily,
    ) -> GraspCandidate | None:
        candidates = self.generate_candidates(source=source, avoid_objects=avoid_objects)
        valid = [cand for cand in candidates if is_valid(cand, min_keepout_clearance_m=self._min_keepout_clearance_m)]
        if not valid:
            return None
        return rank_candidates(valid, approach_preference=approach_preference)[0]

    def generate_candidates(
        self,
        *,
        source: ObjectHypothesis,
        avoid_objects: tuple[ObjectHypothesis, ...] = (),
    ) -> list[GraspCandidate]:
        candidates: list[GraspCandidate] = []
        for family in ALL_FAMILIES:
            cand = self._make_candidate(source, avoid_objects, family)
            ik_feasible = self._ik_checker(cand) if self._ik_checker else True
            collision_free = self._collision_checker(cand) if self._collision_checker else True
            candidates.append(
                GraspCandidate(
                    family=cand.family,
                    pregrasp_xyz=cand.pregrasp_xyz,
                    grasp_xyz=cand.grasp_xyz,
                    retreat_xyz=cand.retreat_xyz,
                    quat_xyzw=cand.quat_xyzw,
                    clearance_score=cand.clearance_score,
                    ik_feasible=bool(ik_feasible),
                    collision_free=bool(collision_free),
                    keepout_clearance_m=cand.keepout_clearance_m,
                    reason=cand.reason,
                )
            )
        return candidates

    def _make_candidate(
        self,
        source: ObjectHypothesis,
        avoid_objects: tuple[ObjectHypothesis, ...],
        family: ApproachFamily,
    ) -> GraspCandidate:
        sx, sy, sz = source.xyz
        support_z = float(sz - max(0.0, source.height_m) * 0.5)
        surface_z = float(sz + max(0.0, source.height_m) * 0.5)
        if family in {ApproachFamily.TOP_DOWN, ApproachFamily.TOP_DOWN_YAW_45}:
            grasp_z = max(
                support_z + self._top_down_min_support_clearance_m,
                surface_z - self._top_down_surface_penetration_m,
            )
            grasp_x = float(sx + self._top_down_x_bias_m)
            pregrasp = (grasp_x, float(sy), float(grasp_z + self._pregrasp_lift_m))
            grasp = (grasp_x, float(sy), float(grasp_z))
            retreat = (grasp_x, float(sy), float(grasp_z + self._retreat_lift_m))
        else:
            grasp_z = float(sz + min(0.03, max(0.0, source.height_m * 0.45)))
            ax, ay = approach_vector_xy(family)
            pregrasp = (
                float(sx + ax * self._side_pregrasp_offset_m),
                float(sy + ay * self._side_pregrasp_offset_m),
                float(grasp_z),
            )
            grasp = (float(sx), float(sy), float(grasp_z))
            retreat = (
                float(sx + ax * self._side_pregrasp_offset_m),
                float(sy + ay * self._side_pregrasp_offset_m),
                float(grasp_z + self._retreat_lift_m * 0.5),
            )
        clearance = self._keepout_clearance(pregrasp, grasp, source, avoid_objects)
        return GraspCandidate(
            family=family,
            pregrasp_xyz=pregrasp,
            grasp_xyz=grasp,
            retreat_xyz=retreat,
            quat_xyzw=grasp_quaternion(family),
            clearance_score=float(1.0 / (1.0 + math.exp(-50.0 * clearance))),
            ik_feasible=True,
            collision_free=True,
            keepout_clearance_m=clearance,
            reason=f"{family.value} candidate with deterministic keepout clearance",
        )

    def _keepout_clearance(
        self,
        pregrasp: tuple[float, float, float],
        grasp: tuple[float, float, float],
        source: ObjectHypothesis,
        avoid_objects: tuple[ObjectHypothesis, ...],
    ) -> float:
        if not avoid_objects:
            return float("inf")
        tool_radius = max(0.012, footprint_radius(source.footprint_xy) * 0.5)
        values: list[float] = []
        for obj in avoid_objects:
            radius = footprint_radius(obj.footprint_xy) + tool_radius
            values.append(
                segment_point_clearance_xy(
                    pregrasp,
                    grasp,
                    (obj.xyz[0], obj.xyz[1]),
                    radius_m=radius,
                )
            )
        return float(min(values))
