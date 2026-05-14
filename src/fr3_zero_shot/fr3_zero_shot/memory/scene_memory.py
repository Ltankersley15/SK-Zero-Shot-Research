from __future__ import annotations

from fr3_zero_shot.core.geometry import footprint_radius, xy_distance
from fr3_zero_shot.core.types import ObjectHypothesis, SceneRelation, SceneSummary

from .object_table import ObjectTable


class SceneMemory:
    def __init__(
        self,
        *,
        association_xy_tolerance_m: float = 0.06,
        near_tolerance_m: float = 0.11,
        top_down_block_radius_m: float = 0.075,
    ) -> None:
        self._table = ObjectTable(association_xy_tolerance_m=association_xy_tolerance_m)
        self._near_tolerance_m = max(0.0, float(near_tolerance_m))
        self._top_down_block_radius_m = max(0.0, float(top_down_block_radius_m))

    def update(self, objects: list[ObjectHypothesis]) -> SceneSummary:
        self._table.upsert_many(objects)
        return self.summary()

    def get(self, object_id: str | None) -> ObjectHypothesis | None:
        return self._table.get(object_id)

    def objects(self, *, visible_only: bool = False) -> tuple[ObjectHypothesis, ...]:
        return self._table.objects(visible_only=visible_only)

    def summary(self) -> SceneSummary:
        objects = self.objects()
        relations = tuple(self._build_relation(obj, objects) for obj in objects)
        return SceneSummary(objects=objects, relations=relations)

    def _build_relation(
        self,
        obj: ObjectHypothesis,
        objects: tuple[ObjectHypothesis, ...],
    ) -> SceneRelation:
        near_ids: list[str] = []
        top_down_clear = True
        side_clear = True
        obj_xy = (obj.xyz[0], obj.xyz[1])
        for other in objects:
            if other.object_id == obj.object_id:
                continue
            clearance = xy_distance(obj_xy, (other.xyz[0], other.xyz[1]))
            clearance -= footprint_radius(obj.footprint_xy) + footprint_radius(other.footprint_xy)
            if clearance <= self._near_tolerance_m:
                near_ids.append(other.object_id)
            if other.container_like and clearance <= self._top_down_block_radius_m:
                top_down_clear = False
            if other.container_like and clearance <= 0.0:
                side_clear = False
        return SceneRelation(
            object_id=obj.object_id,
            near_ids=tuple(near_ids),
            top_down_clear=top_down_clear,
            side_clear=side_clear,
        )

