from __future__ import annotations

from dataclasses import replace
from time import time

from fr3_zero_shot.core.geometry import xy_distance
from fr3_zero_shot.core.types import ObjectHypothesis


class ObjectTable:
    def __init__(self, *, association_xy_tolerance_m: float = 0.06) -> None:
        self._association_xy_tolerance_m = max(0.0, float(association_xy_tolerance_m))
        self._objects: dict[str, ObjectHypothesis] = {}
        self._last_seen: dict[str, float] = {}

    def upsert_many(self, objects: list[ObjectHypothesis]) -> tuple[ObjectHypothesis, ...]:
        now = time()
        updated: list[ObjectHypothesis] = []
        for obj in objects:
            object_id = self._match_existing_id(obj) or obj.object_id
            stored = replace(obj, object_id=object_id)
            self._objects[object_id] = stored
            self._last_seen[object_id] = now
            updated.append(stored)
        return tuple(updated)

    def get(self, object_id: str | None) -> ObjectHypothesis | None:
        if object_id is None:
            return None
        return self._objects.get(str(object_id))

    def objects(self, *, visible_only: bool = False) -> tuple[ObjectHypothesis, ...]:
        values = tuple(self._objects.values())
        if not visible_only:
            return values
        return tuple(obj for obj in values if obj.visible)

    def forget(self, object_id: str) -> None:
        self._objects.pop(str(object_id), None)
        self._last_seen.pop(str(object_id), None)

    def _match_existing_id(self, incoming: ObjectHypothesis) -> str | None:
        incoming_xy = (incoming.xyz[0], incoming.xyz[1])
        best: tuple[float, str] | None = None
        for object_id, existing in self._objects.items():
            if existing.color and incoming.color and existing.color != incoming.color:
                continue
            if existing.shape and incoming.shape and existing.shape != incoming.shape:
                continue
            dist = xy_distance((existing.xyz[0], existing.xyz[1]), incoming_xy)
            if dist > self._association_xy_tolerance_m:
                continue
            if best is None or dist < best[0]:
                best = (dist, object_id)
        return None if best is None else best[1]

