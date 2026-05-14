from __future__ import annotations

import json
from pathlib import Path

from fr3_zero_shot.core.types import ObjectHypothesis, SceneSummary
from fr3_zero_shot.memory.scene_memory import SceneMemory

from .object_detector import object_from_mapping


class SceneCensus:
    def __init__(self, memory: SceneMemory | None = None) -> None:
        self._memory = SceneMemory() if memory is None else memory

    def from_objects(self, objects: list[ObjectHypothesis]) -> SceneSummary:
        return self._memory.update(objects)

    def from_json_file(self, path: str | Path) -> SceneSummary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        items = payload.get("objects", payload if isinstance(payload, list) else [])
        objects = [object_from_mapping(item) for item in items]
        return self.from_objects(objects)

