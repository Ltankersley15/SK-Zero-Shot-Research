from __future__ import annotations

import json

from fr3_zero_shot.core.types import SceneSummary


def build_symbolic_prompt(command: str, scene: SceneSummary | None = None) -> str:
    payload = {"command": str(command), "objects": []}
    if scene is not None:
        relation_by_id = {rel.object_id: rel for rel in scene.relations}
        for obj in scene.objects:
            rel = relation_by_id.get(obj.object_id)
            payload["objects"].append(
                {
                    "id": obj.object_id,
                    "label": obj.label,
                    "color": obj.color,
                    "shape": obj.shape,
                    "pickable": obj.pickable,
                    "container_like": obj.container_like,
                    "near": [] if rel is None else list(rel.near_ids),
                    "top_down_clear": True if rel is None else rel.top_down_clear,
                    "side_clear": True if rel is None else rel.side_clear,
                }
            )
    return json.dumps(payload, sort_keys=True)

