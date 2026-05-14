from __future__ import annotations

from fr3_zero_shot.core.types import ObjectHypothesis


def object_from_mapping(data: dict) -> ObjectHypothesis:
    return ObjectHypothesis(
        object_id=str(data["object_id"]),
        label=str(data.get("label", data["object_id"])),
        color=None if data.get("color") is None else str(data.get("color")),
        shape=None if data.get("shape") is None else str(data.get("shape")),
        xyz=tuple(float(v) for v in data["xyz"]),  # type: ignore[arg-type]
        footprint_xy=tuple(float(v) for v in data.get("footprint_xy", (0.04, 0.04))),  # type: ignore[arg-type]
        height_m=float(data.get("height_m", 0.04)),
        bbox_xyxy=tuple(int(v) for v in data.get("bbox_xyxy", (0, 0, 0, 0))),  # type: ignore[arg-type]
        confidence=float(data.get("confidence", 1.0)),
        visible=bool(data.get("visible", True)),
        pickable=bool(data.get("pickable", True)),
        container_like=bool(data.get("container_like", False)),
        metadata=dict(data.get("metadata", {})),
    )

