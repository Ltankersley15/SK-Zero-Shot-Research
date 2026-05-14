from __future__ import annotations

from dataclasses import replace

from fr3_zero_shot.core.types import ObjectHypothesis
from fr3_zero_shot.live_pick_place_utils import colored_pickables_by_area


def select_green_from_captures(captures: list[tuple[str, list[ObjectHypothesis]]]) -> ObjectHypothesis | None:
    by_view = {
        name: [obj for obj in objects if obj.color == 'green' and obj.pickable and not obj.container_like]
        for name, objects in captures
    }
    preferred: list[ObjectHypothesis] = []
    reobserve_preferred = any(by_view.get(name) for name in ('green_reobserve', 'green_wide_reobserve'))
    view_order = (
        ('green_reobserve', 'green_wide_reobserve', 'cup_top')
        if reobserve_preferred
        else ('initial_cup_side', 'cup_top')
    )
    for name in view_order:
        preferred.extend(sorted(by_view.get(name, ()), key=lambda obj: int(obj.metadata.get('area_px', 0) or 0), reverse=True)[:1])
    if len(preferred) >= 2:
        return _merge_green_observations(preferred)
    if preferred:
        return preferred[0]
    green = colored_pickables_by_area([obj for _, objects in captures for obj in objects], 'green')
    return green[0] if green else None


def _merge_green_observations(objects: list[ObjectHypothesis]) -> ObjectHypothesis:
    base = objects[0]
    count = float(len(objects))
    xyz = (
        float(sum(obj.xyz[0] for obj in objects) / count),
        float(sum(obj.xyz[1] for obj in objects) / count),
        float(base.xyz[2]),
    )
    metadata = dict(base.metadata)
    metadata['merged_green_views'] = [
        {
            'xyz': obj.xyz,
            'area_px': obj.metadata.get('area_px'),
            'center_uv': obj.metadata.get('center_uv'),
        }
        for obj in objects
    ]
    return replace(base, xyz=xyz, metadata=metadata)
