from __future__ import annotations

import math

from fr3_zero_shot.core.types import ApproachFamily, ObjectHypothesis


SEQUENCE_RELEASE_HEIGHT_OFFSET_M = 0.050


def select_cup(
    cup_views: list[tuple[str, list[ObjectHypothesis]]],
    *,
    agreement_m: float = 0.06,
) -> tuple[ObjectHypothesis | None, str]:
    by_view = [
        (view, obj)
        for view, objects in cup_views
        for obj in objects
        if obj.container_like
    ]
    top = [obj for view, obj in by_view if view == 'cup_top']
    if top:
        return sorted(top, key=lambda obj: obj.confidence, reverse=True)[0], 'top-view cup detection'
    for i, (left_view, left) in enumerate(by_view):
        for right_view, right in by_view[i + 1:]:
            if left_view == right_view:
                continue
            if xy_dist(left.xyz, right.xyz) <= agreement_m:
                return merge_cups(left, right), f'two-view cup agreement {left_view}+{right_view}'
    return None, 'no top-view cup and no two-view cup agreement'


def merge_cups(left: ObjectHypothesis, right: ObjectHypothesis) -> ObjectHypothesis:
    xyz = (
        float((left.xyz[0] + right.xyz[0]) * 0.5),
        float((left.xyz[1] + right.xyz[1]) * 0.5),
        float(max(left.xyz[2], right.xyz[2])),
    )
    footprint = (
        float(max(left.footprint_xy[0], right.footprint_xy[0])),
        float(max(left.footprint_xy[1], right.footprint_xy[1])),
    )
    metadata = {
        **left.metadata,
        'merged_from': [left.metadata, right.metadata],
    }
    return ObjectHypothesis(
        object_id='obj_cup',
        label='cup',
        color=None,
        shape='cup',
        xyz=xyz,
        footprint_xy=footprint,
        height_m=float(max(left.height_m, right.height_m)),
        bbox_xyxy=left.bbox_xyxy,
        confidence=float(max(left.confidence, right.confidence)),
        visible=True,
        pickable=False,
        container_like=True,
        metadata=metadata,
    )


def xy_dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def best_colored_pickable(
    objects: list[ObjectHypothesis],
    color: str,
) -> ObjectHypothesis | None:
    matches = [
        obj
        for obj in objects
        if obj.color == color and obj.pickable and not obj.container_like
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda obj: obj.confidence, reverse=True)[0]


def pickable_area_px(obj: ObjectHypothesis) -> int:
    return int(obj.metadata.get('area_px', 0) or 0)


def colored_pickables_by_area(
    objects: list[ObjectHypothesis],
    color: str,
    *,
    merge_radius_m: float = 0.08,
) -> list[ObjectHypothesis]:
    matches = [
        obj
        for obj in objects
        if obj.color == color and obj.pickable and not obj.container_like
    ]
    clusters: list[ObjectHypothesis] = []
    for obj in sorted(matches, key=lambda item: pickable_area_px(item), reverse=True):
        duplicate = False
        for existing in clusters:
            if xy_dist(obj.xyz, existing.xyz) <= merge_radius_m:
                duplicate = True
                break
        if not duplicate:
            clusters.append(obj)
    return sorted(clusters, key=lambda item: pickable_area_px(item), reverse=True)


def side_family_away_from_keepout(
    source: ObjectHypothesis,
    keepout: ObjectHypothesis,
) -> ApproachFamily:
    dx = float(source.xyz[0]) - float(keepout.xyz[0])
    dy = float(source.xyz[1]) - float(keepout.xyz[1])
    if abs(dx) >= abs(dy):
        return ApproachFamily.SIDE_BACK if dx >= 0.0 else ApproachFamily.SIDE_FRONT
    return ApproachFamily.SIDE_RIGHT if dy >= 0.0 else ApproachFamily.SIDE_LEFT


def source_still_on_table(
    source: ObjectHypothesis,
    objects: list[ObjectHypothesis],
    *,
    table_z: float,
    match_radius_m: float = 0.08,
) -> bool:
    for obj in objects:
        if obj.color != source.color:
            continue
        if xy_dist(obj.xyz, source.xyz) > float(match_radius_m):
            continue
        raw_xyz = obj.metadata.get('raw_depth_base_xyz')
        if not raw_xyz:
            continue
        if float(raw_xyz[2]) <= float(table_z) + 0.10:
            return True
    return False


def color_visible_outside_anchor(
    objects: list[ObjectHypothesis],
    *,
    color: str,
    anchor: ObjectHypothesis,
    table_z: float,
    radius_margin_m: float = 0.035,
) -> bool:
    radius = (
        max(float(anchor.footprint_xy[0]), float(anchor.footprint_xy[1])) * 0.5
        + float(radius_margin_m)
    )
    for obj in objects:
        if obj.color != color:
            continue
        raw_xyz = obj.metadata.get('raw_depth_base_xyz')
        on_table = raw_xyz is None or float(raw_xyz[2]) <= float(table_z) + 0.10
        if on_table and xy_dist(obj.xyz, anchor.xyz) > radius:
            return True
    return False


def normalize_color_sequence(colors: list[str]) -> list[str]:
    allowed = {'red', 'blue', 'yellow', 'green'}
    normalized: list[str] = []
    for color in colors:
        value = str(color).strip().lower()
        if not value:
            continue
        if value not in allowed:
            raise ValueError(f"unsupported color '{color}'")
        normalized.append(value)
    if not normalized:
        raise ValueError('empty color sequence')
    return normalized


def release_drop_for_sequence_step(step_index: int) -> float:
    base_release_drop_m = min(0.050, 0.045 + 0.005 * max(0, int(step_index)))
    return float(base_release_drop_m + SEQUENCE_RELEASE_HEIGHT_OFFSET_M)
