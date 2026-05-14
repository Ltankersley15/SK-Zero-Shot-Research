from __future__ import annotations


def should_skip_sam2_for_yolo(
    *,
    attribute_required: bool,
    prefer_yolo_fast_path: bool,
    skip_sam2_when_yolo_present: bool,
    yolo_props_count: int,
    yolo_attr_count: int,
) -> bool:
    if not attribute_required:
        return False
    if not prefer_yolo_fast_path:
        return False
    if yolo_props_count <= 0:
        return False
    # Only trust the YOLO-only fast path when attribute matching yields a
    # single unambiguous candidate. Competing same-color/shape proposals need
    # the slower geometry-aware path.
    if yolo_attr_count == 1:
        return True
    if yolo_attr_count > 1:
        return False
    return bool(skip_sam2_when_yolo_present)
