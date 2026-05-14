from __future__ import annotations


def bbox_center_xyxy(bbox_xyxy: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox_xyxy
    return (int(round((x0 + x1) * 0.5)), int(round((y0 + y1) * 0.5)))


def bbox_area_xyxy(bbox_xyxy: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = bbox_xyxy
    return max(0, int(x1) - int(x0)) * max(0, int(y1) - int(y0))

