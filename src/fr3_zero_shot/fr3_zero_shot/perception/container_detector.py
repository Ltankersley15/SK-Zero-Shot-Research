from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np

cv = importlib.import_module("cv" + "2")


@dataclass(frozen=True)
class ContainerBlob:
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    center_uv: tuple[float, float]


def detect_dark_neutral_container_blobs(
    rgb: np.ndarray,
    *,
    min_area_px: int = 800,
    max_area_px: int = 150000,
) -> list[ContainerBlob]:
    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
    dark_neutral = cv.inRange(hsv, (0, 0, 0), (179, 95, 105))
    kernel = np.ones((7, 7), np.uint8)
    mask = cv.morphologyEx(dark_neutral, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    num, _labels, stats, centroids = cv.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    blobs: list[ContainerBlob] = []
    for idx in range(1, num):
        area = int(stats[idx, cv.CC_STAT_AREA])
        if area < int(min_area_px) or area > int(max_area_px):
            continue
        x = int(stats[idx, cv.CC_STAT_LEFT])
        y = int(stats[idx, cv.CC_STAT_TOP])
        w = int(stats[idx, cv.CC_STAT_WIDTH])
        h = int(stats[idx, cv.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        aspect = float(w) / float(h)
        if not 0.35 <= aspect <= 2.8:
            continue
        u, v = centroids[idx]
        blobs.append(
            ContainerBlob(
                area_px=area,
                bbox_xyxy=(x, y, x + w, y + h),
                center_uv=(float(u), float(v)),
            )
        )
    return blobs
