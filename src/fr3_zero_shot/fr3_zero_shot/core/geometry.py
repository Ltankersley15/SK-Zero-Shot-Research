from __future__ import annotations

import math


def xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def xyz_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(
        math.sqrt(
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
            + (float(a[2]) - float(b[2])) ** 2
        )
    )


def footprint_radius(footprint_xy: tuple[float, float]) -> float:
    return 0.5 * max(0.0, float(footprint_xy[0]), float(footprint_xy[1]))


def bounded_xy_offset(
    offset_xy: tuple[float, float],
    *,
    max_norm_m: float,
) -> tuple[float, float]:
    max_norm = max(0.0, float(max_norm_m))
    x = float(offset_xy[0])
    y = float(offset_xy[1])
    norm = math.hypot(x, y)
    if norm <= max_norm or norm <= 1e-9:
        return (x, y)
    scale = max_norm / norm
    return (float(x * scale), float(y * scale))


def segment_point_clearance_xy(
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    point_xy: tuple[float, float],
    *,
    radius_m: float = 0.0,
) -> float:
    sx, sy = float(start_xyz[0]), float(start_xyz[1])
    ex, ey = float(end_xyz[0]), float(end_xyz[1])
    px, py = float(point_xy[0]), float(point_xy[1])
    vx = ex - sx
    vy = ey - sy
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return float(math.hypot(px - sx, py - sy) - max(0.0, float(radius_m)))
    t = max(0.0, min(1.0, ((px - sx) * vx + (py - sy) * vy) / denom))
    cx = sx + t * vx
    cy = sy + t * vy
    return float(math.hypot(px - cx, py - cy) - max(0.0, float(radius_m)))

