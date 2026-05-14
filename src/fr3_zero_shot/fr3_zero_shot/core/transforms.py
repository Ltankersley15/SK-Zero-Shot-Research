from __future__ import annotations

import math

from .types import ApproachFamily


def quaternion_multiply(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def normalize_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(v) / norm for v in q)  # type: ignore[return-value]


def quaternion_from_axis_angle(
    axis: tuple[float, float, float],
    angle_rad: float,
) -> tuple[float, float, float, float]:
    ax, ay, az = (float(axis[0]), float(axis[1]), float(axis[2]))
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    half = 0.5 * float(angle_rad)
    scale = math.sin(half) / norm
    return normalize_quaternion((ax * scale, ay * scale, az * scale, math.cos(half)))


def quaternion_from_yaw(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def top_down_quaternion(yaw_rad: float = 0.0) -> tuple[float, float, float, float]:
    down = (1.0, 0.0, 0.0, 0.0)
    return normalize_quaternion(quaternion_multiply(quaternion_from_yaw(yaw_rad), down))


def grasp_quaternion(family: ApproachFamily) -> tuple[float, float, float, float]:
    if family == ApproachFamily.TOP_DOWN_YAW_45:
        return top_down_quaternion(math.radians(45.0))
    if family in {ApproachFamily.TOP_DOWN, ApproachFamily.AUTO}:
        return top_down_quaternion(0.0)
    # Side families are wrist-pitch approximations for candidate generation.
    # MoveIt/IK owns final feasibility; the pipeline does not let language set this.
    if family == ApproachFamily.SIDE_LEFT:
        return normalize_quaternion(quaternion_multiply(quaternion_from_yaw(math.pi / 2.0), (0.0, 0.7071, 0.0, 0.7071)))
    if family == ApproachFamily.SIDE_RIGHT:
        return normalize_quaternion(quaternion_multiply(quaternion_from_yaw(-math.pi / 2.0), (0.0, 0.7071, 0.0, 0.7071)))
    if family == ApproachFamily.SIDE_FRONT:
        return normalize_quaternion((0.0, 0.7071, 0.0, 0.7071))
    if family == ApproachFamily.SIDE_BACK:
        return normalize_quaternion((0.0, -0.7071, 0.0, 0.7071))
    return top_down_quaternion(0.0)
