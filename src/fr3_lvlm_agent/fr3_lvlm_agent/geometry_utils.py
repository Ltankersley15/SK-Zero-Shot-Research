from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from geometry_msgs.msg import Quaternion


def wrap_pi(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convert roll, pitch, yaw (radians) to Quaternion.
    
    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
    
    Returns:
        Quaternion message
    """
    cr = math.cos(0.5 * roll)
    sr = math.sin(0.5 * roll)
    cp = math.cos(0.5 * pitch)
    sp = math.sin(0.5 * pitch)
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )


def compute_grasp_orientation_from_bbox(
    R_cam_in_base: np.ndarray,
    bbox_angle_rad: float | None,
    align_gripper_with_bbox: bool,
    bbox_yaw_offset_deg: float,
    bbox_yaw_smoothing_alpha: float,
    ema_grasp_yaw: float | None,
) -> Quaternion | None:
    """Compute grasp orientation quaternion from bounding box angle.
    
    Args:
        R_cam_in_base: 3x3 rotation matrix from camera frame to base frame
        bbox_angle_rad: Bounding box angle in radians (in image plane), or None
        align_gripper_with_bbox: Whether to enable alignment feature
        bbox_yaw_offset_deg: Additional yaw offset in degrees
        bbox_yaw_smoothing_alpha: EMA smoothing factor (0.0-1.0)
        ema_grasp_yaw: Previous EMA yaw value for smoothing, or None
    
    Returns:
        Quaternion for grasp orientation, or None if disabled/invalid
    """
    if not align_gripper_with_bbox:
        return None
    if bbox_angle_rad is None:
        return None

    # Image angle -> camera plane vector -> base XY yaw.
    theta = float(bbox_angle_rad) + math.radians(float(bbox_yaw_offset_deg))
    cam_x = R_cam_in_base[:, 0]
    cam_y = R_cam_in_base[:, 1]
    v_base = math.cos(theta) * cam_x + math.sin(theta) * cam_y
    vx = float(v_base[0])
    vy = float(v_base[1])
    if (vx * vx + vy * vy) < 1e-8:
        return None

    yaw = math.atan2(vy, vx)
    a = min(1.0, max(0.0, float(bbox_yaw_smoothing_alpha)))
    if ema_grasp_yaw is None:
        ema_grasp_yaw = yaw
    else:
        ema_grasp_yaw = wrap_pi(ema_grasp_yaw + a * wrap_pi(yaw - ema_grasp_yaw))
    yaw_use = float(ema_grasp_yaw)
    # Keep the same "tool down" attitude while changing in-plane yaw.
    return quat_from_rpy(math.pi, 0.0, yaw_use)


def map_color_uv_to_depth_uv(
    u: int,
    v: int,
    color_wh: Tuple[int, int],
    depth_wh: Tuple[int, int],
) -> tuple[int, int]:
    w_c, h_c = int(color_wh[0]), int(color_wh[1])
    w_d, h_d = int(depth_wh[0]), int(depth_wh[1])
    if w_c <= 0 or h_c <= 0:
        return int(u), int(v)
    u_d = int(round(float(u) * float(w_d) / float(w_c)))
    v_d = int(round(float(v) * float(h_d) / float(h_c)))
    return u_d, v_d


def backproject_pixel_to_cam(
    u: int,
    v: int,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    flip_x: bool = True,
    flip_y: bool = False,
) -> np.ndarray:
    """Backproject an image pixel to camera coordinates."""
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("fx and fy must be positive")
    x = (float(u) - float(cx)) * float(depth_m) / float(fx)
    y = (float(v) - float(cy)) * float(depth_m) / float(fy)
    if flip_x:
        x = -x
    if flip_y:
        y = -y
    return np.array([x, y, float(depth_m)], dtype=np.float64)


def project_cam_to_pixel(
    xyz_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    flip_x: bool = True,
    flip_y: bool = False,
) -> tuple[float, float]:
    """Project a camera-frame 3D point into image pixel coordinates."""
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("fx and fy must be positive")
    x = float(xyz_cam[0])
    y = float(xyz_cam[1])
    z = float(xyz_cam[2])
    if z <= 1e-6:
        return float("nan"), float("nan")
    if flip_x:
        u = (-float(fx) * x / z) + float(cx)
    else:
        u = (float(fx) * x / z) + float(cx)
    if flip_y:
        v = (-float(fy) * y / z) + float(cy)
    else:
        v = (float(fy) * y / z) + float(cy)
    return float(u), float(v)


def select_cross_depth_value(
    depth: np.ndarray,
    reproj_err: np.ndarray,
    max_err_px: float,
) -> float:
    if depth.size == 0 or reproj_err.size == 0:
        return float("nan")
    idx = int(np.argmin(reproj_err))
    best_err = float(reproj_err[idx])
    if (not np.isfinite(best_err)) or best_err > float(max_err_px):
        return float("nan")
    best_depth = float(depth[idx])
    return best_depth if np.isfinite(best_depth) else float("nan")


def project_pixel_to_plane(
    u: int,
    v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    R: np.ndarray,
    t: np.ndarray,
    plane_z: float,
) -> tuple[float, float] | None:
    if fx <= 0.0 or fy <= 0.0:
        return None
    dir_cam = np.array(
        [
            (float(u) - float(cx)) / float(fx),
            (float(v) - float(cy)) / float(fy),
            1.0,
        ],
        dtype=np.float64,
    )
    dir_base = R @ dir_cam
    origin = t
    dz = float(dir_base[2])
    if abs(dz) < 1e-6:
        return None
    s = (float(plane_z) - float(origin[2])) / dz
    if s <= 0.0:
        return None
    x = float(origin[0]) + s * float(dir_base[0])
    y = float(origin[1]) + s * float(dir_base[1])
    return x, y
