"""Tests for gripper orientation alignment from bounding box."""

from __future__ import annotations

import math

import numpy as np

from fr3_lvlm_agent.geometry_utils import (
    compute_grasp_orientation_from_bbox,
    quat_from_rpy,
    wrap_pi,
)


def test_quat_from_rpy_identity() -> None:
    """Test quaternion from RPY with zero angles returns identity."""
    q = quat_from_rpy(0.0, 0.0, 0.0)
    assert abs(q.x) < 1e-6
    assert abs(q.y) < 1e-6
    assert abs(q.z) < 1e-6
    assert abs(q.w - 1.0) < 1e-6


def test_quat_from_rpy_tool_down() -> None:
    """Test quaternion for tool-down orientation (roll=pi, pitch=0, yaw=0)."""
    q = quat_from_rpy(math.pi, 0.0, 0.0)
    # Tool-down: roll=180deg around X axis
    assert abs(q.x - 1.0) < 1e-6 or abs(q.x + 1.0) < 1e-6
    assert abs(q.y) < 1e-6
    assert abs(q.z) < 1e-6
    assert abs(q.w) < 1e-6


def test_wrap_pi_normalizes() -> None:
    """Test that wrap_pi normalizes angles to [-pi, pi]."""
    assert abs(wrap_pi(0.0)) < 1e-6
    assert abs(wrap_pi(math.pi) - math.pi) < 1e-6  # pi -> pi (already in range)
    assert abs(wrap_pi(-math.pi) + math.pi) < 1e-6  # -pi -> -pi (already in range)
    assert abs(wrap_pi(2 * math.pi)) < 1e-6  # 2pi -> 0
    assert abs(wrap_pi(3 * math.pi) - math.pi) < 1e-6  # 3pi -> pi


def test_compute_grasp_orientation_disabled() -> None:
    """Test that None is returned when alignment is disabled."""
    R_cam_in_base = np.eye(3, dtype=np.float64)
    result = compute_grasp_orientation_from_bbox(
        R_cam_in_base=R_cam_in_base,
        bbox_angle_rad=0.5,
        align_gripper_with_bbox=False,
        bbox_yaw_offset_deg=0.0,
        bbox_yaw_smoothing_alpha=1.0,
        ema_grasp_yaw=None,
    )
    assert result is None


def test_compute_grasp_orientation_no_bbox_angle() -> None:
    """Test that None is returned when no bbox angle is available."""
    R_cam_in_base = np.eye(3, dtype=np.float64)
    result = compute_grasp_orientation_from_bbox(
        R_cam_in_base=R_cam_in_base,
        bbox_angle_rad=None,
        align_gripper_with_bbox=True,
        bbox_yaw_offset_deg=0.0,
        bbox_yaw_smoothing_alpha=1.0,
        ema_grasp_yaw=None,
    )
    assert result is None


def test_compute_grasp_orientation_basic() -> None:
    """Test basic grasp orientation computation with identity rotation."""
    R_cam_in_base = np.eye(3, dtype=np.float64)
    bbox_angle_rad = 0.0  # Horizontal bbox
    result = compute_grasp_orientation_from_bbox(
        R_cam_in_base=R_cam_in_base,
        bbox_angle_rad=bbox_angle_rad,
        align_gripper_with_bbox=True,
        bbox_yaw_offset_deg=0.0,
        bbox_yaw_smoothing_alpha=1.0,
        ema_grasp_yaw=None,
    )
    assert result is not None
    # With identity rotation and 0 bbox angle, yaw should be 0
    # Tool-down orientation: roll=pi, pitch=0, yaw=0
    assert abs(result.x - 1.0) < 1e-6 or abs(result.x + 1.0) < 1e-6
    assert abs(result.y) < 1e-6
    assert abs(result.z) < 1e-6
    assert abs(result.w) < 1e-6


def test_compute_grasp_orientation_with_offset() -> None:
    """Test grasp orientation with bbox yaw offset."""
    R_cam_in_base = np.eye(3, dtype=np.float64)
    bbox_angle_rad = math.pi / 4  # 45 degrees
    offset_deg = 45.0  # Additional 45 degree offset
    result = compute_grasp_orientation_from_bbox(
        R_cam_in_base=R_cam_in_base,
        bbox_angle_rad=bbox_angle_rad,
        align_gripper_with_bbox=True,
        bbox_yaw_offset_deg=offset_deg,
        bbox_yaw_smoothing_alpha=1.0,
        ema_grasp_yaw=None,
    )
    assert result is not None
    # Total angle should be 45 + 45 = 90 degrees = pi/2
    # Yaw in base frame should be pi/2


def test_compute_grasp_orientation_with_smoothing() -> None:
    """Test EMA smoothing of grasp yaw."""
    R_cam_in_base = np.eye(3, dtype=np.float64)
    prev_ema = 0.0
    new_bbox_angle = math.pi / 2  # 90 degrees
    alpha = 0.5  # Smoothing factor
    
    result = compute_grasp_orientation_from_bbox(
        R_cam_in_base=R_cam_in_base,
        bbox_angle_rad=new_bbox_angle,
        align_gripper_with_bbox=True,
        bbox_yaw_offset_deg=0.0,
        bbox_yaw_smoothing_alpha=alpha,
        ema_grasp_yaw=prev_ema,
    )
    assert result is not None
    # EMA should be: prev + alpha * (new - prev) = 0 + 0.5 * (pi/2 - 0) = pi/4


def test_compute_grasp_orientation_degenerate_axis() -> None:
    """Test that None is returned when projected axis is degenerate."""
    # Create a rotation matrix that projects to near-zero XY
    R_cam_in_base = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    result = compute_grasp_orientation_from_bbox(
        R_cam_in_base=R_cam_in_base,
        bbox_angle_rad=0.0,
        align_gripper_with_bbox=True,
        bbox_yaw_offset_deg=0.0,
        bbox_yaw_smoothing_alpha=1.0,
        ema_grasp_yaw=None,
    )
    # This should return None due to degenerate projection
    assert result is None
