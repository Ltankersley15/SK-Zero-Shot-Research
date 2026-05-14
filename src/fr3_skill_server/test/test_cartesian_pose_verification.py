from __future__ import annotations

import math

import pytest
from geometry_msgs.msg import Point, PoseStamped, Quaternion

from fr3_skill_server.skill_server import FR3SkillServer


def _pose(x: float, y: float, z: float, q: Quaternion | None = None) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "fr3_link0"
    pose.pose.position = Point(x=float(x), y=float(y), z=float(z))
    pose.pose.orientation = q if q is not None else Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
    return pose


def test_tcp_pose_from_ee_pose_applies_link_frame_offset() -> None:
    ee_pose = _pose(0.45, 0.10, 0.14)

    tcp_pose = FR3SkillServer._tcp_pose_from_ee_pose(ee_pose, (0.0, 0.0, 0.12))

    assert tcp_pose.pose.position.x == pytest.approx(0.45)
    assert tcp_pose.pose.position.y == pytest.approx(0.10)
    assert tcp_pose.pose.position.z == pytest.approx(0.02)


def test_apply_tcp_offset_does_not_mutate_requested_tcp_pose() -> None:
    server = FR3SkillServer.__new__(FR3SkillServer)
    server.ee_tcp_offset = (0.0, 0.0, 0.12)
    target_pose = _pose(0.45, 0.10, 0.42)

    ik_pose = server._apply_tcp_offset(target_pose)

    assert target_pose.pose.position.x == pytest.approx(0.45)
    assert target_pose.pose.position.y == pytest.approx(0.10)
    assert target_pose.pose.position.z == pytest.approx(0.42)
    assert ik_pose.pose.position.x == pytest.approx(0.45)
    assert ik_pose.pose.position.y == pytest.approx(0.10)
    assert ik_pose.pose.position.z == pytest.approx(0.54)


def test_pose_error_reports_cartesian_and_angular_distance() -> None:
    target_pose = _pose(0.60, -0.29, 0.02, Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
    actual_pose = _pose(0.70, -0.17, 0.03, Quaternion(x=0.0, y=0.0, z=math.sin(math.pi / 8), w=math.cos(math.pi / 8)))

    dx, dy, dz, pos_err, ang_err = FR3SkillServer._pose_error(target_pose, actual_pose)

    assert dx == pytest.approx(-0.10)
    assert dy == pytest.approx(-0.12)
    assert dz == pytest.approx(-0.01)
    assert pos_err == pytest.approx(math.sqrt((0.10**2) + (0.12**2) + (0.01**2)))
    assert ang_err == pytest.approx(math.pi / 4)


def test_build_corrected_pose_target_caps_large_corrections() -> None:
    server = FR3SkillServer.__new__(FR3SkillServer)
    server.cartesian_pose_max_correction_step = 0.05

    corrected = server._build_corrected_pose_target(
        current_target=_pose(0.60, -0.29, 0.02),
        dx=-0.10,
        dy=-0.12,
        dz=0.0,
    )

    applied_dx = corrected.pose.position.x - 0.60
    applied_dy = corrected.pose.position.y + 0.29
    applied_norm = math.sqrt((applied_dx * applied_dx) + (applied_dy * applied_dy))
    assert applied_norm == pytest.approx(0.05)
    assert corrected.pose.position.z == pytest.approx(0.02)


def test_pose_correction_progressed_requires_minimum_improvement() -> None:
    server = FR3SkillServer.__new__(FR3SkillServer)
    server.cartesian_pose_min_position_improvement = 0.005

    assert server._pose_correction_progressed(
        previous_pos_err=0.120,
        current_pos_err=0.110,
    )
    assert not server._pose_correction_progressed(
        previous_pos_err=0.120,
        current_pos_err=0.118,
    )
