from __future__ import annotations

from types import SimpleNamespace

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from fr3_moveit_relay.fjt_relay import FollowJointTrajectoryRelay


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, msg) -> None:
        self.messages.append(msg)


class _Clock:
    class _Now:
        @staticmethod
        def to_msg():
            return SimpleNamespace()

    @staticmethod
    def now():
        return _Clock._Now()


class _Logger:
    def info(self, _msg: str) -> None:
        return

    def warn(self, _msg: str) -> None:
        return

    def error(self, _msg: str) -> None:
        return


def test_publish_point_includes_latched_gripper_targets() -> None:
    relay = FollowJointTrajectoryRelay.__new__(FollowJointTrajectoryRelay)
    relay.enable_gripper_latch = True
    relay.gripper_joints = ["fr3_finger_joint1", "fr3_finger_joint2"]
    relay._latched_gripper_targets = {
        "fr3_finger_joint1": 0.015,
        "fr3_finger_joint2": 0.015,
    }
    relay.cmd_pub = _Publisher()
    relay.get_clock = lambda: _Clock()

    relay._publish_point(["fr3_joint1", "fr3_joint2"], [0.1, 0.2])

    msg = relay.cmd_pub.messages[-1]
    assert msg.name == [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_finger_joint1",
        "fr3_finger_joint2",
    ]
    assert list(msg.position) == [0.1, 0.2, 0.015, 0.015]


def test_on_gripper_traj_latches_latest_targets() -> None:
    relay = FollowJointTrajectoryRelay.__new__(FollowJointTrajectoryRelay)
    relay.enable_gripper_latch = True
    relay.gripper_joints = ["fr3_finger_joint1", "fr3_finger_joint2"]
    relay._latched_gripper_targets = {}
    relay.get_logger = lambda: _Logger()

    traj = JointTrajectory()
    traj.joint_names = ["fr3_finger_joint1", "fr3_finger_joint2"]
    point = JointTrajectoryPoint()
    point.positions = [0.012, 0.013]
    traj.points = [point]

    relay._on_gripper_traj(traj)

    assert relay._latched_gripper_targets == {
        "fr3_finger_joint1": 0.012,
        "fr3_finger_joint2": 0.013,
    }


def test_desired_positions_at_time_interpolates_between_points() -> None:
    relay = FollowJointTrajectoryRelay.__new__(FollowJointTrajectoryRelay)
    relay.interpolate_trajectory = True

    p0 = JointTrajectoryPoint()
    p0.positions = [0.0, 0.0]
    p0.time_from_start.sec = 0
    p0.time_from_start.nanosec = 0

    p1 = JointTrajectoryPoint()
    p1.positions = [1.0, -1.0]
    p1.time_from_start.sec = 2
    p1.time_from_start.nanosec = 0

    desired, idx = relay._desired_positions_at_time([p0, p1], 1.0)

    assert idx == 0
    assert desired == [0.5, -0.5]


def test_desired_positions_at_time_can_hold_sparse_waypoints_when_interpolation_disabled() -> None:
    relay = FollowJointTrajectoryRelay.__new__(FollowJointTrajectoryRelay)
    relay.interpolate_trajectory = False

    p0 = JointTrajectoryPoint()
    p0.positions = [0.1, 0.2]
    p0.time_from_start.sec = 0
    p0.time_from_start.nanosec = 0

    p1 = JointTrajectoryPoint()
    p1.positions = [0.9, 1.2]
    p1.time_from_start.sec = 4
    p1.time_from_start.nanosec = 0

    desired, idx = relay._desired_positions_at_time([p0, p1], 2.0)

    assert idx == 0
    assert desired == [0.1, 0.2]
