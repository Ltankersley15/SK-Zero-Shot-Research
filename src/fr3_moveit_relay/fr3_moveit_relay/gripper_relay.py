#!/usr/bin/env python3
# Relay JointTrajectory gripper commands into JointState commands for Isaac.
import time
import threading
from typing import Dict, List

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException

from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState


class GripperTrajectoryRelay(Node):
    def __init__(self):
        super().__init__("gripper_relay")

        # Parameters
        self.declare_parameter("traj_topic", "/fr3_gripper_controller/joint_trajectory")
        self.declare_parameter("command_topic", "/fr3_arm_controller/joint_command")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("gripper_joints", ["fr3_finger_joint1", "fr3_finger_joint2"])
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("hold_sec", 1.0)
        self.declare_parameter("merge_with_joint_states", True)

        self.traj_topic = str(self.get_parameter("traj_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.gripper_joints = list(self.get_parameter("gripper_joints").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.hold_sec = float(self.get_parameter("hold_sec").value)
        self.merge_with_joint_states = bool(self.get_parameter("merge_with_joint_states").value)

        self._last_joint_state: JointState | None = None
        self._last_js_lock = threading.Lock()

        self.cb_group = ReentrantCallbackGroup()
        self.pub = self.create_publisher(JointState, self.command_topic, 10)
        self.sub = self.create_subscription(JointTrajectory, self.traj_topic, self._on_traj, 10, callback_group=self.cb_group)
        self.js_sub = self.create_subscription(JointState, self.joint_states_topic, self._on_joint_states, qos_profile_sensor_data)

        self.get_logger().info(f"Gripper relay: {self.traj_topic} -> {self.command_topic}")
        self.get_logger().info(f"gripper_joints={self.gripper_joints} rate_hz={self.rate_hz:.1f} hold_sec={self.hold_sec:.2f}")

    def _on_joint_states(self, msg: JointState):
        with self._last_js_lock:
            self._last_joint_state = msg

    def _on_traj(self, msg: JointTrajectory):
        if not msg.points:
            self.get_logger().warn("Received empty JointTrajectory for gripper.")
            return

        joint_names = list(msg.joint_names) if msg.joint_names else list(self.gripper_joints)
        positions = list(msg.points[-1].positions)
        if len(positions) != len(joint_names):
            self.get_logger().warn("Gripper traj joint_names length != positions length. Ignoring.")
            return

        t = threading.Thread(target=self._publish_burst, args=(joint_names, positions), daemon=True)
        t.start()

    def _compose_message(self, joint_names: List[str], positions: List[float]) -> JointState:
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()

        if self.merge_with_joint_states:
            with self._last_js_lock:
                last = self._last_joint_state
            if last is not None and last.name and last.position:
                name_to_pos: Dict[str, float] = {n: float(p) for n, p in zip(last.name, last.position)}
                for n, p in zip(joint_names, positions):
                    name_to_pos[n] = float(p)
                out.name = list(last.name)
                out.position = [name_to_pos.get(n, 0.0) for n in out.name]
                return out

        out.name = list(joint_names)
        out.position = [float(p) for p in positions]
        return out

    def _publish_burst(self, joint_names: List[str], positions: List[float]):
        n = max(1, int(max(self.rate_hz, 1.0) * max(self.hold_sec, 0.1)))
        dt = 1.0 / max(self.rate_hz, 1.0)
        for _ in range(n):
            self.pub.publish(self._compose_message(joint_names, positions))
            time.sleep(dt)


def main():
    rclpy.init()
    node = GripperTrajectoryRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except BaseException:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except BaseException:
            pass


if __name__ == "__main__":
    main()
