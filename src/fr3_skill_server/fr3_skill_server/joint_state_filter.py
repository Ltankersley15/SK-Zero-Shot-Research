#!/usr/bin/env python3
from typing import Dict, List

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState


ARM_JOINTS: List[str] = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3",
    "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7",
]


class JointStateFilter(Node):
    """
    Filters a JointState stream down to the 7 arm joints only.

    Default:
      in_topic  = /joint_states            (Isaac publishes arm + fingers here)
      out_topic = /fr3/joint_states_arm    (arm-only topic consumed by MoveIt/RSP/skill_server)

    Override via params:
      ros2 run fr3_skill_server joint_state_filter --ros-args -p in_topic:=... -p out_topic:=...
    """

    def __init__(self):
        super().__init__("fr3_joint_state_filter")

        self.declare_parameter("in_topic", "/joint_states")
        self.declare_parameter("out_topic", "/fr3/joint_states_arm")

        self.in_topic = str(self.get_parameter("in_topic").value)
        self.out_topic = str(self.get_parameter("out_topic").value)

        self.pub = self.create_publisher(JointState, self.out_topic, 10)
        self.sub = self.create_subscription(JointState, self.in_topic, self.cb, qos_profile_sensor_data)

        self.get_logger().info(f"Filtering joint states: {self.in_topic} -> {self.out_topic}")
        self.get_logger().info("Publishing ONLY: " + ", ".join(ARM_JOINTS))

    def cb(self, msg: JointState):
        if not msg.name or not msg.position:
            return

        name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(msg.name)}
        if not all(j in name_to_idx for j in ARM_JOINTS):
            return

        out = JointState()
        out.header = msg.header
        out.name = list(ARM_JOINTS)
        out.position = [float(msg.position[name_to_idx[j]]) for j in ARM_JOINTS]

        # preserve velocity/effort if present (optional)
        if msg.velocity and len(msg.velocity) == len(msg.name):
            out.velocity = [float(msg.velocity[name_to_idx[j]]) for j in ARM_JOINTS]
        if msg.effort and len(msg.effort) == len(msg.name):
            out.effort = [float(msg.effort[name_to_idx[j]]) for j in ARM_JOINTS]

        self.pub.publish(out)


def main():
    rclpy.init()
    node = JointStateFilter()
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
