# ~/ws_moveit2/src/fr3_lvlm_agent/fr3_lvlm_agent/camera_info_repub.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo


class CameraInfoRepub(Node):
    def __init__(self):
        super().__init__("camera_info_repub")

        # Params
        self.declare_parameter("in_topic", "/camera_info")
        self.declare_parameter("out_color_topic", "/fr3/d455/color/camera_info")
        self.declare_parameter("out_depth_topic", "/fr3/d455/depth/camera_info")
        self.declare_parameter("color_frame", "d455_color_optical_frame")
        self.declare_parameter("depth_frame", "d455_depth_optical_frame")

        self.in_topic = self.get_parameter("in_topic").value
        self.out_color_topic = self.get_parameter("out_color_topic").value
        self.out_depth_topic = self.get_parameter("out_depth_topic").value
        self.color_frame = self.get_parameter("color_frame").value
        self.depth_frame = self.get_parameter("depth_frame").value

        self.pub_color = self.create_publisher(CameraInfo, self.out_color_topic, 10)
        self.pub_depth = self.create_publisher(CameraInfo, self.out_depth_topic, 10)

        self.sub = self.create_subscription(CameraInfo, self.in_topic, self.cb, 10)

        self.get_logger().info(f"Republishing {self.in_topic} -> {self.out_color_topic} [{self.color_frame}]")
        self.get_logger().info(f"Republishing {self.in_topic} -> {self.out_depth_topic} [{self.depth_frame}]")

    def cb(self, msg: CameraInfo):
        # Color info
        c = CameraInfo()
        c.header = msg.header
        c.header.frame_id = self.color_frame
        c.height = msg.height
        c.width = msg.width
        c.distortion_model = msg.distortion_model
        c.d = list(msg.d)
        c.k = list(msg.k)
        c.r = list(msg.r)
        c.p = list(msg.p)
        c.binning_x = msg.binning_x
        c.binning_y = msg.binning_y
        c.roi = msg.roi
        self.pub_color.publish(c)

        # Depth info (same intrinsics; different frame)
        d = CameraInfo()
        d.header = msg.header
        d.header.frame_id = self.depth_frame
        d.height = msg.height
        d.width = msg.width
        d.distortion_model = msg.distortion_model
        d.d = list(msg.d)
        d.k = list(msg.k)
        d.r = list(msg.r)
        d.p = list(msg.p)
        d.binning_x = msg.binning_x
        d.binning_y = msg.binning_y
        d.roi = msg.roi
        self.pub_depth.publish(d)


def main():
    rclpy.init()
    node = CameraInfoRepub()
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
