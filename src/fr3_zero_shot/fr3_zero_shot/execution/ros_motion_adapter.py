from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformException, TransformListener

from fr3_skill_interfaces.srv import PlanExecutePoseGoal

from fr3_zero_shot.core.transforms import top_down_quaternion
from fr3_zero_shot.core.types import ObjectHypothesis

from .gripper_adapter import GRIPPER_CLOSE_M, GRIPPER_OPEN_M, contact_detected_from_width
from .motion_adapter import AdapterResult


class RosMotionAdapter:
    def __init__(
        self,
        node: Node,
        *,
        base_frame: str = 'fr3_link0',
        ee_link: str = 'fr3_link8',
        pose_service: str = '/fr3_skill_server/plan_execute_pose_goal',
        gripper_topic: str = '/fr3_gripper_controller/joint_trajectory',
        joint_states_topic: str = '/joint_states',
        gripper_joints: tuple[str, str] = (
            'fr3_finger_joint1',
            'fr3_finger_joint2',
        ),
        tcp_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.10),
        position_tolerance_m: float = 0.025,
        orientation_tolerance_rad: float = 0.45,
        service_timeout_sec: float = 45.0,
    ) -> None:
        self._node = node
        self._base_frame = str(base_frame)
        self._ee_link = str(ee_link)
        self._position_tolerance_m = float(position_tolerance_m)
        self._orientation_tolerance_rad = float(orientation_tolerance_rad)
        self._service_timeout_sec = max(1.0, float(service_timeout_sec))
        self._tcp_offset_xyz = tuple(float(v) for v in tcp_offset_xyz)
        self._pose_client = node.create_client(PlanExecutePoseGoal, pose_service)
        self._gripper_pub = node.create_publisher(JointTrajectory, gripper_topic, 10)
        self._gripper_joints = list(gripper_joints)
        self._latest_joint_state: JointState | None = None
        self._last_gripper_command = GRIPPER_OPEN_M
        self._holding = False
        self._joint_state_sub = node.create_subscription(
            JointState,
            str(joint_states_topic),
            self._joint_state_cb,
            qos_profile_sensor_data,
        )
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, node)

    def move_to_pose(
        self,
        *,
        xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float],
    ) -> AdapterResult:
        if not self._pose_client.wait_for_service(timeout_sec=5.0):
            return AdapterResult(False, 'pose service unavailable')

        req = PlanExecutePoseGoal.Request()
        req.target = PoseStamped()
        req.target.header = Header(frame_id=self._base_frame)
        req.target.pose.position.x = float(xyz[0])
        req.target.pose.position.y = float(xyz[1])
        req.target.pose.position.z = float(xyz[2])
        req.target.pose.orientation = Quaternion(
            x=float(quat_xyzw[0]),
            y=float(quat_xyzw[1]),
            z=float(quat_xyzw[2]),
            w=float(quat_xyzw[3]),
        )
        req.execute = True
        req.position_tolerance = self._position_tolerance_m
        req.orientation_tolerance = self._orientation_tolerance_rad

        started = time.time()
        future = self._pose_client.call_async(req)
        deadline = started + self._service_timeout_sec
        while rclpy.ok() and not future.done() and time.time() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.05)
        if not future.done():
            return AdapterResult(False, 'pose service timed out')
        try:
            result = future.result()
        except Exception as exc:
            return AdapterResult(False, f'pose service failed: {exc}')
        return AdapterResult(
            success=bool(result.success),
            message=str(result.message),
            planning_time=float(result.planning_time),
        )

    def open_gripper(self) -> AdapterResult:
        self._send_gripper(GRIPPER_OPEN_M)
        self._last_gripper_command = GRIPPER_OPEN_M
        self._holding = False
        return AdapterResult(True, 'gripper open command published')

    def close_gripper(
        self,
        *,
        position_m: float = GRIPPER_CLOSE_M,
        contact_margin_m: float = 0.002,
    ) -> AdapterResult:
        self._send_gripper(float(position_m))
        self._last_gripper_command = float(position_m)
        sensed = self.current_gripper_width()
        contact = self.gripper_contact_detected(commanded_width=float(position_m), margin_m=float(contact_margin_m))
        self._holding = bool(contact)
        effort = self.current_gripper_effort()
        detail = 'unknown'
        if sensed is not None:
            detail = f'sensed_width={sensed:.4f}'
            if effort is not None:
                detail += f' sensed_effort={effort:.4f}'
        return AdapterResult(bool(contact), f'gripper close contact={bool(contact)} {detail}')

    def verify_carry(self, source: ObjectHypothesis) -> bool:
        _ = source
        return bool(self._holding and self.gripper_contact_detected(commanded_width=GRIPPER_CLOSE_M))

    def verify_place_in(
        self,
        held_object: ObjectHypothesis,
        container: ObjectHypothesis,
    ) -> bool:
        _ = held_object
        _ = container
        return not self._holding

    def current_tcp_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._ee_link,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
        except TransformException:
            return None
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        q_xyzw = (float(q.x), float(q.y), float(q.z), float(q.w))
        dx, dy, dz = _rotate_vector(q_xyzw, self._tcp_offset_xyz)
        return (
            (float(t.x + dx), float(t.y + dy), float(t.z + dz)),
            q_xyzw,
        )

    def default_place_quat_xyzw(self) -> tuple[float, float, float, float]:
        return top_down_quaternion(0.0)

    def current_gripper_width(self) -> float | None:
        state = self._latest_joint_state
        if state is None:
            return None
        values: list[float] = []
        for joint in self._gripper_joints:
            if joint in state.name:
                idx = state.name.index(joint)
                if idx < len(state.position):
                    values.append(float(state.position[idx]))
        if not values:
            return None
        return float(sum(values) / len(values))

    def current_gripper_effort(self) -> float | None:
        state = self._latest_joint_state
        if state is None or not state.effort:
            return None
        values: list[float] = []
        for joint in self._gripper_joints:
            if joint in state.name:
                idx = state.name.index(joint)
                if idx < len(state.effort):
                    values.append(abs(float(state.effort[idx])))
        if not values:
            return None
        return float(sum(values) / len(values))

    def gripper_contact_detected(
        self,
        *,
        commanded_width: float = GRIPPER_CLOSE_M,
        margin_m: float = 0.002,
    ) -> bool:
        width = self.current_gripper_width()
        return contact_detected_from_width(width, commanded_width_m=commanded_width, margin_m=margin_m)

    def gripper_is_open(self, *, open_threshold_m: float = 0.030) -> bool:
        width = self.current_gripper_width()
        if width is None:
            return self._last_gripper_command >= open_threshold_m
        return bool(float(width) >= float(open_threshold_m))

    def _send_gripper(self, position_m: float) -> None:
        traj = JointTrajectory()
        traj.joint_names = list(self._gripper_joints)
        point = JointTrajectoryPoint()
        point.positions = [float(position_m), float(position_m)]
        point.time_from_start.sec = 1
        traj.points = [point]
        self._gripper_pub.publish(traj)
        end = time.time() + 1.2
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def _joint_state_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg


def _rotate_vector(
    quat_xyzw: tuple[float, float, float, float],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z, w = quat_xyzw
    vx, vy, vz = vec
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm > 1e-9:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )
