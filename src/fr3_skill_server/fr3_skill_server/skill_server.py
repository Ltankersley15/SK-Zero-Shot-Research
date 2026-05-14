#ws_moveit2/src/fr3_skill_server/fr3_skill_server/skill_server.py
#!/usr/bin/env python3
import math
import time
from typing import Optional, List, Tuple

import threading
import concurrent.futures

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from rclpy.duration import Duration

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf2_ros import Buffer, TransformException, TransformListener

from fr3_skill_interfaces.srv import PlanExecuteJointGoal, PlanExecutePoseGoal

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK

from control_msgs.action import FollowJointTrajectory


JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3",
    "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"
]

DEFAULT_BASE_FRAME = "fr3_link0"
EE_LINK = "fr3_link8"


class FR3SkillServer(Node):
    def __init__(self):
        super().__init__("fr3_skill_server")
        self.cb_group = ReentrantCallbackGroup()

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        try:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        except Exception:
            pass

        self.declare_parameter("joint_states_topic", "/fr3/joint_states_arm")
        self.declare_parameter("base_frame", DEFAULT_BASE_FRAME)
        self.declare_parameter("ee_link", EE_LINK)
        self.declare_parameter("ee_tcp_offset_x", 0.0)
        self.declare_parameter("ee_tcp_offset_y", 0.0)
        self.declare_parameter("ee_tcp_offset_z", 0.10)
        self.declare_parameter("fjt_action_name", "")
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("planning_attempts", 5)
        self.declare_parameter("joint_goal_constraint_tolerance", 0.01)
        self.declare_parameter("max_velocity_scaling_factor", 0.4)
        self.declare_parameter("max_acceleration_scaling_factor", 0.4)
        self.declare_parameter("replan", True)
        self.declare_parameter("replan_attempts", 2)
        self.declare_parameter("replan_delay", 0.2)
        self.declare_parameter("cartesian_pose_verification_enabled", True)
        self.declare_parameter("cartesian_pose_correction_attempts", 1)
        self.declare_parameter("cartesian_pose_max_correction_step", 0.06)
        self.declare_parameter("cartesian_pose_min_position_improvement", 0.005)
        self.declare_parameter("cartesian_pose_default_position_tolerance", 0.02)
        self.declare_parameter("cartesian_pose_default_orientation_tolerance", 0.35)
        self.declare_parameter("cartesian_pose_tf_timeout", 0.25)
        self.declare_parameter("cartesian_pose_post_execute_settle_time", 0.0)

        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_link = str(self.get_parameter("ee_link").value)
        self.ee_link_fallback = EE_LINK
        self._ee_link_fallback_used = False
        self.ee_tcp_offset = (
            float(self.get_parameter("ee_tcp_offset_x").value),
            float(self.get_parameter("ee_tcp_offset_y").value),
            float(self.get_parameter("ee_tcp_offset_z").value),
        )
        self.allowed_planning_time = float(self.get_parameter("allowed_planning_time").value)
        self.planning_attempts = int(self.get_parameter("planning_attempts").value)
        self.joint_goal_constraint_tolerance = max(
            0.0, float(self.get_parameter("joint_goal_constraint_tolerance").value)
        )
        self.max_velocity_scaling_factor = float(
            self.get_parameter("max_velocity_scaling_factor").value
        )
        self.max_acceleration_scaling_factor = float(
            self.get_parameter("max_acceleration_scaling_factor").value
        )
        self.replan = bool(self.get_parameter("replan").value)
        self.replan_attempts = int(self.get_parameter("replan_attempts").value)
        self.replan_delay = float(self.get_parameter("replan_delay").value)
        self.cartesian_pose_verification_enabled = bool(
            self.get_parameter("cartesian_pose_verification_enabled").value
        )
        self.cartesian_pose_correction_attempts = max(
            0, int(self.get_parameter("cartesian_pose_correction_attempts").value)
        )
        self.cartesian_pose_max_correction_step = max(
            0.0, float(self.get_parameter("cartesian_pose_max_correction_step").value)
        )
        self.cartesian_pose_min_position_improvement = max(
            0.0, float(self.get_parameter("cartesian_pose_min_position_improvement").value)
        )
        self.cartesian_pose_default_position_tolerance = max(
            0.0, float(self.get_parameter("cartesian_pose_default_position_tolerance").value)
        )
        self.cartesian_pose_default_orientation_tolerance = max(
            0.0, float(self.get_parameter("cartesian_pose_default_orientation_tolerance").value)
        )
        self.cartesian_pose_tf_timeout = max(
            0.01, float(self.get_parameter("cartesian_pose_tf_timeout").value)
        )
        self.cartesian_pose_post_execute_settle_time = max(
            0.0, float(self.get_parameter("cartesian_pose_post_execute_settle_time").value)
        )

        self.move_action_name = "/fr3/move_action"
        self.ik_service_name = "/fr3/compute_ik"
        fjt_action_override = str(self.get_parameter("fjt_action_name").value).strip()
        self.fjt_action_name = self._pick_fjt_action_name(override=fjt_action_override)

        self.get_logger().info(f"use_sim_time: {self.get_parameter('use_sim_time').value}")
        self.get_logger().info(f"Joint states topic: {self.joint_states_topic}")
        self.get_logger().info(f"Base frame: {self.base_frame}")
        self.get_logger().info(f"IK ee_link: {self.ee_link}")
        self.get_logger().info(f"IK ee_link fallback: {self.ee_link_fallback}")
        self.get_logger().info(
            f"IK ee_tcp_offset(x,y,z): ({self.ee_tcp_offset[0]:.4f}, "
            f"{self.ee_tcp_offset[1]:.4f}, {self.ee_tcp_offset[2]:.4f})"
        )
        self.get_logger().info(
            f"Joint goal constraint tolerance: {self.joint_goal_constraint_tolerance:.4f} rad"
        )
        self.get_logger().info(
            f"Cartesian post-execute settle time: {self.cartesian_pose_post_execute_settle_time:.3f}s"
        )
        self.get_logger().info(f"Using MoveGroup action: {self.move_action_name}")
        self.get_logger().info(f"Using IK service:     {self.ik_service_name}")
        self.get_logger().info(f"Using FJT action:     {self.fjt_action_name}")

        self.move_group = ActionClient(self, MoveGroup, self.move_action_name, callback_group=self.cb_group)
        self.fjt = ActionClient(self, FollowJointTrajectory, self.fjt_action_name, callback_group=self.cb_group)
        self.ik_client = self.create_client(GetPositionIK, self.ik_service_name)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.latest_arm_js: Optional[JointState] = None
        self.create_subscription(JointState, self.joint_states_topic, self._js_cb, qos_profile_sensor_data)

        self.srv_joint = self.create_service(
            PlanExecuteJointGoal,
            "/fr3_skill_server/plan_execute_joint_goal",
            self.handle_plan_execute_joint_goal,
            callback_group=self.cb_group
        )
        self.srv_pose = self.create_service(
            PlanExecutePoseGoal,
            "/fr3_skill_server/plan_execute_pose_goal",
            self.handle_plan_execute_pose_goal,
            callback_group=self.cb_group
        )

        self.get_logger().info("FR3SkillServer up. Ready.")

    def _js_cb(self, msg: JointState):
        if not msg.name or not msg.position:
            return
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        if not all(j in name_to_pos for j in JOINT_NAMES):
            return

        js = JointState()
        js.header = msg.header
        js.name = list(JOINT_NAMES)
        js.position = [float(name_to_pos[j]) for j in JOINT_NAMES]
        self.latest_arm_js = js

    def _pick_fjt_action_name(self, *, override: str = "") -> str:
        if override:
            return override
        candidates = [
            "/fr3_arm_controller/follow_joint_trajectory_relay",
            "/fr3/fr3_arm_controller/follow_joint_trajectory_relay",
            "/fr3_arm_controller/follow_joint_trajectory",
            "/fr3/fr3_arm_controller/follow_joint_trajectory",
        ]
        for name in candidates:
            tmp = ActionClient(self, FollowJointTrajectory, name, callback_group=self.cb_group)
            if tmp.wait_for_server(timeout_sec=0.5):
                return name
        return candidates[0]

    def _ecode_name(self, code: int) -> str:
        for k, v in MoveItErrorCodes.__dict__.items():
            if isinstance(v, int) and v == code:
                return k
        return f"UNKNOWN({code})"

    @staticmethod
    def _fjt_error_name(code: int) -> str:
        code = int(code)
        mapping = {
            int(FollowJointTrajectory.Result.SUCCESSFUL): "SUCCESSFUL",
            int(FollowJointTrajectory.Result.INVALID_GOAL): "INVALID_GOAL",
            int(FollowJointTrajectory.Result.INVALID_JOINTS): "INVALID_JOINTS",
            int(FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP): "OLD_HEADER_TIMESTAMP",
            int(FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED): "PATH_TOLERANCE_VIOLATED",
            int(FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED): "GOAL_TOLERANCE_VIOLATED",
        }
        return mapping.get(code, f"UNKNOWN({code})")

    def _wait_for_servers(self, timeout_sec: float = 3.0) -> Optional[str]:
        if not self.move_group.wait_for_server(timeout_sec=timeout_sec):
            return f"MoveGroup not available at {self.move_action_name}"
        if not self.fjt.wait_for_server(timeout_sec=timeout_sec):
            return f"FJT not available at {self.fjt_action_name}"
        if not self.ik_client.wait_for_service(timeout_sec=timeout_sec):
            return f"IK service not available at {self.ik_service_name}"
        return None

    def _spin_wait(self, fut, timeout_sec: float, label: str) -> bool:
        """
        DO NOT call rclpy.spin_once() here.
        The MultiThreadedExecutor in main() is already spinning this node.
        Just wait for the future completion signal.
        """
        done_evt = threading.Event()

        def _done_cb(_):
            done_evt.set()

        try:
            fut.add_done_callback(_done_cb)
        except Exception:
            # Some rclpy futures are not standard; still try to block safely
            pass

        if not done_evt.wait(timeout=timeout_sec):
            self.get_logger().error(f"Timeout waiting for {label}")
            return False
        return True
        
    def _build_joint_constraints(self, joints_7: List[float]) -> Constraints:
        c = Constraints()
        c.name = "joint_goal"
        for name, pos in zip(JOINT_NAMES, joints_7):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = float(self.joint_goal_constraint_tolerance)
            jc.tolerance_below = float(self.joint_goal_constraint_tolerance)
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    @staticmethod
    def _joint_delta_summary(requested: List[float], planned: List[float]) -> tuple[str, float]:
        rows = []
        max_abs = 0.0
        for name, req, got in zip(JOINT_NAMES, requested, planned):
            delta = float(got - req)
            max_abs = max(max_abs, abs(delta))
            rows.append(
                f"{name}=req:{float(req):+.4f} plan:{float(got):+.4f} delta:{delta:+.4f}"
            )
        return "; ".join(rows), max_abs

    def _make_start_state(self) -> Optional[RobotState]:
        if self.latest_arm_js is None:
            return None
        rs = RobotState()
        rs.joint_state = self.latest_arm_js
        rs.is_diff = True
        rs.joint_state.header.stamp = self.get_clock().now().to_msg()
        return rs

    def _plan_joint_goal(self, joints_7: List[float], allowed_time: Optional[float] = None):
        if allowed_time is None:
            allowed_time = self.allowed_planning_time
        constraints = self._build_joint_constraints(joints_7)

        goal = MoveGroup.Goal()
        goal.request.pipeline_id = "ompl"
        goal.request.planner_id = "RRTConnectkConfigDefault"

        goal.request.group_name = "fr3_arm"
        goal.request.goal_constraints = [constraints]
        goal.request.allowed_planning_time = float(allowed_time)
        goal.request.num_planning_attempts = int(self.planning_attempts)
        goal.request.max_velocity_scaling_factor = float(self.max_velocity_scaling_factor)
        goal.request.max_acceleration_scaling_factor = float(self.max_acceleration_scaling_factor)

        start_state = self._make_start_state()
        if start_state is not None:
            goal.request.start_state = start_state

        goal.planning_options.plan_only = True
        goal.planning_options.replan = bool(self.replan)
        goal.planning_options.replan_attempts = int(self.replan_attempts)
        goal.planning_options.replan_delay = float(self.replan_delay)

        start = time.time()
        send_fut = self.move_group.send_goal_async(goal)
        if not self._spin_wait(send_fut, 10.0, "MoveGroup send_goal (joint)"):
            return None, 0.0, "MoveGroup send timeout (joint)"
        gh = send_fut.result()
        if gh is None or not gh.accepted:
            return None, 0.0, "MoveGroup goal rejected (joint)"

        res_fut = gh.get_result_async()
        if not self._spin_wait(res_fut, 25.0, "MoveGroup result (joint)"):
            return None, 0.0, "MoveGroup result timeout (joint)"
        wrap = res_fut.result()
        if wrap is None:
            return None, 0.0, "MoveGroup returned no result (joint)"

        res = wrap.result
        planning_time = time.time() - start

        code = int(res.error_code.val)
        if code != MoveItErrorCodes.SUCCESS:
            return None, planning_time, f"Planning failed: {code} ({self._ecode_name(code)})"

        jt = res.planned_trajectory.joint_trajectory
        if not jt.joint_names or not jt.points:
            return None, planning_time, "MoveIt returned empty trajectory"

        planned_name_to_pos = {
            name: float(pos)
            for name, pos in zip(jt.joint_names, jt.points[-1].positions)
        }
        planned_final = [planned_name_to_pos.get(name, float("nan")) for name in JOINT_NAMES]
        delta_summary, max_abs_delta = self._joint_delta_summary(joints_7, planned_final)
        if max_abs_delta > max(0.03, self.joint_goal_constraint_tolerance * 1.5):
            self.get_logger().warn(
                "Planned final joints deviate from requested IK target: "
                f"max_abs_delta={max_abs_delta:.4f} rad | {delta_summary}"
            )

        return jt, planning_time, f"Planned joint goal ({len(jt.points)} points)"

    def _normalize_trajectory_for_controller(self, jt):
        """
        Many FJT implementations expect a non-zero header.stamp and strictly positive time_from_start.
        This function makes that true without changing the path shape.
        """
        # stamp start slightly in the future
        start_ns = self.get_clock().now().nanoseconds + int(0.10 * 1e9)
        jt.header.stamp.sec = int(start_ns // 1_000_000_000)
        jt.header.stamp.nanosec = int(start_ns % 1_000_000_000)

        # Ensure first point time_from_start > 0
        if jt.points:
            p0 = jt.points[0]
            t0 = p0.time_from_start.sec + p0.time_from_start.nanosec * 1e-9
            if t0 <= 0.0:
                # shift all points by +0.05s
                shift = 0.05
                for p in jt.points:
                    t = p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                    t += shift
                    sec = int(t)
                    nsec = int((t - sec) * 1e9)
                    p.time_from_start.sec = sec
                    p.time_from_start.nanosec = nsec
        return jt

    def _execute_trajectory(self, jt):
        jt = self._normalize_trajectory_for_controller(jt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt

        send_fut = self.fjt.send_goal_async(goal)
        if not self._spin_wait(send_fut, 10.0, "FJT send_goal"):
            return False, "Controller send timeout"

        gh = send_fut.result()
        if gh is None or not gh.accepted:
            return False, "Controller rejected goal"

        res_fut = gh.get_result_async()
        ok = self._spin_wait(res_fut, 30.0, "FJT result")
        if not ok:
            # attempt cancel so we don't wedge permanently
            try:
                gh.cancel_goal_async()
            except Exception:
                pass
            return False, "Controller result timeout (cancel attempted)"

        wrap = res_fut.result()
        if wrap is None or wrap.result is None:
            return False, "Controller returned no result"

        res = wrap.result
        if int(res.error_code) != 0:
            code = int(res.error_code)
            return False, f"Controller error_code={code} ({self._fjt_error_name(code)})"

        return True, "Executed successfully"

    def _compute_ik_with_link(self, target: PoseStamped, ik_link_name: str) -> Tuple[Optional[List[float]], str]:
        req = GetPositionIK.Request()
        req.ik_request.group_name = "fr3_arm"
        req.ik_request.ik_link_name = ik_link_name
        req.ik_request.pose_stamped = self._apply_tcp_offset(target)

        if self.latest_arm_js is not None:
            req.ik_request.robot_state.joint_state = self.latest_arm_js

        req.ik_request.timeout.sec = 1
        req.ik_request.timeout.nanosec = 0

        fut = self.ik_client.call_async(req)
        if not self._spin_wait(fut, 5.0, "compute_ik"):
            return None, "IK call timeout"

        resp = fut.result()
        if resp is None:
            return None, "IK returned no response"

        code = int(resp.error_code.val)
        if code != MoveItErrorCodes.SUCCESS:
            return None, f"IK failed: {code} ({self._ecode_name(code)})"

        js = resp.solution.joint_state
        name_to_pos = {n: p for n, p in zip(js.name, js.position)}
        joints_7 = []
        for n in JOINT_NAMES:
            if n not in name_to_pos:
                return None, f"IK solution missing joint {n}"
            joints_7.append(float(name_to_pos[n]))

        return joints_7, "IK success"

    def _compute_ik(self, target: PoseStamped) -> Tuple[Optional[List[float]], str]:
        joints_7, msg = self._compute_ik_with_link(target, self.ee_link)
        if joints_7 is not None:
            return joints_7, msg

        if self.ee_link != self.ee_link_fallback:
            fallback_joints, fallback_msg = self._compute_ik_with_link(target, self.ee_link_fallback)
            if fallback_joints is not None:
                if not self._ee_link_fallback_used:
                    self.get_logger().warn(
                        f"[IK] Configured ee_link='{self.ee_link}' failed; "
                        f"falling back to ee_link='{self.ee_link_fallback}'."
                    )
                    self._ee_link_fallback_used = True
                self.ee_link = self.ee_link_fallback
                return fallback_joints, f"IK success (fallback ee_link={self.ee_link_fallback})"
            return None, f"{msg}; fallback {self.ee_link_fallback}: {fallback_msg}"

        return None, msg

    def _apply_tcp_offset(self, target: PoseStamped) -> PoseStamped:
        """
        Convert a desired TCP pose into the IK link pose.
        Offset is expressed in the IK link frame.
        x_ik = x_tcp - R_tcp * offset
        """
        ox, oy, oz = self.ee_tcp_offset
        if abs(ox) < 1e-9 and abs(oy) < 1e-9 and abs(oz) < 1e-9:
            return target

        q = target.pose.orientation
        qx = float(q.x)
        qy = float(q.y)
        qz = float(q.z)
        qw = float(q.w)
        n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
        if n > 0.0:
            qx /= n
            qy /= n
            qz /= n
            qw /= n

        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)
        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)
        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)

        dx = r00 * ox + r01 * oy + r02 * oz
        dy = r10 * ox + r11 * oy + r12 * oz
        dz = r20 * ox + r21 * oy + r22 * oz

        out = PoseStamped()
        out.header = target.header
        out.pose.position = Point(
            x=float(target.pose.position.x - dx),
            y=float(target.pose.position.y - dy),
            z=float(target.pose.position.z - dz),
        )
        out.pose.orientation = Quaternion(
            x=float(target.pose.orientation.x),
            y=float(target.pose.orientation.y),
            z=float(target.pose.orientation.z),
            w=float(target.pose.orientation.w),
        )
        return out

    @staticmethod
    def _normalize_quaternion(q: Quaternion) -> tuple[float, float, float, float]:
        qx = float(q.x)
        qy = float(q.y)
        qz = float(q.z)
        qw = float(q.w)
        n = math.sqrt((qx * qx) + (qy * qy) + (qz * qz) + (qw * qw))
        if n <= 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        return qx / n, qy / n, qz / n, qw / n

    @classmethod
    def _quat_to_rot_matrix(cls, q: Quaternion) -> tuple[tuple[float, float, float], ...]:
        qx, qy, qz, qw = cls._normalize_quaternion(q)
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        return (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        )

    @classmethod
    def _tcp_pose_from_ee_pose(
        cls,
        ee_pose: PoseStamped,
        tcp_offset: tuple[float, float, float],
    ) -> PoseStamped:
        rot = cls._quat_to_rot_matrix(ee_pose.pose.orientation)
        ox, oy, oz = tcp_offset
        dx = (rot[0][0] * ox) + (rot[0][1] * oy) + (rot[0][2] * oz)
        dy = (rot[1][0] * ox) + (rot[1][1] * oy) + (rot[1][2] * oz)
        dz = (rot[2][0] * ox) + (rot[2][1] * oy) + (rot[2][2] * oz)

        tcp_pose = PoseStamped()
        tcp_pose.header = ee_pose.header
        tcp_pose.pose.orientation = ee_pose.pose.orientation
        tcp_pose.pose.position.x = float(ee_pose.pose.position.x + dx)
        tcp_pose.pose.position.y = float(ee_pose.pose.position.y + dy)
        tcp_pose.pose.position.z = float(ee_pose.pose.position.z + dz)
        return tcp_pose

    @classmethod
    def _orientation_error_rad(cls, target_q: Quaternion, actual_q: Quaternion) -> float:
        tx, ty, tz, tw = cls._normalize_quaternion(target_q)
        ax, ay, az, aw = cls._normalize_quaternion(actual_q)
        dot = abs((tx * ax) + (ty * ay) + (tz * az) + (tw * aw))
        dot = max(-1.0, min(1.0, dot))
        return float(2.0 * math.acos(dot))

    @classmethod
    def _pose_error(
        cls,
        target_pose: PoseStamped,
        actual_pose: PoseStamped,
    ) -> tuple[float, float, float, float, float]:
        dx = float(target_pose.pose.position.x - actual_pose.pose.position.x)
        dy = float(target_pose.pose.position.y - actual_pose.pose.position.y)
        dz = float(target_pose.pose.position.z - actual_pose.pose.position.z)
        pos_err = float(math.sqrt((dx * dx) + (dy * dy) + (dz * dz)))
        ang_err = cls._orientation_error_rad(target_pose.pose.orientation, actual_pose.pose.orientation)
        return dx, dy, dz, pos_err, ang_err

    def _lookup_actual_tcp_pose(self) -> tuple[Optional[PoseStamped], str]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_link,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.cartesian_pose_tf_timeout),
            )
        except TransformException as exc:
            return None, f"TF lookup failed for {self.base_frame}<-{self.ee_link}: {exc}"

        ee_pose = PoseStamped()
        ee_pose.header = tf_msg.header
        ee_pose.pose.position = Point(
            x=float(tf_msg.transform.translation.x),
            y=float(tf_msg.transform.translation.y),
            z=float(tf_msg.transform.translation.z),
        )
        ee_pose.pose.orientation = Quaternion(
            x=float(tf_msg.transform.rotation.x),
            y=float(tf_msg.transform.rotation.y),
            z=float(tf_msg.transform.rotation.z),
            w=float(tf_msg.transform.rotation.w),
        )
        return self._tcp_pose_from_ee_pose(ee_pose, self.ee_tcp_offset), ""

    @staticmethod
    def _resolved_tolerance(requested: float, default_value: float) -> float:
        requested = float(requested)
        if requested > 0.0:
            return requested
        return float(default_value)

    def _pose_correction_progressed(
        self,
        *,
        previous_pos_err: float,
        current_pos_err: float,
    ) -> bool:
        return (float(previous_pos_err) - float(current_pos_err)) >= float(
            self.cartesian_pose_min_position_improvement
        )

    def _build_corrected_pose_target(
        self,
        *,
        current_target: PoseStamped,
        dx: float,
        dy: float,
        dz: float,
    ) -> PoseStamped:
        correction_norm = math.sqrt((dx * dx) + (dy * dy) + (dz * dz))
        if correction_norm > self.cartesian_pose_max_correction_step > 0.0:
            scale = float(self.cartesian_pose_max_correction_step / correction_norm)
            dx *= scale
            dy *= scale
            dz *= scale

        corrected = PoseStamped()
        corrected.header = current_target.header
        corrected.pose.orientation = current_target.pose.orientation
        corrected.pose.position.x = float(current_target.pose.position.x + dx)
        corrected.pose.position.y = float(current_target.pose.position.y + dy)
        corrected.pose.position.z = float(current_target.pose.position.z + dz)
        return corrected

    def handle_plan_execute_joint_goal(self, req, resp):
        self.get_logger().info(f"[plan_execute_joint_goal] execute={req.execute} joints={list(req.joints)}")

        if len(req.joints) != 7:
            resp.success = False
            resp.message = f"Expected 7 joints, got {len(req.joints)}"
            resp.planning_time = 0.0
            return resp

        err = self._wait_for_servers(timeout_sec=2.0)
        if err:
            resp.success = False
            resp.message = err
            resp.planning_time = 0.0
            return resp

        jt, t, msg = self._plan_joint_goal(list(req.joints))
        resp.planning_time = float(t)

        if jt is None:
            resp.success = False
            resp.message = msg
            return resp

        if req.execute:
            ok, emsg = self._execute_trajectory(jt)
            resp.success = bool(ok)
            resp.message = f"{msg}; {emsg}"
        else:
            resp.success = True
            resp.message = f"{msg}; execute=False"

        return resp

    def handle_plan_execute_pose_goal(self, req, resp):
        self.get_logger().info("[pose] entered callback")
        target = req.target
        if not target.header.frame_id:
            target.header.frame_id = self.base_frame
            
        self.get_logger().info(f"[pose] returning success={resp.success} msg={resp.message}")
        self.get_logger().info(
            f"[plan_execute_pose_goal] execute={req.execute} frame={target.header.frame_id} "
            f"pos=({target.pose.position.x:.3f},{target.pose.position.y:.3f},{target.pose.position.z:.3f})"
        )

        err = self._wait_for_servers(timeout_sec=2.0)
        if err:
            resp.success = False
            resp.message = err
            resp.planning_time = 0.0
            return resp

        joints_7, ik_msg = self._compute_ik(target)
        if joints_7 is None:
            resp.success = False
            resp.message = ik_msg
            resp.planning_time = 0.0
            return resp

        jt, t, plan_msg = self._plan_joint_goal(joints_7)
        resp.planning_time = float(t)

        if jt is None:
            resp.success = False
            resp.message = f"{ik_msg}; {plan_msg}"
            return resp

        if req.execute:
            target_pose = target
            last_msg = ""
            attempts_total = 1 + (
                self.cartesian_pose_correction_attempts if self.cartesian_pose_verification_enabled else 0
            )
            pos_tol = self._resolved_tolerance(
                req.position_tolerance, self.cartesian_pose_default_position_tolerance
            )
            ang_tol = self._resolved_tolerance(
                req.orientation_tolerance, self.cartesian_pose_default_orientation_tolerance
            )

            ok = False
            previous_pos_err: Optional[float] = None
            for attempt_idx in range(attempts_total):
                if attempt_idx > 0:
                    joints_7, ik_msg = self._compute_ik(target_pose)
                    if joints_7 is None:
                        resp.success = False
                        resp.message = f"{last_msg}; correction_ik_failed: {ik_msg}"
                        return resp
                    jt, t, plan_msg = self._plan_joint_goal(joints_7)
                    resp.planning_time += float(t)
                    if jt is None:
                        resp.success = False
                        resp.message = f"{last_msg}; correction_plan_failed: {plan_msg}"
                        return resp

                ok, emsg = self._execute_trajectory(jt)
                last_msg = f"{ik_msg}; {plan_msg}; {emsg}"
                if not ok:
                    break
                if not self.cartesian_pose_verification_enabled:
                    break
                if self.cartesian_pose_post_execute_settle_time > 0.0:
                    # Isaac can report success slightly before the arm visually settles.
                    # Give the simulator a brief moment to converge before sampling TF.
                    time.sleep(self.cartesian_pose_post_execute_settle_time)

                actual_pose, tf_msg = self._lookup_actual_tcp_pose()
                if actual_pose is None:
                    last_msg = f"{last_msg}; pose_verification_failed: {tf_msg}"
                    ok = False
                    break

                dx, dy, dz, pos_err, ang_err = self._pose_error(target_pose, actual_pose)
                self.get_logger().info(
                    "[pose_verify] "
                    f"target=({target_pose.pose.position.x:.3f},{target_pose.pose.position.y:.3f},{target_pose.pose.position.z:.3f}) "
                    f"actual=({actual_pose.pose.position.x:.3f},{actual_pose.pose.position.y:.3f},{actual_pose.pose.position.z:.3f}) "
                    f"delta=({dx:.3f},{dy:.3f},{dz:.3f}) pos_err={pos_err:.3f} ang_err={ang_err:.3f}"
                )
                if pos_err <= pos_tol and ang_err <= ang_tol:
                    last_msg = (
                        f"{last_msg}; pose_verified pos_err={pos_err:.3f}<=tol={pos_tol:.3f} "
                        f"ang_err={ang_err:.3f}<=tol={ang_tol:.3f}"
                    )
                    break

                if previous_pos_err is not None and not self._pose_correction_progressed(
                    previous_pos_err=previous_pos_err,
                    current_pos_err=pos_err,
                ):
                    last_msg = (
                        f"{last_msg}; pose_correction_stalled pos_err={pos_err:.3f} "
                        f"prev_pos_err={previous_pos_err:.3f} "
                        f"min_improvement={self.cartesian_pose_min_position_improvement:.3f}"
                    )
                    ok = False
                    break

                if attempt_idx >= (attempts_total - 1):
                    last_msg = (
                        f"{last_msg}; pose_miss pos_err={pos_err:.3f}>tol={pos_tol:.3f} "
                        f"ang_err={ang_err:.3f}>tol={ang_tol:.3f}"
                    )
                    ok = False
                    break

                previous_pos_err = pos_err

                target_pose = self._build_corrected_pose_target(
                    current_target=target_pose,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                )
                self.get_logger().warn(
                    "[pose_verify] applying Cartesian correction "
                    f"attempt={attempt_idx + 1}/{attempts_total - 1} "
                    f"next_target=({target_pose.pose.position.x:.3f},{target_pose.pose.position.y:.3f},{target_pose.pose.position.z:.3f})"
                )

            resp.success = bool(ok)
            resp.message = last_msg
        else:
            resp.success = True
            resp.message = f"{ik_msg}; {plan_msg}; execute=False"

        return resp


def main():
    rclpy.init()
    node = FR3SkillServer()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            executor.shutdown()
        except BaseException:
            pass
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
