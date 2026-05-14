#ws_moveit2/src/fr3_moveit_relay/fr3_moveit_relay/fjt_relay.py
#!/usr/bin/env python3
from typing import Dict, List, Optional

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def dur_to_sec(d: DurationMsg) -> float:
    return float(d.sec) + float(d.nanosec) * 1e-9


def sec_to_dur(t: float) -> DurationMsg:
    t = max(0.0, float(t))
    sec = int(t)
    nanosec = int((t - sec) * 1e9)
    out = DurationMsg()
    out.sec = sec
    out.nanosec = nanosec
    return out


class FollowJointTrajectoryRelay(Node):
    """
    FollowJointTrajectory -> publishes JointState targets to Isaac.

    Fixes:
      - Ignore header stamp by default (sim vs wall mismatch).
      - If MoveIt returns a trajectory with absurdly small total time (e.g. 0.05s),
        auto-stretch it so the sim can actually follow and joint_states can converge.
    """

    def __init__(self):
        super().__init__("fjt_relay")

        # Parameters
        self.declare_parameter("action_name", "/fr3_arm_controller/follow_joint_trajectory")
        self.declare_parameter("command_topic", "/fr3_arm_controller/joint_command")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("enable_gripper_latch", True)
        self.declare_parameter("gripper_traj_topic", "/fr3_gripper_controller/joint_trajectory")
        self.declare_parameter("gripper_joints", ["fr3_finger_joint1", "fr3_finger_joint2"])

        self.declare_parameter("goal_tolerance_rad", 0.08)   # slightly looser than 0.05
        self.declare_parameter("extra_settle_time", 2.0)     # allow convergence after last point
        self.declare_parameter("max_goal_time", 30.0)        # hard ceiling
        self.declare_parameter("use_header_stamp", False)
        self.declare_parameter("stamp_accept_window", 2.0)

        # NEW: retiming
        self.declare_parameter("min_traj_duration", 0.6)     # stretch tiny plans to at least this
        self.declare_parameter("min_final_time_for_scaling", 0.15)  # if final_time < this, scale
        self.declare_parameter("interpolate_trajectory", True)

        self.action_name = str(self.get_parameter("action_name").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.enable_gripper_latch = bool(self.get_parameter("enable_gripper_latch").value)
        self.gripper_traj_topic = str(self.get_parameter("gripper_traj_topic").value)
        self.gripper_joints = [str(v) for v in self.get_parameter("gripper_joints").value]

        self.goal_tol = float(self.get_parameter("goal_tolerance_rad").value)
        self.extra_settle_time = float(self.get_parameter("extra_settle_time").value)
        self.max_goal_time = float(self.get_parameter("max_goal_time").value)
        self.use_header_stamp = bool(self.get_parameter("use_header_stamp").value)
        self.stamp_accept_window = float(self.get_parameter("stamp_accept_window").value)

        self.min_traj_duration = float(self.get_parameter("min_traj_duration").value)
        self.min_final_time_for_scaling = float(self.get_parameter("min_final_time_for_scaling").value)
        self.interpolate_trajectory = bool(self.get_parameter("interpolate_trajectory").value)

        # Latest measured joint positions
        self._measured: Dict[str, float] = {}
        self._latched_gripper_targets: Dict[str, float] = {}

        # Pub/Sub
        self.cmd_pub = self.create_publisher(JointState, self.command_topic, 10)
        self.js_sub = self.create_subscription(JointState, self.joint_states_topic, self._on_joint_states, 50)
        self.gripper_sub = self.create_subscription(
            JointTrajectory,
            self.gripper_traj_topic,
            self._on_gripper_traj,
            10,
        )

        # Action server
        self.cb_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            self.action_name,
            execute_callback=self._execute_cb,
            callback_group=self.cb_group,
        )

        self.get_logger().info(f"Relay ready. Action: {self.action_name} -> Topic: {self.command_topic}")
        self.get_logger().info(
            f"use_header_stamp={self.use_header_stamp} rate_hz={self.rate_hz:.1f} "
            f"tol={self.goal_tol:.3f} settle={self.extra_settle_time:.1f}s "
            f"min_traj_duration={self.min_traj_duration:.2f}s"
        )
        self.get_logger().info(f"interpolate_trajectory={self.interpolate_trajectory}")
        self.get_logger().info(
            f"gripper_latch={self.enable_gripper_latch} gripper_traj_topic={self.gripper_traj_topic} "
            f"gripper_joints={self.gripper_joints}"
        )

    def _on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._measured[name] = float(pos)

    def _on_gripper_traj(self, msg: JointTrajectory):
        if not self.enable_gripper_latch or not msg.points:
            return
        joint_names = list(msg.joint_names) if msg.joint_names else list(self.gripper_joints)
        positions = list(msg.points[-1].positions)
        if len(joint_names) != len(positions):
            self.get_logger().warn("Ignoring gripper trajectory with mismatched joint_names and positions lengths.")
            return
        self._latched_gripper_targets = {
            str(name): float(pos)
            for name, pos in zip(joint_names, positions)
        }

    @staticmethod
    def _validate_trajectory(traj: JointTrajectory) -> Optional[str]:
        if not traj.joint_names:
            return "Trajectory has no joint_names."
        if not traj.points:
            return "Trajectory has no points."
        for i, p in enumerate(traj.points):
            if len(p.positions) != len(traj.joint_names):
                return f"Point {i} positions length != joint_names length."
        last_t = -1.0
        for i, p in enumerate(traj.points):
            ti = dur_to_sec(p.time_from_start)
            if ti < last_t - 1e-9:
                return f"Point {i} time_from_start is not nondecreasing."
            last_t = ti
        return None

    def _publish_point(self, joint_names: List[str], positions: List[float]):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joint_names)
        msg.position = [float(x) for x in positions]
        if self.enable_gripper_latch and self._latched_gripper_targets:
            seen = set(msg.name)
            for name in self.gripper_joints:
                if name in self._latched_gripper_targets and name not in seen:
                    msg.name.append(name)
                    msg.position.append(float(self._latched_gripper_targets[name]))
                    seen.add(name)
        self.cmd_pub.publish(msg)

    def _within_tolerance(self, joint_names: List[str], target: List[float]) -> bool:
        if any(name not in self._measured for name in joint_names):
            return False
        for name, tgt in zip(joint_names, target):
            if abs(self._measured[name] - float(tgt)) > self.goal_tol:
                return False
        return True

    def _joint_error_snapshot(self, joint_names: List[str], target: List[float]) -> tuple[str, float | None]:
        missing = [name for name in joint_names if name not in self._measured]
        if missing:
            return f"missing measured joints: {', '.join(missing)}", None

        rows = []
        max_abs_err = 0.0
        for name, tgt in zip(joint_names, target):
            measured = float(self._measured[name])
            err = float(measured - float(tgt))
            max_abs_err = max(max_abs_err, abs(err))
            rows.append(
                f"{name}=target:{float(tgt):+.4f} measured:{measured:+.4f} err:{err:+.4f}"
            )
        return "; ".join(rows), max_abs_err

    @staticmethod
    def _point_positions(point: JointTrajectoryPoint) -> List[float]:
        return [float(v) for v in point.positions]

    def _desired_positions_at_time(
        self,
        points: List[JointTrajectoryPoint],
        t: float,
    ) -> tuple[List[float], int]:
        # Isaac tracks joint targets more reliably when we stream a smooth target rather than
        # holding a sparse waypoint until the next timestamp. This mirrors Bennett's realtime
        # UR10e control style: keep nudging the robot toward the in-between target.
        if not points:
            return [], 0
        if len(points) == 1:
            return self._point_positions(points[0]), 0

        if t <= dur_to_sec(points[0].time_from_start):
            return self._point_positions(points[0]), 0

        for i in range(len(points) - 1):
            t0 = dur_to_sec(points[i].time_from_start)
            t1 = dur_to_sec(points[i + 1].time_from_start)
            if t <= t1:
                if not self.interpolate_trajectory or t1 <= t0 + 1e-9:
                    return self._point_positions(points[i]), i
                alpha = max(0.0, min(1.0, float((t - t0) / (t1 - t0))))
                start = self._point_positions(points[i])
                end = self._point_positions(points[i + 1])
                interp = [
                    float(s + alpha * (e - s))
                    for s, e in zip(start, end)
                ]
                return interp, i

        return self._point_positions(points[-1]), len(points) - 1

    def _choose_start_time(self, traj: JointTrajectory):
        now = self.get_clock().now()

        if not self.use_header_stamp:
            return now, "start=NOW (header stamp ignored)"

        if traj.header.stamp.sec == 0 and traj.header.stamp.nanosec == 0:
            return now, "start=NOW (header stamp was zero)"

        stamp = rclpy.time.Time.from_msg(traj.header.stamp)
        dt = abs((now - stamp).nanoseconds) * 1e-9
        if dt > self.stamp_accept_window:
            return now, f"start=NOW (header stamp rejected; |now-stamp|={dt:.3f}s)"
        return stamp, f"start=HEADER_STAMP (accepted; |now-stamp|={dt:.3f}s)"

    def _maybe_retime_points(self, points: List[JointTrajectoryPoint]) -> (List[JointTrajectoryPoint], float, str):
        """If final_time is extremely small, scale all time_from_start to a safer duration."""
        final_time = dur_to_sec(points[-1].time_from_start)
        if final_time <= 0.0:
            # Degenerate: make evenly spaced times over min_traj_duration
            n = len(points)
            if n == 1:
                points[0].time_from_start = sec_to_dur(self.min_traj_duration)
            else:
                dt = self.min_traj_duration / float(n - 1)
                for i, p in enumerate(points):
                    p.time_from_start = sec_to_dur(i * dt)
            return points, dur_to_sec(points[-1].time_from_start), "retime=DEGENERATE->MIN_DURATION"

        if final_time < self.min_final_time_for_scaling:
            scale = self.min_traj_duration / final_time if final_time > 1e-6 else 1.0
            for p in points:
                t = dur_to_sec(p.time_from_start)
                p.time_from_start = sec_to_dur(t * scale)
            return points, dur_to_sec(points[-1].time_from_start), f"retime=SCALED x{scale:.2f}"

        return points, final_time, "retime=NONE"

    def _execute_cb(self, goal_handle: ServerGoalHandle) -> FollowJointTrajectory.Result:
        traj = goal_handle.request.trajectory
        err = self._validate_trajectory(traj)
        if err:
            self.get_logger().error(err)
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = err
            return result

        joint_names = list(traj.joint_names)
        points: List[JointTrajectoryPoint] = list(traj.points)
        final_positions = list(points[-1].positions)

        # NEW: protect against absurdly fast trajectories
        pre_retime_final_time = dur_to_sec(points[-1].time_from_start)
        points, final_time, retime_msg = self._maybe_retime_points(points)
        if pre_retime_final_time > 1e-6:
            retime_scale = final_time / pre_retime_final_time
            if retime_scale > 5.0:
                self.get_logger().warn(
                    "Large relay retime scale: "
                    f"original_final_time={pre_retime_final_time:.3f}s "
                    f"scaled_final_time={final_time:.3f}s "
                    f"scale={retime_scale:.2f}"
                )

        start, start_msg = self._choose_start_time(traj)
        settle_deadline = final_time + max(0.0, self.extra_settle_time)
        hard_deadline = min(settle_deadline, self.max_goal_time)

        dt = 1.0 / max(self.rate_hz, 1.0)
        self.get_logger().info(
            f"Executing trajectory with {len(points)} points at {self.rate_hz:.1f} Hz "
            f"(final_time={final_time:.3f}s, hard_deadline={hard_deadline:.3f}s) | {start_msg} | {retime_msg}"
        )

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = "Goal canceled."
                return result

            now = self.get_clock().now()
            t = (now - start).nanoseconds * 1e-9

            if t < 0.0:
                time.sleep(dt)
                continue

            desired_positions, _point_idx = self._desired_positions_at_time(points, t)
            self._publish_point(joint_names, desired_positions)

            fb = FollowJointTrajectory.Feedback()
            fb.joint_names = joint_names
            fb.desired.positions = list(desired_positions)
            goal_handle.publish_feedback(fb)

            if t >= final_time:
                if self._within_tolerance(joint_names, final_positions):
                    goal_handle.succeed()
                    result = FollowJointTrajectory.Result()
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    result.error_string = "Executed."
                    return result

                if t >= hard_deadline:
                    snapshot, max_abs_err = self._joint_error_snapshot(joint_names, final_positions)
                    self.get_logger().error(
                        f"Timeout: reached t={t:.3f}s but never hit tolerance (tol={self.goal_tol}). Aborting."
                    )
                    self.get_logger().error(
                        "Final joint residuals at timeout: "
                        f"{snapshot}"
                    )
                    if max_abs_err is not None:
                        self.get_logger().error(
                            f"Max final joint abs error at timeout: {max_abs_err:.4f} rad"
                        )
                    goal_handle.abort()
                    result = FollowJointTrajectory.Result()
                    result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    result.error_string = (
                        "Timeout waiting to reach final joint tolerance."
                        if max_abs_err is None
                        else f"Timeout waiting to reach final joint tolerance (max_abs_err={max_abs_err:.4f} rad)."
                    )
                    return result

            time.sleep(dt)

        goal_handle.abort()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
        result.error_string = "ROS shutdown during execution."
        return result


def main():
    rclpy.init()
    node = FollowJointTrajectoryRelay()
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
