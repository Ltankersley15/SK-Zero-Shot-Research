from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    use_sim_time = True

    publish_camera_tf = LaunchConfiguration("publish_camera_tf")
    publish_base_alias_tf = LaunchConfiguration("publish_base_alias_tf")
    publish_world_alias_tf = LaunchConfiguration("publish_world_alias_tf")
    mount_parent_link = LaunchConfiguration("mount_parent_link")
    mount_x = LaunchConfiguration("mount_x")
    mount_y = LaunchConfiguration("mount_y")
    mount_z = LaunchConfiguration("mount_z")
    mount_qx = LaunchConfiguration("mount_qx")
    mount_qy = LaunchConfiguration("mount_qy")
    mount_qz = LaunchConfiguration("mount_qz")
    mount_qw = LaunchConfiguration("mount_qw")
    depth_opt_qx = LaunchConfiguration("depth_opt_qx")
    depth_opt_qy = LaunchConfiguration("depth_opt_qy")
    depth_opt_qz = LaunchConfiguration("depth_opt_qz")
    depth_opt_qw = LaunchConfiguration("depth_opt_qw")
    color_opt_x = LaunchConfiguration("color_opt_x")
    color_opt_y = LaunchConfiguration("color_opt_y")
    color_opt_z = LaunchConfiguration("color_opt_z")
    color_opt_qx = LaunchConfiguration("color_opt_qx")
    color_opt_qy = LaunchConfiguration("color_opt_qy")
    color_opt_qz = LaunchConfiguration("color_opt_qz")
    color_opt_qw = LaunchConfiguration("color_opt_qw")
    enable_gripper_relay = LaunchConfiguration("enable_gripper_relay")
    fjt_action_name = LaunchConfiguration("fjt_action_name")
    gripper_traj_topic = LaunchConfiguration("gripper_traj_topic")
    gripper_command_topic = LaunchConfiguration("gripper_command_topic")
    gripper_joint_states_topic = LaunchConfiguration("gripper_joint_states_topic")
    gripper_joints = LaunchConfiguration("gripper_joints")
    enable_camera_info_repub = LaunchConfiguration("enable_camera_info_repub")
    fjt_rate_hz = LaunchConfiguration("fjt_rate_hz")
    fjt_extra_settle_time = LaunchConfiguration("fjt_extra_settle_time")
    fjt_min_traj_duration = LaunchConfiguration("fjt_min_traj_duration")
    fjt_goal_tolerance_rad = LaunchConfiguration("fjt_goal_tolerance_rad")
    skill_ee_link = LaunchConfiguration("skill_ee_link")
    skill_ee_tcp_offset_x = LaunchConfiguration("skill_ee_tcp_offset_x")
    skill_ee_tcp_offset_y = LaunchConfiguration("skill_ee_tcp_offset_y")
    skill_ee_tcp_offset_z = LaunchConfiguration("skill_ee_tcp_offset_z")
    skill_allowed_planning_time = LaunchConfiguration("skill_allowed_planning_time")
    skill_planning_attempts = LaunchConfiguration("skill_planning_attempts")
    skill_max_velocity_scaling = LaunchConfiguration("skill_max_velocity_scaling")
    skill_max_acceleration_scaling = LaunchConfiguration("skill_max_acceleration_scaling")
    skill_replan = LaunchConfiguration("skill_replan")
    skill_replan_attempts = LaunchConfiguration("skill_replan_attempts")
    skill_replan_delay = LaunchConfiguration("skill_replan_delay")
    skill_cartesian_pose_verification_enabled = LaunchConfiguration(
        "skill_cartesian_pose_verification_enabled"
    )
    skill_cartesian_pose_correction_attempts = LaunchConfiguration(
        "skill_cartesian_pose_correction_attempts"
    )
    skill_cartesian_pose_default_position_tolerance = LaunchConfiguration(
        "skill_cartesian_pose_default_position_tolerance"
    )
    skill_cartesian_pose_default_orientation_tolerance = LaunchConfiguration(
        "skill_cartesian_pose_default_orientation_tolerance"
    )
    skill_cartesian_pose_post_execute_settle_time = LaunchConfiguration(
        "skill_cartesian_pose_post_execute_settle_time"
    )
    skill_cartesian_pose_max_correction_step = LaunchConfiguration(
        "skill_cartesian_pose_max_correction_step"
    )
    skill_cartesian_pose_min_position_improvement = LaunchConfiguration(
        "skill_cartesian_pose_min_position_improvement"
    )

    declare_args = [
        DeclareLaunchArgument(
            "publish_camera_tf",
            default_value="true",
            description="Publish static TF frames for d455. Set false if Isaac already publishes them.",
        ),
        DeclareLaunchArgument(
            "publish_base_alias_tf",
            default_value="true",
            description="Publish static identity TF base->fr3_link0 to connect MoveIt planning frame.",
        ),
        DeclareLaunchArgument(
            "publish_world_alias_tf",
            default_value="true",
            description="Publish static identity TF world->fr3_link0 for convenience when no world frame exists.",
        ),
        DeclareLaunchArgument(
            "mount_parent_link",
            default_value="fr3_link8",
            description=(
                "Parent link to mount camera under. Use fr3_link8 by default so camera TF is "
                "connected even when MoveIt launches with load_gripper:=false. Use fr3_hand only "
                "if that link exists in your active robot model."
            ),
        ),
        DeclareLaunchArgument("mount_x", default_value="0.03549", description="d455 mount x (m)."),
        DeclareLaunchArgument("mount_y", default_value="0.00083", description="d455 mount y (m)."),
        DeclareLaunchArgument("mount_z", default_value="0.05476", description="d455 mount z (m)."),
        # Defaults from /fr3/fr3_hand/Realsense in fr3_test.usda.
        # When publishing directly under fr3_link8, the fixed fr3_link8 -> fr3_hand yaw
        # from the Franka model must be precomposed into the mount quaternion.
        # USDA stores quaternion as (w, x, y, z); ROS uses (x, y, z, w).
        # USDA Realsense: orient=(0.027091, 0.729604, -0.007906, 0.683287) [w, x, y, z] -> ROS: (x, y, z, w)
        # Effective fr3_link8 -> d455_link after applying the FR3 hand joint yaw:
        # q=(0.67104073, -0.28651179, 0.62090766, 0.28651180)
        DeclareLaunchArgument("mount_qx", default_value="0.67104073", description="d455 mount qx."),
        DeclareLaunchArgument("mount_qy", default_value="-0.28651179", description="d455 mount qy."),
        DeclareLaunchArgument("mount_qz", default_value="0.62090766", description="d455 mount qz."),
        DeclareLaunchArgument("mount_qw", default_value="0.2865118", description="d455 mount qw."),
        # Depth camera local orientation from Camera_Pseudo_Depth in fr3_test.usda.
        # The (0.5, 0.5, 0.5, 0.5) quaternion is the standard ROS optical frame rotation:
        #   - Converts from sensor frame (X-forward, Y-left, Z-up) to optical frame (X-right, Y-down, Z-forward)
        #   - This matches the REP-103 camera optical frame convention
        # IMPORTANT: If you experience Y-coordinate sign flips in object detection, verify this
        # quaternion is correctly applied in your TF tree. The active zero-shot agent uses this rotation
        # when use_hand_reference=True to transform depth points from optical frame to hand frame.
        DeclareLaunchArgument("depth_opt_qx", default_value="0.5", description="depth optical qx."),
        DeclareLaunchArgument("depth_opt_qy", default_value="0.5", description="depth optical qy."),
        DeclareLaunchArgument("depth_opt_qz", default_value="0.5", description="depth optical qz."),
        DeclareLaunchArgument("depth_opt_qw", default_value="0.5", description="depth optical qw."),
        # Isaac's rsd455.usd composes the color camera pose through an intermediate RSD455 body
        # transform before the camera's local optical rotation. Collapsed into a single
        # d455_link -> d455_color_optical_frame ROS TF, the effective pose retains the previous
        # optical quaternion and adds the missing +11.5 mm Y offset.
        DeclareLaunchArgument("color_opt_x", default_value="0.0", description="color optical x (m)."),
        DeclareLaunchArgument("color_opt_y", default_value="0.0115", description="color optical y (m)."),
        DeclareLaunchArgument("color_opt_z", default_value="0.0", description="color optical z (m)."),
        DeclareLaunchArgument("color_opt_qx", default_value="-0.5", description="color optical qx."),
        DeclareLaunchArgument("color_opt_qy", default_value="0.5", description="color optical qy."),
        DeclareLaunchArgument("color_opt_qz", default_value="-0.5", description="color optical qz."),
        DeclareLaunchArgument("color_opt_qw", default_value="0.5", description="color optical qw."),
        DeclareLaunchArgument(
            "enable_gripper_relay",
            default_value="true",
            description="Run gripper relay to convert JointTrajectory -> JointState commands.",
        ),
        DeclareLaunchArgument(
            "fjt_action_name",
            default_value="/fr3_arm_controller/follow_joint_trajectory_relay",
            description=(
                "FJT action name exposed by fjt_relay and used by fr3_skill_server. "
                "Use a dedicated relay name to avoid duplicate action-server collisions."
            ),
        ),
        DeclareLaunchArgument(
            "gripper_traj_topic",
            default_value="/fr3_gripper_controller/joint_trajectory",
            description="Input JointTrajectory topic for gripper.",
        ),
        DeclareLaunchArgument(
            "gripper_command_topic",
            default_value="/fr3_arm_controller/joint_command",
            description="Output JointState command topic for gripper (set to gripper command topic if needed).",
        ),
        DeclareLaunchArgument(
            "gripper_joint_states_topic",
            default_value="/joint_states",
            description="JointState source for merging gripper commands with current arm joints.",
        ),
        DeclareLaunchArgument(
            "gripper_joints",
            default_value="['fr3_finger_joint1','fr3_finger_joint2']",
            description="Gripper joint names (override if Isaac uses panda_finger_joint*).",
        ),
        DeclareLaunchArgument(
            "enable_camera_info_repub",
            default_value="true",
            description="Enable only if Isaac does not publish /fr3/d455/*/camera_info topics.",
        ),
        DeclareLaunchArgument(
            "fjt_rate_hz",
            default_value="120.0",
            description="FollowJointTrajectory relay publish rate.",
        ),
        DeclareLaunchArgument(
            "fjt_extra_settle_time",
            default_value="0.8",
            description="Extra settle time after the final relay waypoint (s).",
        ),
        DeclareLaunchArgument(
            "fjt_min_traj_duration",
            default_value="0.4",
            description="Minimum relay trajectory duration after retiming (s).",
        ),
        DeclareLaunchArgument(
            "fjt_goal_tolerance_rad",
            default_value="0.08",
            description=(
                "Per-joint completion tolerance used by the FJT relay before it reports success. "
                "Tighter values improve Cartesian accuracy but may require longer settle time."
            ),
        ),
        DeclareLaunchArgument(
            "skill_ee_link",
            default_value="fr3_link8",
            description="IK end-effector link used by fr3_skill_server (use fr3_hand_tcp only if present in MoveIt model).",
        ),
        DeclareLaunchArgument(
            "skill_ee_tcp_offset_x",
            default_value="0.0",
            description="TCP offset x (m) in ee_link frame for IK compensation.",
        ),
        DeclareLaunchArgument(
            "skill_ee_tcp_offset_y",
            default_value="0.0",
            description="TCP offset y (m) in ee_link frame for IK compensation.",
        ),
        DeclareLaunchArgument(
            "skill_ee_tcp_offset_z",
            default_value="0.10",
            description=(
                "TCP offset z (m) in ee_link frame for IK compensation. "
                "For FR3 in Isaac with ee_link=fr3_link8, ~0.10m better matches the simulated fingertip contact point."
            ),
        ),
        DeclareLaunchArgument(
            "skill_allowed_planning_time",
            default_value="3.0",
            description="MoveIt allowed planning time for skill_server (s).",
        ),
        DeclareLaunchArgument(
            "skill_planning_attempts",
            default_value="3",
            description="MoveIt planning attempts for skill_server.",
        ),
        DeclareLaunchArgument(
            "skill_max_velocity_scaling",
            default_value="0.7",
            description="MoveIt max velocity scaling for skill_server.",
        ),
        DeclareLaunchArgument(
            "skill_max_acceleration_scaling",
            default_value="0.7",
            description="MoveIt max acceleration scaling for skill_server.",
        ),
        DeclareLaunchArgument(
            "skill_replan",
            default_value="false",
            description="Enable MoveIt replanning in skill_server.",
        ),
        DeclareLaunchArgument(
            "skill_replan_attempts",
            default_value="1",
            description="MoveIt replanning attempts in skill_server.",
        ),
        DeclareLaunchArgument(
            "skill_replan_delay",
            default_value="0.1",
            description="MoveIt replanning delay in skill_server.",
        ),
        DeclareLaunchArgument(
            "skill_cartesian_pose_verification_enabled",
            default_value="true",
            description="Enable post-execution TCP pose verification in skill_server.",
        ),
        DeclareLaunchArgument(
            "skill_cartesian_pose_correction_attempts",
            default_value="1",
            description="Number of Cartesian correction retries after pose verification misses.",
        ),
        DeclareLaunchArgument(
            "skill_cartesian_pose_default_position_tolerance",
            default_value="0.02",
            description="Default Cartesian position tolerance (m) used when callers do not provide one.",
        ),
        DeclareLaunchArgument(
            "skill_cartesian_pose_default_orientation_tolerance",
            default_value="0.35",
            description="Default Cartesian orientation tolerance (rad) used when callers do not provide one.",
        ),
        DeclareLaunchArgument(
            "skill_cartesian_pose_post_execute_settle_time",
            default_value="0.5",
            description="Extra wait (s) before skill_server samples TF after execution for pose verification.",
        ),
        DeclareLaunchArgument(
            "skill_cartesian_pose_max_correction_step",
            default_value="0.06",
            description="Maximum Cartesian correction step size (m) applied after verification misses.",
        ),
        DeclareLaunchArgument(
            "skill_cartesian_pose_min_position_improvement",
            default_value="0.005",
            description="Minimum Cartesian position improvement (m) required to continue correction retries.",
        ),
    ]

    # ---- MoveIt move_group.launch.py from franka_fr3_moveit_config ----
    moveit_pkg = "franka_fr3_moveit_config"
    moveit_share = get_package_share_directory(moveit_pkg)
    move_group_launch = os.path.join(moveit_share, "launch", "move_group.launch.py")

    # ---- Joint State Filter (ONE instance only) ----
    # Reads Isaac /joint_states (arm+fingers), publishes /fr3/joint_states_arm (arm-only)
    js_filter = Node(
        package="fr3_skill_server",
        executable="joint_state_filter",
        name="fr3_joint_state_filter",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"in_topic": "/joint_states"},
            {"out_topic": "/fr3/joint_states_arm"},
        ],
    )

    # =============================================================================
    # CAMERA TF (optional)
    #
    # IMPORTANT:
    # - Only enable this if Isaac is NOT already publishing these camera frames.
    # - Duplicated TF (same child frame from two publishers) can cause TF weirdness.
    #
    # Mount: <mount_parent_link> -> d455_link
    # =============================================================================
    base_alias_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_alias_tf_pub",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "1", "base", "fr3_link0"],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(publish_base_alias_tf),
    )

    world_alias_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_alias_tf_pub",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "1", "world", "base"],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(publish_world_alias_tf),
    )

    d455_mount_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="d455_mount_tf_pub",
        output="screen",
        arguments=[
            mount_x, mount_y, mount_z,
            mount_qx, mount_qy, mount_qz, mount_qw,
            mount_parent_link, "d455_link",
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(publish_camera_tf),
    )

    # Depth frame: d455_link -> d455_depth_frame
    d455_depth_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="d455_depth_tf_pub",
        output="screen",
        arguments=[
            "0", "0", "0",
            "0", "0", "0", "1",
            "d455_link", "d455_depth_frame",
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(publish_camera_tf),
    )

    # Depth optical: d455_depth_frame -> d455_depth_optical_frame
    d455_depth_optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="d455_depth_optical_tf_pub",
        output="screen",
        arguments=[
            "0", "0", "0",
            depth_opt_qx, depth_opt_qy, depth_opt_qz, depth_opt_qw,
            "d455_depth_frame", "d455_depth_optical_frame",
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(publish_camera_tf),
    )

    # Color optical: d455_link -> d455_color_optical_frame
    d455_color_optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="d455_color_optical_tf_pub",
        output="screen",
        arguments=[
            color_opt_x, color_opt_y, color_opt_z,
            color_opt_qx, color_opt_qy, color_opt_qz, color_opt_qw,
            "d455_link", "d455_color_optical_frame",
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(publish_camera_tf),
    )

    cam_info_repub = Node(
        package="fr3_lvlm_agent",
        executable="camera_info_repub",
        name="camera_info_repub",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(enable_camera_info_repub),
    )

    # ---- MoveIt group ----
    moveit_group = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(move_group_launch),
            launch_arguments={
                "robot_ip": "192.168.131.40",
                "namespace": "fr3",
                "load_gripper": "false",
                "use_fake_hardware": "false",
                "fake_sensor_commands": "false",
                "use_sim_time": "true",
            }.items(),
        ),
    ])

    # ---- FJT relay ----
    # IMPORTANT: point relay's "measured joint_states_topic" to the filtered ARM-only topic
    # so tolerance checks don’t get confused by finger joints or missing /joint_states
    relay = Node(
        package="fr3_moveit_relay",
        executable="fjt_relay",
        name="fjt_relay",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"action_name": fjt_action_name},
            {"use_header_stamp": False},
            {"joint_states_topic": "/fr3/joint_states_arm"},
            {"rate_hz": fjt_rate_hz},
            {"extra_settle_time": fjt_extra_settle_time},
            {"min_traj_duration": fjt_min_traj_duration},
            {"goal_tolerance_rad": fjt_goal_tolerance_rad},
        ],
    )

    # ---- Gripper relay ----
    gripper_relay = Node(
        package="fr3_moveit_relay",
        executable="gripper_relay",
        name="gripper_relay",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"traj_topic": gripper_traj_topic},
            {"command_topic": gripper_command_topic},
            {"joint_states_topic": gripper_joint_states_topic},
            {"gripper_joints": gripper_joints},
        ],
        condition=IfCondition(enable_gripper_relay),
    )

    # ---- Skill server ----
    skill_server = Node(
        package="fr3_skill_server",
        executable="fr3_skill_server",
        name="fr3_skill_server",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"fjt_action_name": fjt_action_name},
            {"ee_link": skill_ee_link},
            {"ee_tcp_offset_x": skill_ee_tcp_offset_x},
            {"ee_tcp_offset_y": skill_ee_tcp_offset_y},
            {"ee_tcp_offset_z": skill_ee_tcp_offset_z},
            {"allowed_planning_time": skill_allowed_planning_time},
            {"planning_attempts": skill_planning_attempts},
            {"max_velocity_scaling_factor": skill_max_velocity_scaling},
            {"max_acceleration_scaling_factor": skill_max_acceleration_scaling},
            {"replan": skill_replan},
            {"replan_attempts": skill_replan_attempts},
            {"replan_delay": skill_replan_delay},
            {"cartesian_pose_verification_enabled": skill_cartesian_pose_verification_enabled},
            {"cartesian_pose_correction_attempts": skill_cartesian_pose_correction_attempts},
            {
                "cartesian_pose_default_position_tolerance": skill_cartesian_pose_default_position_tolerance
            },
            {
                "cartesian_pose_default_orientation_tolerance": skill_cartesian_pose_default_orientation_tolerance
            },
            {"cartesian_pose_post_execute_settle_time": skill_cartesian_pose_post_execute_settle_time},
            {"cartesian_pose_max_correction_step": skill_cartesian_pose_max_correction_step},
            {"cartesian_pose_min_position_improvement": skill_cartesian_pose_min_position_improvement},
        ],
        remappings=[
            ("joint_states", "/fr3/joint_states_arm"),
        ],
    )

    # Build camera TF list conditionally (simple: include or don’t include)
    camera_tf_nodes = [
        base_alias_tf,
        world_alias_tf,
        d455_mount_tf,
        d455_depth_tf,
        d455_depth_optical_tf,
        d455_color_optical_tf,
    ]

    # NOTE: Launch "condition" for Nodes is more annoying than it should be;
    # simplest: user sets publish_camera_tf and you comment out the list if needed.
    # But we’ll still wire it in a readable way:
    ld_entities = []
    ld_entities += declare_args

    # If you set publish_camera_tf:=false, COMMENT OUT these 4 nodes in the return list below.
    ld_entities += camera_tf_nodes

    ld_entities += [
        cam_info_repub,
        js_filter,
        moveit_group,
        relay,
        gripper_relay,
        skill_server,
    ]

    return LaunchDescription(ld_entities)
