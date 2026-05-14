#ws_moveit2/src/franka_fr3_moveit_config/launch/move_group.launch.py 
import os
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def generate_launch_description():
    # ---- Launch args ----
    robot_ip_parameter_name = "robot_ip"
    load_gripper_parameter_name = "load_gripper"
    use_fake_hardware_parameter_name = "use_fake_hardware"
    fake_sensor_commands_parameter_name = "fake_sensor_commands"
    namespace_parameter_name = "namespace"
    use_sim_time_parameter_name = "use_sim_time"

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    namespace = LaunchConfiguration(namespace_parameter_name)
    use_sim_time = LaunchConfiguration(use_sim_time_parameter_name)

    declare_args = [
        DeclareLaunchArgument(
            namespace_parameter_name,
            default_value="fr3",
            description="Namespace for the robot.",
        ),
        DeclareLaunchArgument(
            load_gripper_parameter_name,
            default_value="false",
            description="Whether to load the gripper (true/false).",
        ),
        DeclareLaunchArgument(
            use_sim_time_parameter_name,
            default_value="true",
            description="Use simulation time.",
        ),
        DeclareLaunchArgument(
            robot_ip_parameter_name,
            default_value="127.0.0.1",
            description="Hostname or IP address of the robot (dummy for simulation).",
        ),
        DeclareLaunchArgument(
            use_fake_hardware_parameter_name,
            default_value="false",
            description="Use fake hardware (true/false).",
        ),
        DeclareLaunchArgument(
            fake_sensor_commands_parameter_name,
            default_value="false",
            description="Fake sensor commands (only if use_fake_hardware is true).",
        ),
    ]

    # ---- Robot description (URDF) ----
    franka_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots",
        "fr3",
        "fr3.urdf.xacro",
    )

    robot_description_command = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            franka_xacro_file,
            " ros2_control:=false",
            " hand:=",
            load_gripper,
            " robot_type:=fr3",
            " robot_ip:=",
            robot_ip,
            " use_fake_hardware:=",
            use_fake_hardware,
            " fake_sensor_commands:=",
            fake_sensor_commands,
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(robot_description_command, value_type=str)
    }

    # ---- Semantic description (SRDF) ----
    franka_semantic_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots",
        "fr3",
        "fr3.srdf.xacro",
    )

    robot_description_semantic_command = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            franka_semantic_xacro_file,
            " hand:=",
            load_gripper,
        ]
    )

    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            robot_description_semantic_command, value_type=str
        )
    }

    # ---- Kinematics ----
    kinematics_yaml = load_yaml("franka_fr3_moveit_config", "config/kinematics.yaml") or {}

    # ---- OMPL planning config ----
    ompl_yaml = load_yaml("franka_fr3_moveit_config", "config/ompl_planning.yaml") or {}

    # ---- Controllers config ----
    controllers_yaml = load_yaml(
        "franka_fr3_moveit_config", "config/move_group_controllers_params.yaml"
    ) or {}

    # ---- CRITICAL topic choice ----
    # Always consume arm-only joint states (filtered) in this stack.
    # Your filter publishes: /fr3/joint_states_arm
    JOINT_STATES_TOPIC = "/fr3/joint_states_arm"

    # ---- robot_state_publisher (TF) ----
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=namespace,
        output="screen",
        parameters=[
            robot_description,
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            ("joint_states", JOINT_STATES_TOPIC),
        ],
    )

    # ---- move_group ----
    # Force OMPL selection explicitly (prevents MoveIt from choosing CHOMP “by accident”)
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace=namespace,
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            ompl_yaml,
            controllers_yaml,
            {"use_sim_time": use_sim_time},

            # Hard-force OMPL (prevents: "Multiple planning plugins ... Using CHOMPPlanner")
            {"planning_plugin": "ompl_interface/OMPLPlanner"},
            {"default_planning_pipeline": "ompl"},
            {"planning_pipelines": ["ompl", "chomp"]},

            {"moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager"},
        ],
        remappings=[
            ("joint_states", JOINT_STATES_TOPIC),
        ],
    )

    return LaunchDescription(
        declare_args + [
            robot_state_publisher,
            run_move_group_node,
        ]
    )

