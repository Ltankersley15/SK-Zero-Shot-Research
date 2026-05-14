from __future__ import annotations

import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions


def _load_launch_module():
    path = Path(__file__).resolve().parents[1] / "launch" / "fr3_isaac_moveit_bringup.launch.py"
    spec = importlib.util.spec_from_file_location("fr3_isaac_moveit_bringup_launch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _launch_defaults() -> dict[str, str]:
    module = _load_launch_module()
    launch_description = module.generate_launch_description()
    context = LaunchContext()
    defaults = {}
    for entity in launch_description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            defaults[entity.name] = perform_substitutions(context, entity.default_value)
    return defaults


def test_color_camera_tf_defaults_match_isaac_rsd455_asset() -> None:
    defaults = _launch_defaults()

    assert defaults["color_opt_x"] == "0.0"
    assert defaults["color_opt_y"] == "0.0115"
    assert defaults["color_opt_z"] == "0.0"
    assert defaults["color_opt_qx"] == "-0.5"
    assert defaults["color_opt_qy"] == "0.5"
    assert defaults["color_opt_qz"] == "-0.5"
    assert defaults["color_opt_qw"] == "0.5"


def test_mount_defaults_include_fr3_hand_yaw_when_parent_is_link8() -> None:
    defaults = _launch_defaults()

    assert defaults["mount_parent_link"] == "fr3_link8"
    assert defaults["mount_qx"] == "0.67104073"
    assert defaults["mount_qy"] == "-0.28651179"
    assert defaults["mount_qz"] == "0.62090766"
    assert defaults["mount_qw"] == "0.2865118"


def test_skill_tcp_z_default_matches_isaac_gripper_height_calibration() -> None:
    defaults = _launch_defaults()

    assert defaults["skill_ee_tcp_offset_z"] == "0.10"
