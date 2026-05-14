from __future__ import annotations

import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions


def _load_launch_module():
    path = Path(__file__).resolve().parents[1] / "launch" / "fr3_isaac_moveit_zero_shot_agent.launch.py"
    spec = importlib.util.spec_from_file_location("fr3_isaac_moveit_agent_launch", path)
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


def test_agent_launch_uses_balanced_motion_backend_defaults() -> None:
    defaults = _launch_defaults()

    assert defaults["skill_max_velocity_scaling"] == "0.10"
    assert defaults["skill_max_acceleration_scaling"] == "0.08"
    assert defaults["fjt_extra_settle_time"] == "1.0"
    assert defaults["fjt_min_traj_duration"] == "0.8"
    assert defaults["fjt_goal_tolerance_rad"] == "0.05"
    assert defaults["skill_cartesian_pose_post_execute_settle_time"] == "0.5"
    assert defaults["embodiment_gate_enabled"] == "true"
    assert defaults["embodiment_gate_xy_tolerance"] == "0.025"
    assert defaults["embodiment_gate_z_tolerance"] == "0.020"
    assert defaults["embodiment_gate_orientation_tolerance"] == "0.22"


def test_agent_launch_extends_service_wait_budget_for_verified_moves() -> None:
    defaults = _launch_defaults()

    assert defaults["pose_service_timeout_sec"] == "45.0"
    assert defaults["joint_service_timeout_sec"] == "45.0"
    assert defaults["runtime_stream_gap_reset_sec"] == "1.5"
    assert defaults["runtime_stream_resume_settle_sec"] == "3.0"


def test_agent_launch_uses_container_aware_place_defaults() -> None:
    defaults = _launch_defaults()

    assert defaults["place_hover_height"] == "0.10"
    assert defaults["place_on_surface_offset"] == "0.05"
    assert defaults["place_side_offset"] == "0.07"
    assert defaults["place_in_release_above_rim"] == "0.06"
    assert defaults["place_in_min_target_z_offset"] == "0.14"
    assert defaults["place_in_hover_above_rim"] == "0.16"


def test_agent_launch_stages_through_ready_and_observation_views() -> None:
    defaults = _launch_defaults()

    assert defaults["observe_before_grounding"] == "true"
    assert defaults["ready_before_observation"] == "true"


def test_agent_launch_uses_deeper_and_stricter_top_down_grasp_defaults() -> None:
    defaults = _launch_defaults()

    assert defaults["enable_command_gui"] == "false"
    assert defaults["pick_top_down_surface_penetration"] == "0.030"
    assert defaults["pick_target_x_bias"] == "0.015"
    assert defaults["pick_target_x_bias_decay_start_x"] == "0.50"
    assert defaults["pick_target_x_bias_side_decay_start_abs_y"] == "0.22"
    assert defaults["pick_command_runtime_budget_sec"] == "240.0"
    assert defaults["pick_source_search_budget_per_command"] == "6"
    assert defaults["grasp_stable_contact_margin"] == "0.003"
    assert defaults["pick_align_wrist_to_target_yaw"] == "true"
    assert defaults["pick_final_contact_position_tolerance"] == "0.01"
    assert defaults["pick_top_down_min_support_clearance"] == "0.004"
    assert defaults["pick_close_safe_xy_tolerance"] == "0.012"
    assert defaults["pick_close_safe_above_target_z_max"] == "0.012"
    assert defaults["pick_close_safe_below_target_z_max"] == "0.012"
    assert defaults["pick_no_contact_retry_extra_descent"] == "0.008"
    assert defaults["pick_no_contact_retry_min_clearance"] == "0.0"
    assert defaults["pick_reobserve_enabled"] == "true"


def test_agent_launch_uses_local_vl_assisted_grounding_defaults() -> None:
    defaults = _launch_defaults()

    assert defaults["local_vl_model"] == "qwen3-vl:8b"
    assert defaults["local_vl_authority_mode"] == "assistive"
    assert defaults["local_vl_timeout_sec"] == "45.0"
    assert defaults["local_vl_max_crop_px"] == "2048"
    assert defaults["local_vl_max_calls_per_command"] == "8"
    assert defaults["local_vl_candidate_limit"] == "8"
    assert defaults["local_vl_select_threshold"] == "0.55"
    assert defaults["local_vl_verify_threshold"] == "0.70"
    assert defaults["local_vl_post_place_timeout_sec"] == "45.0"
    assert defaults["source_local_vl_enabled"] == "true"
    assert defaults["grounder_query_expansion_enabled"] == "true"
    assert defaults["reobserve_occlusion_escalation_enabled"] == "true"
