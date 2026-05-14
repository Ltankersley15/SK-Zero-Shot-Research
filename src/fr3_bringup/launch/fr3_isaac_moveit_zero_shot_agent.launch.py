from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Resolve the bringup package once so we can include the shared FR3 + MoveIt stack below.
    bringup_pkg = get_package_share_directory("fr3_bringup")

    # These LaunchConfiguration objects are "late bound":
    # each one is filled in from a launch argument at runtime, not at import time.
    enable_command_gui = LaunchConfiguration("enable_command_gui")
    llm_provider = LaunchConfiguration("llm_provider")
    llm_model = LaunchConfiguration("llm_model")
    llm_base_url = LaunchConfiguration("llm_base_url")
    planner_max_calls_per_command = LaunchConfiguration("planner_max_calls_per_command")
    planner_max_scene_replans = LaunchConfiguration("planner_max_scene_replans")
    metric_pose_ablation_enabled = LaunchConfiguration("metric_pose_ablation_enabled")
    local_vl_verifier_enabled = LaunchConfiguration("local_vl_verifier_enabled")
    local_vl_model = LaunchConfiguration("local_vl_model")
    local_vl_timeout_sec = LaunchConfiguration("local_vl_timeout_sec")
    local_vl_max_crop_px = LaunchConfiguration("local_vl_max_crop_px")
    local_vl_max_calls_per_command = LaunchConfiguration("local_vl_max_calls_per_command")
    local_vl_candidate_limit = LaunchConfiguration("local_vl_candidate_limit")
    local_vl_confirm_threshold = LaunchConfiguration("local_vl_confirm_threshold")
    local_vl_authority_mode = LaunchConfiguration("local_vl_authority_mode")
    local_vl_select_threshold = LaunchConfiguration("local_vl_select_threshold")
    local_vl_verify_threshold = LaunchConfiguration("local_vl_verify_threshold")
    local_vl_post_place_timeout_sec = LaunchConfiguration("local_vl_post_place_timeout_sec")
    local_vl_disagreement_policy = LaunchConfiguration("local_vl_disagreement_policy")
    local_vl_enable_post_place_check = LaunchConfiguration("local_vl_enable_post_place_check")
    local_vl_enable_named_container_rerank = LaunchConfiguration("local_vl_enable_named_container_rerank")
    source_local_vl_enabled = LaunchConfiguration("source_local_vl_enabled")
    source_local_vl_max_calls_per_command = LaunchConfiguration("source_local_vl_max_calls_per_command")
    source_local_vl_candidate_limit = LaunchConfiguration("source_local_vl_candidate_limit")
    source_local_vl_confirm_threshold = LaunchConfiguration("source_local_vl_confirm_threshold")
    source_local_vl_disagreement_policy = LaunchConfiguration("source_local_vl_disagreement_policy")
    grounder_query_expansion_enabled = LaunchConfiguration("grounder_query_expansion_enabled")
    reobserve_occlusion_escalation_enabled = LaunchConfiguration("reobserve_occlusion_escalation_enabled")
    yolo_device = LaunchConfiguration("yolo_device")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    yolo_conf_thres = LaunchConfiguration("yolo_conf_thres")
    enable_sam2_fallback = LaunchConfiguration("enable_sam2_fallback")
    enable_color_blob_fallback = LaunchConfiguration("enable_color_blob_fallback")
    color_blob_min_area_frac = LaunchConfiguration("color_blob_min_area_frac")
    color_blob_max_area_frac = LaunchConfiguration("color_blob_max_area_frac")
    ee_link = LaunchConfiguration("ee_link")
    ee_tcp_offset_x = LaunchConfiguration("ee_tcp_offset_x")
    ee_tcp_offset_y = LaunchConfiguration("ee_tcp_offset_y")
    ee_tcp_offset_z = LaunchConfiguration("ee_tcp_offset_z")
    aligned_depth = LaunchConfiguration("aligned_depth")
    observation_x = LaunchConfiguration("observation_x")
    observation_y = LaunchConfiguration("observation_y")
    observation_z = LaunchConfiguration("observation_z")
    observation_yaw_rad = LaunchConfiguration("observation_yaw_rad")
    observation_position_tolerance = LaunchConfiguration("observation_position_tolerance")
    observation_orientation_tolerance = LaunchConfiguration("observation_orientation_tolerance")
    pose_service_timeout_sec = LaunchConfiguration("pose_service_timeout_sec")
    joint_service_timeout_sec = LaunchConfiguration("joint_service_timeout_sec")
    expected_table_z = LaunchConfiguration("expected_table_z")
    table_projection_enabled = LaunchConfiguration("table_projection_enabled")
    table_projection_xy_alpha = LaunchConfiguration("table_projection_xy_alpha")
    table_projection_max_xy_shift = LaunchConfiguration("table_projection_max_xy_shift")
    planning_scene_obstacles_enabled = LaunchConfiguration("planning_scene_obstacles_enabled")
    planning_scene_obstacle_topic = LaunchConfiguration("planning_scene_obstacle_topic")
    planning_scene_obstacle_update_period_sec = LaunchConfiguration("planning_scene_obstacle_update_period_sec")
    planning_scene_obstacle_labels = LaunchConfiguration("planning_scene_obstacle_labels")
    planning_scene_obstacle_shape_tokens = LaunchConfiguration("planning_scene_obstacle_shape_tokens")
    planning_scene_obstacle_radius_pad = LaunchConfiguration("planning_scene_obstacle_radius_pad")
    planning_scene_obstacle_box_pad = LaunchConfiguration("planning_scene_obstacle_box_pad")
    planning_scene_obstacle_cylinder_height = LaunchConfiguration("planning_scene_obstacle_cylinder_height")
    planning_scene_obstacle_box_height = LaunchConfiguration("planning_scene_obstacle_box_height")
    planning_scene_obstacle_max_count = LaunchConfiguration("planning_scene_obstacle_max_count")
    planning_scene_generic_obstacle_min_score = LaunchConfiguration("planning_scene_generic_obstacle_min_score")
    planning_scene_target_bbox_iou_threshold = LaunchConfiguration("planning_scene_target_bbox_iou_threshold")
    planning_scene_target_xy_separation_m = LaunchConfiguration("planning_scene_target_xy_separation_m")
    planning_scene_apply_service = LaunchConfiguration("planning_scene_apply_service")
    planning_scene_apply_timeout_sec = LaunchConfiguration("planning_scene_apply_timeout_sec")
    runtime_stream_gap_reset_sec = LaunchConfiguration("runtime_stream_gap_reset_sec")
    runtime_stream_resume_settle_sec = LaunchConfiguration("runtime_stream_resume_settle_sec")
    scene_memory_enabled = LaunchConfiguration("scene_memory_enabled")
    move_position_tolerance = LaunchConfiguration("move_position_tolerance")
    place_hover_height = LaunchConfiguration("place_hover_height")
    place_on_surface_offset = LaunchConfiguration("place_on_surface_offset")
    place_side_offset = LaunchConfiguration("place_side_offset")
    place_in_release_above_rim = LaunchConfiguration("place_in_release_above_rim")
    place_in_min_target_z_offset = LaunchConfiguration("place_in_min_target_z_offset")
    place_in_hover_above_rim = LaunchConfiguration("place_in_hover_above_rim")
    container_insertion_center_max_offset_m = LaunchConfiguration("container_insertion_center_max_offset_m")
    place_container_mode = LaunchConfiguration("place_container_mode")
    oracle_container_x = LaunchConfiguration("oracle_container_x")
    oracle_container_y = LaunchConfiguration("oracle_container_y")
    oracle_container_z = LaunchConfiguration("oracle_container_z")
    oracle_container_support_z = LaunchConfiguration("oracle_container_support_z")
    oracle_container_rim_z = LaunchConfiguration("oracle_container_rim_z")
    oracle_container_outer_footprint_x = LaunchConfiguration("oracle_container_outer_footprint_x")
    oracle_container_outer_footprint_y = LaunchConfiguration("oracle_container_outer_footprint_y")
    place_in_release_xy_bias_x = LaunchConfiguration("place_in_release_xy_bias_x")
    place_in_release_xy_bias_y = LaunchConfiguration("place_in_release_xy_bias_y")
    pick_position_tolerance = LaunchConfiguration("pick_position_tolerance")
    pick_final_contact_position_tolerance = LaunchConfiguration("pick_final_contact_position_tolerance")
    pick_orientation_tolerance = LaunchConfiguration("pick_orientation_tolerance")
    pick_hover_orientation_tolerance = LaunchConfiguration("pick_hover_orientation_tolerance")
    pick_final_contact_orientation_tolerance = LaunchConfiguration("pick_final_contact_orientation_tolerance")
    observe_before_grounding = LaunchConfiguration("observe_before_grounding")
    ready_before_observation = LaunchConfiguration("ready_before_observation")
    pick_top_down_table_clearance = LaunchConfiguration("pick_top_down_table_clearance")
    pick_top_down_contact_offset = LaunchConfiguration("pick_top_down_contact_offset")
    pick_top_down_surface_penetration = LaunchConfiguration("pick_top_down_surface_penetration")
    pick_top_down_min_support_clearance = LaunchConfiguration("pick_top_down_min_support_clearance")
    pick_inspect_backoff_x = LaunchConfiguration("pick_inspect_backoff_x")
    pick_inspect_height = LaunchConfiguration("pick_inspect_height")
    pick_target_x_bias = LaunchConfiguration("pick_target_x_bias")
    pick_target_x_bias_decay_start_x = LaunchConfiguration("pick_target_x_bias_decay_start_x")
    pick_target_x_bias_side_decay_start_abs_y = LaunchConfiguration("pick_target_x_bias_side_decay_start_abs_y")
    pick_search_enabled = LaunchConfiguration("pick_search_enabled")
    pick_search_max_steps = LaunchConfiguration("pick_search_max_steps")
    pick_command_runtime_budget_sec = LaunchConfiguration("pick_command_runtime_budget_sec")
    pick_source_search_budget_per_command = LaunchConfiguration("pick_source_search_budget_per_command")
    pick_search_view_z = LaunchConfiguration("pick_search_view_z")
    pick_search_settle_sec = LaunchConfiguration("pick_search_settle_sec")
    pick_search_y_span = LaunchConfiguration("pick_search_y_span")
    pick_search_edge_margin_px = LaunchConfiguration("pick_search_edge_margin_px")
    pick_search_max_area_frac = LaunchConfiguration("pick_search_max_area_frac")
    pick_low_approach_height = LaunchConfiguration("pick_low_approach_height")
    pick_low_approach_settle_passes = LaunchConfiguration("pick_low_approach_settle_passes")
    pick_top_down_hover_recenter_trigger_xy = LaunchConfiguration("pick_top_down_hover_recenter_trigger_xy")
    pick_top_down_hover_recenter_max_xy = LaunchConfiguration("pick_top_down_hover_recenter_max_xy")
    pick_top_down_hover_recenter_attempts = LaunchConfiguration("pick_top_down_hover_recenter_attempts")
    pick_top_down_hover_recenter_min_xy_error = LaunchConfiguration("pick_top_down_hover_recenter_min_xy_error")
    pick_top_down_hover_recenter_lift_z = LaunchConfiguration("pick_top_down_hover_recenter_lift_z")
    pick_top_down_hover_recenter_step_xy_max = LaunchConfiguration("pick_top_down_hover_recenter_step_xy_max")
    pick_top_down_descent_gate_enabled = LaunchConfiguration("pick_top_down_descent_gate_enabled")
    pick_top_down_descent_xy_tolerance = LaunchConfiguration("pick_top_down_descent_xy_tolerance")
    pick_top_down_descent_xy_tolerance_after_soft_miss = LaunchConfiguration(
        "pick_top_down_descent_xy_tolerance_after_soft_miss"
    )
    pick_top_down_descent_z_slack_below = LaunchConfiguration("pick_top_down_descent_z_slack_below")
    pick_top_down_descent_z_slack_above = LaunchConfiguration("pick_top_down_descent_z_slack_above")
    pick_top_down_descent_orientation_tolerance = LaunchConfiguration("pick_top_down_descent_orientation_tolerance")
    pick_align_wrist_to_target_yaw = LaunchConfiguration("pick_align_wrist_to_target_yaw")
    pick_top_down_close_from_low_hover_enabled = LaunchConfiguration("pick_top_down_close_from_low_hover_enabled")
    pick_top_down_low_hover_close_xy_tolerance = LaunchConfiguration("pick_top_down_low_hover_close_xy_tolerance")
    pick_top_down_low_hover_close_above_target_z_max = LaunchConfiguration(
        "pick_top_down_low_hover_close_above_target_z_max"
    )
    pick_top_down_low_hover_close_below_target_z_max = LaunchConfiguration(
        "pick_top_down_low_hover_close_below_target_z_max"
    )
    pick_close_on_safe_z_miss_enabled = LaunchConfiguration("pick_close_on_safe_z_miss_enabled")
    pick_close_safe_xy_tolerance = LaunchConfiguration("pick_close_safe_xy_tolerance")
    pick_close_safe_above_target_z_max = LaunchConfiguration("pick_close_safe_above_target_z_max")
    pick_close_safe_below_target_z_max = LaunchConfiguration("pick_close_safe_below_target_z_max")
    pick_close_safe_surface_margin = LaunchConfiguration("pick_close_safe_surface_margin")
    pick_close_safe_orientation_tolerance = LaunchConfiguration("pick_close_safe_orientation_tolerance")
    grasp_stable_contact_margin = LaunchConfiguration("grasp_stable_contact_margin")
    pick_no_contact_retry_enabled = LaunchConfiguration("pick_no_contact_retry_enabled")
    pick_no_contact_retry_extra_descent = LaunchConfiguration("pick_no_contact_retry_extra_descent")
    pick_no_contact_retry_min_clearance = LaunchConfiguration("pick_no_contact_retry_min_clearance")
    pick_reobserve_enabled = LaunchConfiguration("pick_reobserve_enabled")
    pick_reobserve_max_xy_shift = LaunchConfiguration("pick_reobserve_max_xy_shift")
    pick_reobserve_xy_update_limit = LaunchConfiguration("pick_reobserve_xy_update_limit")
    pick_reobserve_max_uv_shift_px = LaunchConfiguration("pick_reobserve_max_uv_shift_px")
    pick_reobserve_confirm_frames = LaunchConfiguration("pick_reobserve_confirm_frames")
    pick_reobserve_confirm_px = LaunchConfiguration("pick_reobserve_confirm_px")
    embodiment_gate_enabled = LaunchConfiguration("embodiment_gate_enabled")
    embodiment_gate_xy_tolerance = LaunchConfiguration("embodiment_gate_xy_tolerance")
    embodiment_gate_z_tolerance = LaunchConfiguration("embodiment_gate_z_tolerance")
    embodiment_gate_orientation_tolerance = LaunchConfiguration("embodiment_gate_orientation_tolerance")
    skill_max_velocity_scaling = LaunchConfiguration("skill_max_velocity_scaling")
    skill_max_acceleration_scaling = LaunchConfiguration("skill_max_acceleration_scaling")
    skill_cartesian_pose_verification_enabled = LaunchConfiguration("skill_cartesian_pose_verification_enabled")
    skill_cartesian_pose_correction_attempts = LaunchConfiguration("skill_cartesian_pose_correction_attempts")
    skill_cartesian_pose_default_position_tolerance = LaunchConfiguration(
        "skill_cartesian_pose_default_position_tolerance"
    )
    skill_cartesian_pose_default_orientation_tolerance = LaunchConfiguration(
        "skill_cartesian_pose_default_orientation_tolerance"
    )
    skill_cartesian_pose_post_execute_settle_time = LaunchConfiguration(
        "skill_cartesian_pose_post_execute_settle_time"
    )
    skill_cartesian_pose_max_correction_step = LaunchConfiguration("skill_cartesian_pose_max_correction_step")
    skill_cartesian_pose_min_position_improvement = LaunchConfiguration(
        "skill_cartesian_pose_min_position_improvement"
    )
    fjt_rate_hz = LaunchConfiguration("fjt_rate_hz")
    fjt_extra_settle_time = LaunchConfiguration("fjt_extra_settle_time")
    fjt_min_traj_duration = LaunchConfiguration("fjt_min_traj_duration")
    fjt_goal_tolerance_rad = LaunchConfiguration("fjt_goal_tolerance_rad")

    # First bring up the shared FR3 motion stack.
    # This includes MoveIt, the skill server, and the trajectory relay.
    # The lean agent sits on top of this stack rather than replacing it.
    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            bringup_pkg + "/launch/fr3_isaac_moveit_bringup.launch.py"
        ),
        launch_arguments={
            # These arguments tune the motion/control backend used by the agent.
            "skill_ee_link": ee_link,
            "skill_ee_tcp_offset_x": ee_tcp_offset_x,
            "skill_ee_tcp_offset_y": ee_tcp_offset_y,
            "skill_ee_tcp_offset_z": ee_tcp_offset_z,
            "skill_max_velocity_scaling": skill_max_velocity_scaling,
            "skill_max_acceleration_scaling": skill_max_acceleration_scaling,
            "skill_cartesian_pose_verification_enabled": skill_cartesian_pose_verification_enabled,
            "skill_cartesian_pose_correction_attempts": skill_cartesian_pose_correction_attempts,
            "skill_cartesian_pose_default_position_tolerance": skill_cartesian_pose_default_position_tolerance,
            "skill_cartesian_pose_default_orientation_tolerance": skill_cartesian_pose_default_orientation_tolerance,
            "skill_cartesian_pose_post_execute_settle_time": skill_cartesian_pose_post_execute_settle_time,
            "skill_cartesian_pose_max_correction_step": skill_cartesian_pose_max_correction_step,
            "skill_cartesian_pose_min_position_improvement": skill_cartesian_pose_min_position_improvement,
            "fjt_rate_hz": fjt_rate_hz,
            "fjt_extra_settle_time": fjt_extra_settle_time,
            "fjt_min_traj_duration": fjt_min_traj_duration,
            "fjt_goal_tolerance_rad": fjt_goal_tolerance_rad,
        }.items(),
    )

    # This is the actual agent node.
    # It owns planning, grounding, command execution, and GUI trace publishing.
    agent_node = Node(
        package="fr3_zero_shot_agent",
        executable="zero_shot_agent_runtime",
        name="fr3_agent",
        output="screen",
        parameters=[
            {
                # Planner parameters.
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "llm_base_url": llm_base_url,
                "planner_max_calls_per_command": planner_max_calls_per_command,
                "planner_max_scene_replans": planner_max_scene_replans,
                "metric_pose_ablation_enabled": metric_pose_ablation_enabled,
                # Local VL guidance / verification parameters.
                "local_vl_verifier_enabled": local_vl_verifier_enabled,
                "local_vl_model": local_vl_model,
                "local_vl_timeout_sec": local_vl_timeout_sec,
                "local_vl_max_crop_px": local_vl_max_crop_px,
                "local_vl_max_calls_per_command": local_vl_max_calls_per_command,
                "local_vl_candidate_limit": local_vl_candidate_limit,
                "local_vl_confirm_threshold": local_vl_confirm_threshold,
                "local_vl_authority_mode": local_vl_authority_mode,
                "local_vl_select_threshold": local_vl_select_threshold,
                "local_vl_verify_threshold": local_vl_verify_threshold,
                "local_vl_post_place_timeout_sec": local_vl_post_place_timeout_sec,
                "local_vl_disagreement_policy": local_vl_disagreement_policy,
                "local_vl_enable_post_place_check": local_vl_enable_post_place_check,
                "local_vl_enable_named_container_rerank": local_vl_enable_named_container_rerank,
                "source_local_vl_enabled": source_local_vl_enabled,
                "source_local_vl_max_calls_per_command": source_local_vl_max_calls_per_command,
                "source_local_vl_candidate_limit": source_local_vl_candidate_limit,
                "source_local_vl_confirm_threshold": source_local_vl_confirm_threshold,
                "source_local_vl_disagreement_policy": source_local_vl_disagreement_policy,
                "grounder_query_expansion_enabled": grounder_query_expansion_enabled,
                "reobserve_occlusion_escalation_enabled": reobserve_occlusion_escalation_enabled,
                # Fast perception parameters.
                "yolo_device": yolo_device,
                "yolo_model_path": yolo_model_path,
                "yolo_conf_thres": yolo_conf_thres,
                "enable_sam2_fallback": enable_sam2_fallback,
                "enable_color_blob_fallback": enable_color_blob_fallback,
                "color_blob_min_area_frac": color_blob_min_area_frac,
                "color_blob_max_area_frac": color_blob_max_area_frac,
                # TCP / embodiment parameters.
                "ee_link": ee_link,
                "ee_tcp_offset_x": ee_tcp_offset_x,
                "ee_tcp_offset_y": ee_tcp_offset_y,
                "ee_tcp_offset_z": ee_tcp_offset_z,
                "aligned_depth": aligned_depth,
                # Observation pose defaults used when we explicitly survey from a staging pose.
                "observation_x": observation_x,
                "observation_y": observation_y,
                "observation_z": observation_z,
                "observation_yaw_rad": observation_yaw_rad,
                "observation_position_tolerance": observation_position_tolerance,
                "observation_orientation_tolerance": observation_orientation_tolerance,
                "pose_service_timeout_sec": pose_service_timeout_sec,
                "joint_service_timeout_sec": joint_service_timeout_sec,
                # Table projection lets us keep grasp points snapped to the tabletop model.
                "expected_table_z": expected_table_z,
                "table_projection_enabled": table_projection_enabled,
                "table_projection_xy_alpha": table_projection_xy_alpha,
                "table_projection_max_xy_shift": table_projection_max_xy_shift,
                "planning_scene_obstacles_enabled": planning_scene_obstacles_enabled,
                "planning_scene_obstacle_topic": planning_scene_obstacle_topic,
                "planning_scene_obstacle_update_period_sec": planning_scene_obstacle_update_period_sec,
                "planning_scene_obstacle_labels": planning_scene_obstacle_labels,
                "planning_scene_obstacle_shape_tokens": planning_scene_obstacle_shape_tokens,
                "planning_scene_obstacle_radius_pad": planning_scene_obstacle_radius_pad,
                "planning_scene_obstacle_box_pad": planning_scene_obstacle_box_pad,
                "planning_scene_obstacle_cylinder_height": planning_scene_obstacle_cylinder_height,
                "planning_scene_obstacle_box_height": planning_scene_obstacle_box_height,
                "planning_scene_obstacle_max_count": planning_scene_obstacle_max_count,
                "planning_scene_generic_obstacle_min_score": planning_scene_generic_obstacle_min_score,
                "planning_scene_target_bbox_iou_threshold": planning_scene_target_bbox_iou_threshold,
                "planning_scene_target_xy_separation_m": planning_scene_target_xy_separation_m,
                "planning_scene_apply_service": planning_scene_apply_service,
                "planning_scene_apply_timeout_sec": planning_scene_apply_timeout_sec,
                "runtime_stream_gap_reset_sec": runtime_stream_gap_reset_sec,
                "runtime_stream_resume_settle_sec": runtime_stream_resume_settle_sec,
                "scene_memory_enabled": scene_memory_enabled,
                # Executor tolerances.
                "move_position_tolerance": move_position_tolerance,
                "place_hover_height": place_hover_height,
                "place_on_surface_offset": place_on_surface_offset,
                "place_side_offset": place_side_offset,
                "place_in_release_above_rim": place_in_release_above_rim,
                "place_in_min_target_z_offset": place_in_min_target_z_offset,
                "place_in_hover_above_rim": place_in_hover_above_rim,
                "container_insertion_center_max_offset_m": container_insertion_center_max_offset_m,
                "place_container_mode": place_container_mode,
                "oracle_container_x": oracle_container_x,
                "oracle_container_y": oracle_container_y,
                "oracle_container_z": oracle_container_z,
                "oracle_container_support_z": oracle_container_support_z,
                "oracle_container_rim_z": oracle_container_rim_z,
                "oracle_container_outer_footprint_x": oracle_container_outer_footprint_x,
                "oracle_container_outer_footprint_y": oracle_container_outer_footprint_y,
                "place_in_release_xy_bias_x": place_in_release_xy_bias_x,
                "place_in_release_xy_bias_y": place_in_release_xy_bias_y,
                "pick_position_tolerance": pick_position_tolerance,
                "pick_final_contact_position_tolerance": pick_final_contact_position_tolerance,
                "pick_orientation_tolerance": pick_orientation_tolerance,
                "pick_hover_orientation_tolerance": pick_hover_orientation_tolerance,
                "pick_final_contact_orientation_tolerance": pick_final_contact_orientation_tolerance,
                "observe_before_grounding": observe_before_grounding,
                "ready_before_observation": ready_before_observation,
                # Top-down pick behavior.
                "pick_top_down_table_clearance": pick_top_down_table_clearance,
                "pick_top_down_contact_offset": pick_top_down_contact_offset,
                "pick_top_down_surface_penetration": pick_top_down_surface_penetration,
                "pick_top_down_min_support_clearance": pick_top_down_min_support_clearance,
                "pick_inspect_backoff_x": pick_inspect_backoff_x,
                "pick_inspect_height": pick_inspect_height,
                "pick_target_x_bias": pick_target_x_bias,
                "pick_target_x_bias_decay_start_x": pick_target_x_bias_decay_start_x,
                "pick_target_x_bias_side_decay_start_abs_y": pick_target_x_bias_side_decay_start_abs_y,
                "pick_search_enabled": pick_search_enabled,
                "pick_search_max_steps": pick_search_max_steps,
                "pick_command_runtime_budget_sec": pick_command_runtime_budget_sec,
                "pick_source_search_budget_per_command": pick_source_search_budget_per_command,
                "pick_search_view_z": pick_search_view_z,
                "pick_search_settle_sec": pick_search_settle_sec,
                "pick_search_y_span": pick_search_y_span,
                "pick_search_edge_margin_px": pick_search_edge_margin_px,
                "pick_search_max_area_frac": pick_search_max_area_frac,
                "pick_low_approach_height": pick_low_approach_height,
                "pick_low_approach_settle_passes": pick_low_approach_settle_passes,
                "pick_top_down_hover_recenter_trigger_xy": pick_top_down_hover_recenter_trigger_xy,
                "pick_top_down_hover_recenter_max_xy": pick_top_down_hover_recenter_max_xy,
                "pick_top_down_hover_recenter_attempts": pick_top_down_hover_recenter_attempts,
                "pick_top_down_hover_recenter_min_xy_error": pick_top_down_hover_recenter_min_xy_error,
                "pick_top_down_hover_recenter_lift_z": pick_top_down_hover_recenter_lift_z,
                "pick_top_down_hover_recenter_step_xy_max": pick_top_down_hover_recenter_step_xy_max,
                "pick_top_down_descent_gate_enabled": pick_top_down_descent_gate_enabled,
                "pick_top_down_descent_xy_tolerance": pick_top_down_descent_xy_tolerance,
                "pick_top_down_descent_xy_tolerance_after_soft_miss": pick_top_down_descent_xy_tolerance_after_soft_miss,
                "pick_top_down_descent_z_slack_below": pick_top_down_descent_z_slack_below,
                "pick_top_down_descent_z_slack_above": pick_top_down_descent_z_slack_above,
                "pick_top_down_descent_orientation_tolerance": pick_top_down_descent_orientation_tolerance,
                "pick_align_wrist_to_target_yaw": pick_align_wrist_to_target_yaw,
                "pick_top_down_close_from_low_hover_enabled": pick_top_down_close_from_low_hover_enabled,
                "pick_top_down_low_hover_close_xy_tolerance": pick_top_down_low_hover_close_xy_tolerance,
                "pick_top_down_low_hover_close_above_target_z_max": pick_top_down_low_hover_close_above_target_z_max,
                "pick_top_down_low_hover_close_below_target_z_max": pick_top_down_low_hover_close_below_target_z_max,
                "pick_close_on_safe_z_miss_enabled": pick_close_on_safe_z_miss_enabled,
                "pick_close_safe_xy_tolerance": pick_close_safe_xy_tolerance,
                "pick_close_safe_above_target_z_max": pick_close_safe_above_target_z_max,
                "pick_close_safe_below_target_z_max": pick_close_safe_below_target_z_max,
                "pick_close_safe_surface_margin": pick_close_safe_surface_margin,
                "pick_close_safe_orientation_tolerance": pick_close_safe_orientation_tolerance,
                "grasp_stable_contact_margin": grasp_stable_contact_margin,
                "pick_no_contact_retry_enabled": pick_no_contact_retry_enabled,
                "pick_no_contact_retry_extra_descent": pick_no_contact_retry_extra_descent,
                "pick_no_contact_retry_min_clearance": pick_no_contact_retry_min_clearance,
                "pick_reobserve_enabled": pick_reobserve_enabled,
                "pick_reobserve_max_xy_shift": pick_reobserve_max_xy_shift,
                "pick_reobserve_xy_update_limit": pick_reobserve_xy_update_limit,
                "pick_reobserve_max_uv_shift_px": pick_reobserve_max_uv_shift_px,
                "pick_reobserve_confirm_frames": pick_reobserve_confirm_frames,
                "pick_reobserve_confirm_px": pick_reobserve_confirm_px,
                "embodiment_gate_enabled": embodiment_gate_enabled,
                "embodiment_gate_xy_tolerance": embodiment_gate_xy_tolerance,
                "embodiment_gate_z_tolerance": embodiment_gate_z_tolerance,
                "embodiment_gate_orientation_tolerance": embodiment_gate_orientation_tolerance,
                # UI topics shared with the existing command GUI.
                "cmd_topic": "/lvlm_agent/command",
                "result_topic": "/lvlm_agent/command_result",
                "reasoning_topic": "/lvlm_agent/reasoning_trace",
                "debug_publish_image": True,
                "debug_image_topic": "/lvlm_agent/debug_image",
            }
        ],
    )

    # Reuse the old LVLM command GUI as a thin front end for the lean agent.
    gui_node = Node(
        package="fr3_lvlm_agent",
        executable="lvlm_command_gui",
        name="lvlm_command_gui",
        output="screen",
        condition=IfCondition(enable_command_gui),
        parameters=[
            {
                "image_topic": "/lvlm_agent/debug_image",
                "cmd_topic": "/lvlm_agent/command",
                "result_topic": "/lvlm_agent/command_result",
                "reasoning_topic": "/lvlm_agent/reasoning_trace",
            }
        ],
    )

    # The launch description is mostly a long list of knobs.
    # Keeping them explicit here makes it easy to inspect what the agent can tune.
    return LaunchDescription(
        [
            DeclareLaunchArgument("enable_command_gui", default_value="false"),
            DeclareLaunchArgument("llm_provider", default_value="openai"),
            DeclareLaunchArgument("llm_model", default_value="gpt-5-mini"),
            DeclareLaunchArgument("llm_base_url", default_value="http://localhost:11434"),
            DeclareLaunchArgument("planner_max_calls_per_command", default_value="1"),
            DeclareLaunchArgument("planner_max_scene_replans", default_value="1"),
            DeclareLaunchArgument("metric_pose_ablation_enabled", default_value="false"),
            DeclareLaunchArgument("local_vl_verifier_enabled", default_value="true"),
            DeclareLaunchArgument("local_vl_model", default_value="qwen3-vl:8b"),
            DeclareLaunchArgument("local_vl_timeout_sec", default_value="45.0"),
            DeclareLaunchArgument("local_vl_max_crop_px", default_value="2048"),
            DeclareLaunchArgument("local_vl_max_calls_per_command", default_value="8"),
            DeclareLaunchArgument("local_vl_candidate_limit", default_value="8"),
            DeclareLaunchArgument("local_vl_confirm_threshold", default_value="0.70"),
            DeclareLaunchArgument("local_vl_authority_mode", default_value="assistive"),
            DeclareLaunchArgument("local_vl_select_threshold", default_value="0.55"),
            DeclareLaunchArgument("local_vl_verify_threshold", default_value="0.70"),
            DeclareLaunchArgument("local_vl_post_place_timeout_sec", default_value="45.0"),
            DeclareLaunchArgument("local_vl_disagreement_policy", default_value="fail_closed"),
            DeclareLaunchArgument("local_vl_enable_post_place_check", default_value="true"),
            DeclareLaunchArgument("local_vl_enable_named_container_rerank", default_value="true"),
            DeclareLaunchArgument("source_local_vl_enabled", default_value="true"),
            DeclareLaunchArgument("source_local_vl_max_calls_per_command", default_value="2"),
            DeclareLaunchArgument("source_local_vl_candidate_limit", default_value="6"),
            DeclareLaunchArgument("source_local_vl_confirm_threshold", default_value="0.45"),
            DeclareLaunchArgument("source_local_vl_disagreement_policy", default_value="fail_closed"),
            DeclareLaunchArgument("grounder_query_expansion_enabled", default_value="true"),
            DeclareLaunchArgument("reobserve_occlusion_escalation_enabled", default_value="true"),
            DeclareLaunchArgument("yolo_device", default_value="cuda"),
            DeclareLaunchArgument("yolo_model_path", default_value=""),
            DeclareLaunchArgument("yolo_conf_thres", default_value="0.05"),
            DeclareLaunchArgument("enable_sam2_fallback", default_value="true"),
            DeclareLaunchArgument("enable_color_blob_fallback", default_value="true"),
            DeclareLaunchArgument("color_blob_min_area_frac", default_value="0.0002"),
            DeclareLaunchArgument("color_blob_max_area_frac", default_value="0.10"),
            DeclareLaunchArgument("ee_link", default_value="fr3_link8"),
            DeclareLaunchArgument("ee_tcp_offset_x", default_value="0.0"),
            DeclareLaunchArgument("ee_tcp_offset_y", default_value="0.0"),
            DeclareLaunchArgument("ee_tcp_offset_z", default_value="0.10"),
            DeclareLaunchArgument("aligned_depth", default_value="true"),
            DeclareLaunchArgument("observation_x", default_value="0.48"),
            DeclareLaunchArgument("observation_y", default_value="0.0"),
            DeclareLaunchArgument("observation_z", default_value="0.38"),
            DeclareLaunchArgument("observation_yaw_rad", default_value="0.0"),
            DeclareLaunchArgument("observation_position_tolerance", default_value="0.08"),
            DeclareLaunchArgument("observation_orientation_tolerance", default_value="0.20"),
            DeclareLaunchArgument("pose_service_timeout_sec", default_value="45.0"),
            DeclareLaunchArgument("joint_service_timeout_sec", default_value="45.0"),
            DeclareLaunchArgument("expected_table_z", default_value="0.02"),
            DeclareLaunchArgument("table_projection_enabled", default_value="true"),
            DeclareLaunchArgument("table_projection_xy_alpha", default_value="0.85"),
            DeclareLaunchArgument("table_projection_max_xy_shift", default_value="0.45"),
            DeclareLaunchArgument("planning_scene_obstacles_enabled", default_value="true"),
            DeclareLaunchArgument("planning_scene_obstacle_topic", default_value="/fr3/planning_scene"),
            DeclareLaunchArgument("planning_scene_obstacle_update_period_sec", default_value="0.5"),
            DeclareLaunchArgument("planning_scene_obstacle_labels", default_value="cup,mug,glass,beaker,bowl,bottle,can,jar,flask"),
            DeclareLaunchArgument("planning_scene_obstacle_shape_tokens", default_value="cylinder"),
            DeclareLaunchArgument("planning_scene_obstacle_radius_pad", default_value="0.015"),
            DeclareLaunchArgument("planning_scene_obstacle_box_pad", default_value="0.020"),
            DeclareLaunchArgument("planning_scene_obstacle_cylinder_height", default_value="0.14"),
            DeclareLaunchArgument("planning_scene_obstacle_box_height", default_value="0.10"),
            DeclareLaunchArgument("planning_scene_obstacle_max_count", default_value="6"),
            DeclareLaunchArgument("planning_scene_generic_obstacle_min_score", default_value="0.20"),
            DeclareLaunchArgument("planning_scene_target_bbox_iou_threshold", default_value="0.25"),
            DeclareLaunchArgument("planning_scene_target_xy_separation_m", default_value="0.06"),
            DeclareLaunchArgument("planning_scene_apply_service", default_value="/fr3/apply_planning_scene"),
            DeclareLaunchArgument("planning_scene_apply_timeout_sec", default_value="0.35"),
            DeclareLaunchArgument("runtime_stream_gap_reset_sec", default_value="1.5"),
            DeclareLaunchArgument("runtime_stream_resume_settle_sec", default_value="3.0"),
            DeclareLaunchArgument("scene_memory_enabled", default_value="true"),
            DeclareLaunchArgument("move_position_tolerance", default_value="0.08"),
            DeclareLaunchArgument("place_hover_height", default_value="0.10"),
            DeclareLaunchArgument("place_on_surface_offset", default_value="0.05"),
            DeclareLaunchArgument("place_side_offset", default_value="0.07"),
            DeclareLaunchArgument("place_in_release_above_rim", default_value="0.06"),
            DeclareLaunchArgument("place_in_min_target_z_offset", default_value="0.14"),
            DeclareLaunchArgument("place_in_hover_above_rim", default_value="0.16"),
            DeclareLaunchArgument("container_insertion_center_max_offset_m", default_value="0.025"),
            DeclareLaunchArgument("place_container_mode", default_value="embodied"),
            DeclareLaunchArgument("oracle_container_x", default_value="0.0"),
            DeclareLaunchArgument("oracle_container_y", default_value="0.0"),
            DeclareLaunchArgument("oracle_container_z", default_value="0.0"),
            DeclareLaunchArgument("oracle_container_support_z", default_value="0.0"),
            DeclareLaunchArgument("oracle_container_rim_z", default_value="0.0"),
            DeclareLaunchArgument("oracle_container_outer_footprint_x", default_value="0.0"),
            DeclareLaunchArgument("oracle_container_outer_footprint_y", default_value="0.0"),
            DeclareLaunchArgument("place_in_release_xy_bias_x", default_value="0.0"),
            DeclareLaunchArgument("place_in_release_xy_bias_y", default_value="0.0"),
            DeclareLaunchArgument("pick_position_tolerance", default_value="0.04"),
            DeclareLaunchArgument("pick_final_contact_position_tolerance", default_value="0.01"),
            DeclareLaunchArgument("pick_orientation_tolerance", default_value="0.20"),
            DeclareLaunchArgument("pick_hover_orientation_tolerance", default_value="0.25"),
            DeclareLaunchArgument("pick_final_contact_orientation_tolerance", default_value="0.12"),
            DeclareLaunchArgument("observe_before_grounding", default_value="true"),
            DeclareLaunchArgument("ready_before_observation", default_value="true"),
            DeclareLaunchArgument("pick_top_down_table_clearance", default_value="0.015"),
            DeclareLaunchArgument("pick_top_down_contact_offset", default_value="0.0"),
            DeclareLaunchArgument("pick_top_down_surface_penetration", default_value="0.030"),
            DeclareLaunchArgument("pick_top_down_min_support_clearance", default_value="0.004"),
            DeclareLaunchArgument("pick_inspect_backoff_x", default_value="0.06"),
            DeclareLaunchArgument("pick_inspect_height", default_value="0.14"),
            DeclareLaunchArgument("pick_target_x_bias", default_value="0.015"),
            DeclareLaunchArgument("pick_target_x_bias_decay_start_x", default_value="0.50"),
            DeclareLaunchArgument("pick_target_x_bias_side_decay_start_abs_y", default_value="0.22"),
            DeclareLaunchArgument("pick_search_enabled", default_value="true"),
            DeclareLaunchArgument("pick_search_max_steps", default_value="5"),
            DeclareLaunchArgument("pick_command_runtime_budget_sec", default_value="240.0"),
            DeclareLaunchArgument("pick_source_search_budget_per_command", default_value="6"),
            DeclareLaunchArgument("pick_search_view_z", default_value="0.42"),
            DeclareLaunchArgument("pick_search_settle_sec", default_value="0.25"),
            DeclareLaunchArgument("pick_search_y_span", default_value="0.28"),
            DeclareLaunchArgument("pick_search_edge_margin_px", default_value="24"),
            DeclareLaunchArgument("pick_search_max_area_frac", default_value="0.20"),
            DeclareLaunchArgument("pick_low_approach_height", default_value="0.05"),
            DeclareLaunchArgument("pick_low_approach_settle_passes", default_value="1"),
            DeclareLaunchArgument("pick_top_down_hover_recenter_trigger_xy", default_value="0.025"),
            DeclareLaunchArgument("pick_top_down_hover_recenter_max_xy", default_value="0.120"),
            DeclareLaunchArgument("pick_top_down_hover_recenter_attempts", default_value="2"),
            DeclareLaunchArgument("pick_top_down_hover_recenter_min_xy_error", default_value="0.045"),
            DeclareLaunchArgument("pick_top_down_hover_recenter_lift_z", default_value="0.0"),
            DeclareLaunchArgument("pick_top_down_hover_recenter_step_xy_max", default_value="0.015"),
            DeclareLaunchArgument("pick_top_down_descent_gate_enabled", default_value="true"),
            DeclareLaunchArgument("pick_top_down_descent_xy_tolerance", default_value="0.04"),
            DeclareLaunchArgument("pick_top_down_descent_xy_tolerance_after_soft_miss", default_value="0.04"),
            DeclareLaunchArgument("pick_top_down_descent_z_slack_below", default_value="0.015"),
            DeclareLaunchArgument("pick_top_down_descent_z_slack_above", default_value="0.03"),
            DeclareLaunchArgument("pick_top_down_descent_orientation_tolerance", default_value="0.12"),
            DeclareLaunchArgument("pick_align_wrist_to_target_yaw", default_value="true"),
            DeclareLaunchArgument("pick_top_down_close_from_low_hover_enabled", default_value="false"),
            DeclareLaunchArgument("pick_top_down_low_hover_close_xy_tolerance", default_value="0.04"),
            DeclareLaunchArgument("pick_top_down_low_hover_close_above_target_z_max", default_value="0.020"),
            DeclareLaunchArgument("pick_top_down_low_hover_close_below_target_z_max", default_value="0.010"),
            DeclareLaunchArgument("pick_close_on_safe_z_miss_enabled", default_value="true"),
            DeclareLaunchArgument("pick_close_safe_xy_tolerance", default_value="0.012"),
            DeclareLaunchArgument("pick_close_safe_above_target_z_max", default_value="0.012"),
            DeclareLaunchArgument("pick_close_safe_below_target_z_max", default_value="0.012"),
            DeclareLaunchArgument("pick_close_safe_surface_margin", default_value="0.005"),
            DeclareLaunchArgument("pick_close_safe_orientation_tolerance", default_value="0.30"),
            DeclareLaunchArgument("grasp_stable_contact_margin", default_value="0.003"),
            DeclareLaunchArgument("pick_no_contact_retry_enabled", default_value="true"),
            DeclareLaunchArgument("pick_no_contact_retry_extra_descent", default_value="0.008"),
            DeclareLaunchArgument("pick_no_contact_retry_min_clearance", default_value="0.0"),
            DeclareLaunchArgument("pick_reobserve_enabled", default_value="true"),
            DeclareLaunchArgument("pick_reobserve_max_xy_shift", default_value="0.08"),
            DeclareLaunchArgument("pick_reobserve_xy_update_limit", default_value="0.01"),
            DeclareLaunchArgument("pick_reobserve_max_uv_shift_px", default_value="180.0"),
            DeclareLaunchArgument("pick_reobserve_confirm_frames", default_value="2"),
            DeclareLaunchArgument("pick_reobserve_confirm_px", default_value="35.0"),
            DeclareLaunchArgument("embodiment_gate_enabled", default_value="true"),
            DeclareLaunchArgument("embodiment_gate_xy_tolerance", default_value="0.025"),
            DeclareLaunchArgument("embodiment_gate_z_tolerance", default_value="0.020"),
            DeclareLaunchArgument("embodiment_gate_orientation_tolerance", default_value="0.22"),
            # Balanced Isaac motion defaults:
            # enough time to settle into the target without stretching every micro-correction into
            # 8-13 second motions that cause the agent to time out and lose embodiment accuracy.
            DeclareLaunchArgument("skill_max_velocity_scaling", default_value="0.10"),
            DeclareLaunchArgument("skill_max_acceleration_scaling", default_value="0.08"),
            DeclareLaunchArgument("skill_cartesian_pose_verification_enabled", default_value="true"),
            DeclareLaunchArgument("skill_cartesian_pose_correction_attempts", default_value="2"),
            DeclareLaunchArgument("skill_cartesian_pose_default_position_tolerance", default_value="0.04"),
            DeclareLaunchArgument("skill_cartesian_pose_default_orientation_tolerance", default_value="0.20"),
            DeclareLaunchArgument("skill_cartesian_pose_post_execute_settle_time", default_value="0.5"),
            DeclareLaunchArgument("skill_cartesian_pose_max_correction_step", default_value="0.03"),
            DeclareLaunchArgument("skill_cartesian_pose_min_position_improvement", default_value="0.003"),
            DeclareLaunchArgument("fjt_rate_hz", default_value="120.0"),
            DeclareLaunchArgument("fjt_extra_settle_time", default_value="1.0"),
            DeclareLaunchArgument("fjt_min_traj_duration", default_value="0.8"),
            # Isaac regularly lands within a few millimeters of the requested TCP pose while
            # leaving one joint a bit above 0.03 rad of residual error, so use a slightly
            # looser completion gate here and let the Cartesian verifier catch real misses.
            DeclareLaunchArgument("fjt_goal_tolerance_rad", default_value="0.05"),
            # Bring up the shared motion backend first.
            base_launch,
            # Then start the agent that consumes the backend.
            agent_node,
            # Finally add the optional GUI.
            gui_node,
        ]
    )
