from __future__ import annotations

import json
import queue
import threading

from std_msgs.msg import String

from fr3_lvlm_agent.command_gui import LVLMCommandGUI


def _gui_like() -> LVLMCommandGUI:
    gui = LVLMCommandGUI.__new__(LVLMCommandGUI)
    gui._event_queue = queue.Queue(maxsize=256)
    gui._planning_start_by_command = {}
    gui._active_plan_lock = threading.Lock()
    gui._active_plan = {}
    gui._planner_status = {}
    gui._scene_snapshot = {}
    gui._execution_status = {}
    gui._result_status = "Awaiting command result..."
    gui._last_command = ""
    return gui


def test_plan_structured_updates_active_plan_lines() -> None:
    gui = _gui_like()

    payload = {
        "stage": "plan_structured",
        "command": "pick up the red cube and put it in the cup",
        "text": "intent=pick_and_place source='red cube' target='cup' placement=in planner=fallback",
        "intent": "pick_and_place",
        "source_object": "red cube",
        "target_object": "cup",
        "placement_mode": "in",
        "planner_source": "fallback",
        "planner_provider": "openai",
        "planner_model": "gpt-5-mini",
        "planner_latency_sec": 1.01,
        "planner_call_index": 1,
        "planner_call_budget": 2,
        "planner_replan_count": 0,
        "used_fallback": True,
        "planner_notes": "LLM returned no usable JSON plan; using deterministic command parser.",
        "reasoning": "Fallback parser matched pick-and-place command.",
        "reasoning_summary": "Planner used the deterministic fallback parser.",
        "raw_response_snippet": "",
        "source_pose_hint": None,
        "target_pose_hint": {"x": 0.565, "y": -0.004, "z": 0.020},
        "approach_preference": "side",
        "avoid_object_ids": ["scene_obj_002"],
        "scene_summary": "objects=[scene_obj_001:red cube@(0.44,0.12), scene_obj_002:cup@(0.56,-0.00)]",
        "scene_context": {
            "summary": "objects=[scene_obj_001:red cube@(0.44,0.12), scene_obj_002:cup@(0.56,-0.00)]",
            "objects": [
                {
                    "object_id": "scene_obj_001",
                    "canonical_label": "red cube",
                    "label": "red cube",
                    "xyz_base": (0.44, 0.12, 0.02),
                    "visible": True,
                },
                {
                    "object_id": "scene_obj_002",
                    "canonical_label": "cup",
                    "label": "cup",
                    "xyz_base": (0.56, -0.00, 0.02),
                    "visible": True,
                },
            ],
        },
        "ts_sec": 100.0,
    }

    gui._on_reasoning(String(data=json.dumps(payload)))

    lines = gui.active_plan_lines()
    assert any("Command: pick up the red cube and put it in the cup" == line for line in lines)
    assert any("Intent: pick_and_place" == line for line in lines)
    assert any("Target: cup" == line for line in lines)
    assert any("Target hint: (0.565, -0.004, 0.020)" == line for line in lines)
    assert any("Approach: side" == line for line in lines)

    planner_lines = gui.planner_lines()
    assert "Provider: openai" in planner_lines
    assert "Model: gpt-5-mini" in planner_lines
    assert "Planner source: fallback" in planner_lines
    assert "Latency: 1.01s" in planner_lines
    assert "Execution mode: (pending)" in planner_lines

    scene_lines = gui.scene_lines()
    assert "Summary: objects=[scene_obj_001:red cube@(0.44,0.12), scene_obj_002:cup@(0.56,-0.00)]" in scene_lines
    assert "red cube: near-left (x=0.44, y=0.12) [visible]" in scene_lines
    assert "cup: mid (x=0.56, y=-0.00) [visible]" in scene_lines


def test_gui_keeps_existing_plan_log_and_selected_scene_events() -> None:
    gui = _gui_like()

    gui._on_reasoning(
        String(
            data=json.dumps(
                {
                    "stage": "command",
                    "command": "pick up the red cube and put it in the cup",
                    "text": "accepted_for_execution",
                    "ts_sec": 10.0,
                }
            )
        )
    )
    gui._on_reasoning(
        String(
            data=json.dumps(
                {
                    "stage": "plan_reasoning",
                    "command": "pick up the red cube and put it in the cup",
                    "text": "LLM returned no usable JSON plan; using deterministic command parser.",
                    "ts_sec": 10.5,
                }
            )
        )
    )
    gui._on_reasoning(
        String(
            data=json.dumps(
                {
                    "stage": "planner_call",
                    "command": "pick up the red cube and put it in the cup",
                    "text": "provider=openai model=gpt-5-mini latency_sec=1.01 call=1/2 replans=0",
                    "ts_sec": 10.6,
                }
            )
        )
    )
    gui._on_reasoning(
        String(
            data=json.dumps(
                {
                    "stage": "plan_summary",
                    "command": "pick up the red cube and put it in the cup",
                    "text": "intent=pick_and_place source='red cube' target='cup' placement=in planner=fallback",
                    "ts_sec": 11.3,
                }
            )
        )
    )
    gui._on_reasoning(
        String(
            data=json.dumps(
                {
                    "stage": "scene",
                    "command": "pick up the red cube and put it in the cup",
                    "text": "place target_mode mode=embodied track=end_to_end",
                    "ts_sec": 12.0,
                }
            )
        )
    )
    gui._on_reasoning(
        String(
            data=json.dumps(
                {
                    "stage": "scene",
                    "command": "pick up the red cube and put it in the cup",
                    "text": "target planning_scene_obstacles=5",
                    "ts_sec": 12.1,
                }
            )
        )
    )
    gui._on_reasoning(
        String(
            data=json.dumps(
                {
                    "stage": "execution_mode",
                    "command": "pick up the red cube and put it in the cup",
                    "text": "mode=visible_fast_path source_visible=1 target_visible=1 placement=in",
                    "execution_mode": "visible_fast_path",
                    "ts_sec": 12.2,
                }
            )
        )
    )
    gui._on_reasoning(
        String(
            data=json.dumps(
                {
                    "stage": "stage_timing",
                    "command": "pick up the red cube and put it in the cup",
                    "text": "stage=scene_census latency_sec=0.24",
                    "stage_name": "scene_census",
                    "latency_sec": 0.24,
                    "ts_sec": 12.3,
                }
            )
        )
    )
    gui._on_reasoning(
        String(
            data=json.dumps(
                {
                    "stage": "place_safety_verdict",
                    "command": "pick up the red cube and put it in the cup",
                    "text": "verdict=safe reason=target='cup'",
                    "verdict": "safe",
                    "reason": "target='cup'",
                    "ts_sec": 12.4,
                }
            )
        )
    )
    gui._on_reasoning(
        String(
            data=json.dumps(
                {
                    "stage": "post_place_verdict",
                    "command": "pick up the red cube and put it in the cup",
                    "text": "verdict=in_cup reason=local_vl_confirmed confidence=0.92",
                    "verdict": "in_cup",
                    "reason": "local_vl_confirmed confidence=0.92",
                    "ts_sec": 12.5,
                }
            )
        )
    )

    events = gui.poll_events(max_items=16)

    assert "Planner: LLM returned no usable JSON plan; using deterministic command parser." in events
    assert "Planner: provider=openai model=gpt-5-mini latency_sec=1.01 call=1/2 replans=0" in events
    assert "Planner: intent=pick_and_place source='red cube' target='cup' placement=in planner=fallback" in events
    assert "Planner: command-to-plan time 1.30s" in events
    assert "Status: mode=visible_fast_path source_visible=1 target_visible=1 placement=in" in events
    assert "Status: stage=scene_census latency_sec=0.24" in events
    assert "Status: verdict=safe reason=target='cup'" in events
    assert "Status: verdict=in_cup reason=local_vl_confirmed confidence=0.92" in events
    assert all("planning_scene_obstacles=5" not in event for event in events)

    planner_lines = gui.planner_lines()
    assert "Execution mode: visible_fast_path" in planner_lines
    assert any("Stage timings: scene_census=0.24s" in line for line in planner_lines)

    result_lines = gui.result_lines()
    assert "Place safety: safe target='cup'" in result_lines
    assert "Post-place: in_cup local_vl_confirmed confidence=0.92" in result_lines
