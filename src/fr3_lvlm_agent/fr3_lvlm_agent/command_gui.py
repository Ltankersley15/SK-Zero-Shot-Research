#!/usr/bin/env python3
from __future__ import annotations

import threading
import queue
import json
from collections import deque

import numpy as np
from PIL import Image, ImageTk

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import String

import tkinter as tk


class LVLMCommandGUI(Node):
    def __init__(self) -> None:
        super().__init__("lvlm_command_gui")
        self.declare_parameter("image_topic", "/lvlm_agent/debug_image")
        self.declare_parameter("cmd_topic", "/lvlm_agent/command")
        self.declare_parameter("result_topic", "/lvlm_agent/command_result")
        self.declare_parameter("reasoning_topic", "/lvlm_agent/reasoning_trace")
        self.declare_parameter("tx_queue_limit", 16)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.result_topic = str(self.get_parameter("result_topic").value)
        self.reasoning_topic = str(self.get_parameter("reasoning_topic").value)
        self.tx_queue_limit = max(1, int(self.get_parameter("tx_queue_limit").value))

        self.cmd_pub = self.create_publisher(String, self.cmd_topic, 10)
        self.image_sub = self.create_subscription(
            ROSImage,
            self.image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self.result_sub = self.create_subscription(String, self.result_topic, self._on_result, 10)
        self.reasoning_sub = self.create_subscription(String, self.reasoning_topic, self._on_reasoning, 10)
        self._tx_timer = self.create_timer(0.03, self._flush_tx_queue)

        self._img_queue: queue.Queue[Image.Image] = queue.Queue(maxsize=1)
        self._event_queue: queue.Queue[str] = queue.Queue(maxsize=256)
        self._last_command = ""
        self._tx_queue: deque[str] = deque()
        self._tx_lock = threading.Lock()
        self._planning_start_by_command: dict[str, float] = {}
        self._active_plan_lock = threading.Lock()
        self._active_plan: dict[str, object] = {}
        self._planner_status: dict[str, object] = {}
        self._scene_snapshot: dict[str, object] = {}
        self._execution_status: dict[str, object] = {}
        self._result_status = "Awaiting command result..."

    def _on_image(self, msg: ROSImage) -> None:
        try:
            img = self._ros_image_to_pil(msg)
            if img is None:
                return
            while not self._img_queue.empty():
                try:
                    self._img_queue.get_nowait()
                except queue.Empty:
                    break
            self._img_queue.put_nowait(img)
        except Exception as e:
            self.get_logger().warn(f"[GUI] image decode failed: {e}")

    def _ros_image_to_pil(self, msg: ROSImage) -> Image.Image | None:
        enc = (msg.encoding or "").lower()
        h = int(msg.height)
        w = int(msg.width)
        if h <= 0 or w <= 0:
            return None
        if enc == "rgb8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
            return Image.fromarray(arr, mode="RGB")
        if enc == "bgr8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
            arr = arr[:, :, ::-1]
            return Image.fromarray(arr, mode="RGB")
        if enc in {"mono8", "8uc1"}:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w))
            return Image.fromarray(arr, mode="L").convert("RGB")
        return None

    def _push_event(self, text: str) -> None:
        line = str(text or "").strip()
        if not line:
            return
        if self._event_queue.full():
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._event_queue.put_nowait(line)
        except queue.Full:
            pass

    def _normalize_result_text(self, text: str) -> str:
        line = str(text or "").strip()
        lowered = line.lower()
        if lowered == "success":
            return "Task completed successfully. Ready for the next prompt."
        if lowered == "failure":
            return "Task failed to complete. Ready for the next prompt."
        return line

    def _on_result(self, msg: String) -> None:
        line = self._normalize_result_text(str(msg.data or ""))
        if not line:
            return
        with self._active_plan_lock:
            self._result_status = line
        self._push_event(f"Result: {line}")

    def _on_reasoning(self, msg: String) -> None:
        raw = str(msg.data or "").strip()
        if not raw:
            return
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                stage = str(payload.get("stage", "reasoning")).strip()
                command = str(payload.get("command", "")).strip()
                text = str(payload.get("text", "")).strip()
                ts_sec = payload.get("ts_sec")
                try:
                    ts_sec = float(ts_sec)
                except (TypeError, ValueError):
                    ts_sec = None
                if stage == "plan_structured" and command:
                    self._update_active_plan(command=command, payload=payload)
                elif stage == "planner_call":
                    self._update_planner_status(payload=payload)
                elif stage == "planner_scene_context":
                    self._update_scene_snapshot(summary=text)
                elif stage == "plan_reasoning":
                    self._update_reasoning_summary(text)
                elif stage in {"execution_mode", "stage_timing", "place_safety_verdict", "post_place_verdict"}:
                    self._update_execution_status(stage=stage, payload=payload)
                elif stage in {"result_detail", "execution_error"}:
                    self._update_result_status(text)
                if text:
                    if stage == "command" and command and ts_sec is not None:
                        self._planning_start_by_command[command] = ts_sec
                    if self._should_display_trace_event(stage=stage, text=text):
                        prefix = self._trace_event_prefix(stage=stage)
                        self._push_event(f"{prefix}: {text}")
                    if stage == "plan_summary" and command and ts_sec is not None:
                        start_sec = self._planning_start_by_command.pop(command, None)
                        if start_sec is not None and ts_sec >= start_sec:
                            self._push_event(f"Planner: command-to-plan time {ts_sec - start_sec:.2f}s")
                    return
        except Exception:
            pass
        # Ignore non-JSON chatter to keep the UI focused.
        return

    def _should_display_trace_event(self, *, stage: str, text: str) -> bool:
        if stage in {
            "plan_reasoning",
            "plan_summary",
            "planner_call",
            "planner_scene_context",
            "scene_census_result",
            "execution_mode",
            "stage_timing",
            "place_safety_verdict",
            "post_place_verdict",
        }:
            return True
        if stage in {"result_detail", "execution_error"}:
            return True
        if stage == "scene":
            return text.startswith(("scene_memory_invalidated_after_contact ",))
        if stage != "survey":
            return False
        return text.startswith(
            (
                "place target_mode ",
                "place descent_retry_once ",
                "pick retry_after_carry_check ",
                "pick carry_verify_fail ",
                "pick visible_fast_path ",
            )
        )

    @staticmethod
    def _trace_event_prefix(*, stage: str) -> str:
        if stage in {"planner_call", "plan_reasoning", "plan_summary"}:
            return "Planner"
        if stage in {"planner_scene_context", "scene_census_result", "scene"}:
            return "Scene"
        if stage in {"execution_mode", "stage_timing", "place_safety_verdict", "post_place_verdict"}:
            return "Status"
        if stage in {"result_detail", "execution_error"}:
            return "Result"
        return "Event"

    def _update_active_plan(self, *, command: str, payload: dict[str, object]) -> None:
        active_plan = {
            "command": str(command),
            "planner_source": str(payload.get("planner_source", "") or ""),
            "planner_provider": str(payload.get("planner_provider", "") or ""),
            "planner_model": str(payload.get("planner_model", "") or ""),
            "planner_latency_sec": float(payload.get("planner_latency_sec", 0.0) or 0.0),
            "planner_call_index": int(payload.get("planner_call_index", 0) or 0),
            "planner_call_budget": int(payload.get("planner_call_budget", 0) or 0),
            "planner_replan_count": int(payload.get("planner_replan_count", 0) or 0),
            "intent": str(payload.get("intent", "") or ""),
            "source_object": str(payload.get("source_object", "") or ""),
            "target_object": str(payload.get("target_object", "") or ""),
            "placement_mode": str(payload.get("placement_mode", "") or ""),
            "source_pose_hint": payload.get("source_pose_hint"),
            "target_pose_hint": payload.get("target_pose_hint"),
            "planner_notes": str(payload.get("planner_notes", "") or ""),
            "reasoning": str(payload.get("reasoning", "") or ""),
            "reasoning_summary": str(payload.get("reasoning_summary", "") or ""),
            "raw_response_snippet": str(payload.get("raw_response_snippet", "") or ""),
            "approach_preference": str(payload.get("approach_preference", "") or ""),
            "avoid_object_ids": list(payload.get("avoid_object_ids", []) or []),
            "used_fallback": bool(payload.get("used_fallback", False)),
            "scene_context": dict(payload.get("scene_context", {}) or {}) if isinstance(payload.get("scene_context"), dict) else {},
            "scene_summary": str(payload.get("scene_summary", "") or ""),
        }
        with self._active_plan_lock:
            self._active_plan = active_plan
            self._planner_status = {
                "provider": active_plan["planner_provider"],
                "model": active_plan["planner_model"],
                "latency_sec": active_plan["planner_latency_sec"],
                "call_index": active_plan["planner_call_index"],
                "call_budget": active_plan["planner_call_budget"],
                "replans": active_plan["planner_replan_count"],
                "source": active_plan["planner_source"],
            }
            self._scene_snapshot = {
                "summary": active_plan["scene_summary"],
                "context": active_plan["scene_context"],
            }

    def _update_planner_status(self, payload: dict[str, object]) -> None:
        with self._active_plan_lock:
            status = dict(self._planner_status)
            status["provider"] = str(payload.get("planner_provider", "") or status.get("provider", ""))
            status["model"] = str(payload.get("planner_model", "") or status.get("model", ""))
            try:
                status["latency_sec"] = float(payload.get("planner_latency_sec", status.get("latency_sec", 0.0)) or 0.0)
            except (TypeError, ValueError):
                pass
            try:
                status["call_index"] = int(payload.get("planner_call_index", status.get("call_index", 0)) or 0)
                status["call_budget"] = int(payload.get("planner_call_budget", status.get("call_budget", 0)) or 0)
                status["replans"] = int(payload.get("planner_replan_count", status.get("replans", 0)) or 0)
            except (TypeError, ValueError):
                pass
            self._planner_status = status

    def _update_execution_status(self, *, stage: str, payload: dict[str, object]) -> None:
        with self._active_plan_lock:
            status = dict(self._execution_status)
            if stage == "execution_mode":
                status["mode"] = str(
                    payload.get("execution_mode", "") or payload.get("mode", "") or status.get("mode", "")
                )
            elif stage == "stage_timing":
                timings = dict(status.get("stage_timings", {}) or {})
                stage_name = str(payload.get("stage_name", "") or "").strip()
                latency = payload.get("latency_sec", None)
                if stage_name:
                    try:
                        timings[stage_name] = float(latency or 0.0)
                    except (TypeError, ValueError):
                        pass
                status["stage_timings"] = timings
            elif stage == "place_safety_verdict":
                status["place_safety_verdict"] = str(payload.get("verdict", "") or "")
                status["place_safety_reason"] = str(payload.get("reason", "") or "")
            elif stage == "post_place_verdict":
                status["post_place_verdict"] = str(payload.get("verdict", "") or "")
                status["post_place_reason"] = str(payload.get("reason", "") or "")
            self._execution_status = status

    def _update_scene_snapshot(self, summary: str) -> None:
        line = str(summary or "").strip()
        if not line:
            return
        with self._active_plan_lock:
            snapshot = dict(self._scene_snapshot)
            snapshot["summary"] = line
            self._scene_snapshot = snapshot

    def _update_reasoning_summary(self, text: str) -> None:
        line = str(text or "").strip()
        if not line:
            return
        with self._active_plan_lock:
            if self._active_plan:
                self._active_plan = {**self._active_plan, "reasoning_summary": line}

    def _update_result_status(self, text: str) -> None:
        line = str(text or "").strip()
        if not line:
            return
        with self._active_plan_lock:
            self._result_status = line

    def planner_lines(self) -> list[str]:
        with self._active_plan_lock:
            status = dict(self._planner_status)
            plan = dict(self._active_plan)
            execution = dict(self._execution_status)
        if not status and not plan:
            return ["Awaiting planner output..."]
        provider = str(status.get("provider", "") or plan.get("planner_provider", "") or "(unknown)")
        model = str(status.get("model", "") or plan.get("planner_model", "") or "(unknown)")
        latency_sec = float(status.get("latency_sec", plan.get("planner_latency_sec", 0.0)) or 0.0)
        call_index = int(status.get("call_index", plan.get("planner_call_index", 0)) or 0)
        call_budget = int(status.get("call_budget", plan.get("planner_call_budget", 0)) or 0)
        replans = int(status.get("replans", plan.get("planner_replan_count", 0)) or 0)
        planner_source = str(status.get("source", "") or plan.get("planner_source", "") or "(unknown)")
        reasoning_summary = str(plan.get("reasoning_summary", "") or "")
        execution_mode = str(execution.get("mode", "") or "(pending)")
        lines = [
            f"Provider: {provider}",
            f"Model: {model}",
            f"Planner source: {planner_source}",
            f"Latency: {latency_sec:.2f}s",
            f"Calls: {call_index or 1}/{call_budget or max(1, call_index or 1)} replans={replans}",
            f"Execution mode: {execution_mode}",
            f"Reasoning summary: {reasoning_summary or '(none)'}",
        ]
        timing_text = self._stage_timing_summary(execution.get("stage_timings", {}))
        if timing_text:
            lines.append(f"Stage timings: {timing_text}")
        return lines

    def scene_lines(self) -> list[str]:
        with self._active_plan_lock:
            snapshot = dict(self._scene_snapshot)
            plan = dict(self._active_plan)
        context = snapshot.get("context", {})
        summary = str(snapshot.get("summary", "") or plan.get("scene_summary", "") or "").strip()
        if not isinstance(context, dict):
            context = {}
        objects = [obj for obj in list(context.get("objects") or []) if isinstance(obj, dict)]
        if not summary and not objects:
            return ["Awaiting scene census..."]
        lines = [f"Summary: {summary or '(none)'}"]
        if not objects:
            lines.append("Objects: (none)")
            return lines
        for obj in objects[:10]:
            lines.append(self._format_scene_object_line(obj))
        return lines

    def result_lines(self) -> list[str]:
        with self._active_plan_lock:
            result_status = str(self._result_status or "")
            command = str(self._active_plan.get("command", "") or self._last_command or "")
            execution = dict(self._execution_status)
        place_safety_verdict = str(execution.get("place_safety_verdict", "") or "")
        place_safety_reason = str(execution.get("place_safety_reason", "") or "")
        post_place_verdict = str(execution.get("post_place_verdict", "") or "")
        post_place_reason = str(execution.get("post_place_reason", "") or "")
        return [
            f"Last command: {command or '(none)'}",
            f"Status: {result_status or 'Awaiting command result...'}",
            f"Place safety: {place_safety_verdict or '(pending)'} {place_safety_reason}".rstrip(),
            f"Post-place: {post_place_verdict or '(pending)'} {post_place_reason}".rstrip(),
        ]

    def active_plan_lines(self) -> list[str]:
        with self._active_plan_lock:
            plan = dict(self._active_plan)
        if not plan:
            return ["Awaiting a structured plan..."]
        lines = [
            f"Command: {plan.get('command', '')}",
            f"Intent: {plan.get('intent', '')}",
            f"Source: {plan.get('source_object', '') or '(none)'}",
            f"Target: {plan.get('target_object', '') or '(none)'}",
            f"Placement: {plan.get('placement_mode', '') or '(none)'}",
            f"Approach: {plan.get('approach_preference', '') or '(none)'}",
            f"Source hint: {self._format_pose_hint_line(plan.get('source_pose_hint'))}",
            f"Target hint: {self._format_pose_hint_line(plan.get('target_pose_hint'))}",
            f"Avoid ids: {', '.join(plan.get('avoid_object_ids', []) or []) or '(none)'}",
        ]
        return lines

    def planner_panel_lines(self) -> list[str]:
        lines = list(self.planner_lines())
        plan_lines = self.active_plan_lines()
        if plan_lines:
            lines.append("")
            lines.extend(plan_lines)
        return lines

    @staticmethod
    def _format_pose_hint_line(value: object) -> str:
        if not isinstance(value, dict):
            return "(none)"
        try:
            return f"({float(value['x']):.3f}, {float(value['y']):.3f}, {float(value['z']):.3f})"
        except Exception:
            return "(none)"

    def _format_scene_object_line(self, obj: dict[str, object]) -> str:
        label = str(obj.get("canonical_label") or obj.get("label") or obj.get("object_id") or "object").strip()
        location = self._object_location_text(obj)
        visible = "visible" if bool(obj.get("visible", True)) else "not visible"
        return f"{label}: {location} [{visible}]"

    @staticmethod
    def _stage_timing_summary(stage_timings: object) -> str:
        if not isinstance(stage_timings, dict):
            return ""
        preferred_order = [
            "scene_census",
            "planner",
            "grounding",
            "carry_verify",
            "place_resolve",
            "place_descent",
            "post_place_verify",
            "execution_total",
        ]
        parts: list[str] = []
        for key in preferred_order:
            if key not in stage_timings:
                continue
            try:
                parts.append(f"{key}={float(stage_timings[key]):.2f}s")
            except (TypeError, ValueError):
                continue
        return ", ".join(parts)

    @staticmethod
    def _object_location_text(obj: dict[str, object]) -> str:
        xyz = obj.get("xyz_base")
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            return "location unavailable"
        try:
            x = float(xyz[0])
            y = float(xyz[1])
        except (TypeError, ValueError):
            return "location unavailable"
        depth = "near" if x <= 0.46 else ("far" if x >= 0.62 else "mid")
        lateral = "left" if y >= 0.08 else ("right" if y <= -0.08 else "center")
        if lateral == "center":
            symbolic = depth
        else:
            symbolic = f"{depth}-{lateral}"
        return f"{symbolic} (x={x:.2f}, y={y:.2f})"

    def enqueue_command(self, text: str) -> bool:
        cmd = str(text or "").strip()
        if not cmd:
            return False
        with self._tx_lock:
            if len(self._tx_queue) >= self.tx_queue_limit:
                self._tx_queue.popleft()
            self._tx_queue.append(cmd)
        self._push_event(f"User: {cmd}")
        return True

    def _flush_tx_queue(self) -> None:
        pending: list[str] = []
        with self._tx_lock:
            while self._tx_queue:
                pending.append(self._tx_queue.popleft())
        for cmd in pending:
            self._last_command = cmd
            self.cmd_pub.publish(String(data=cmd))
            self.get_logger().info(f"[GUI] sent command: {cmd}")

    def poll_events(self, max_items: int = 64) -> list[str]:
        events: list[str] = []
        limit = max(1, int(max_items))
        for _ in range(limit):
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events


def main() -> None:
    rclpy.init()
    node = LVLMCommandGUI()

    root = tk.Tk()
    root.title("LVLM Command + Overlay")

    img_label = tk.Label(root)
    img_label.pack(side=tk.TOP, padx=8, pady=8)

    entry_frame = tk.Frame(root)
    entry_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))

    cmd_var = tk.StringVar()
    entry = tk.Entry(entry_frame, textvariable=cmd_var, width=60)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    status_var = tk.StringVar(value="Last command: (none)")
    status = tk.Label(root, textvariable=status_var)
    status.pack(side=tk.TOP, padx=8, pady=(0, 8))

    panels = tk.Frame(root)
    panels.pack(side=tk.TOP, fill=tk.BOTH, expand=False, padx=8, pady=(0, 8))

    scene_frame = tk.Frame(panels)
    scene_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
    scene_label = tk.Label(scene_frame, text="Scene Summary", anchor="w", font=("TkDefaultFont", 10, "bold"))
    scene_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
    scene_view = tk.Text(scene_frame, height=10, width=52, state=tk.DISABLED)
    scene_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    scene_view.tag_configure("line", spacing1=2, spacing3=2)

    planner_frame = tk.Frame(panels)
    planner_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
    planner_label = tk.Label(planner_frame, text="Planner + Plan", anchor="w", font=("TkDefaultFont", 10, "bold"))
    planner_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
    plan_view = tk.Text(planner_frame, height=14, width=68, state=tk.DISABLED)
    plan_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    plan_view.tag_configure("line", spacing1=2, spacing3=2)

    def send_command(*_args):
        text = cmd_var.get()
        if node.enqueue_command(text):
            status_var.set(f"Last command: {text.strip()}")
            cmd_var.set("")
        entry.focus_set()

    send_btn = tk.Button(entry_frame, text="Send", command=send_command)
    send_btn.pack(side=tk.RIGHT, padx=(8, 0))
    entry.bind("<Return>", send_command)
    root.bind("<KP_Enter>", send_command)
    entry.focus_set()

    result_label = tk.Label(root, text="Command Result", anchor="w", font=("TkDefaultFont", 10, "bold"))
    result_label.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

    result_view = tk.Text(root, height=3, width=120, state=tk.DISABLED)
    result_view.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
    result_view.tag_configure("line", spacing1=2, spacing3=2)

    terminal_label = tk.Label(root, text="Event Feed", anchor="w", font=("TkDefaultFont", 10, "bold"))
    terminal_label.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

    terminal = tk.Text(root, height=8, width=120, state=tk.DISABLED)
    terminal.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
    terminal.tag_configure("line", spacing1=2, spacing3=2)

    exec_thread_running = True
    exec_thread = None

    def spin_ros():
        nonlocal exec_thread_running
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            while exec_thread_running:
                executor.spin_once(timeout_sec=0.1)
        except ExternalShutdownException:
            pass
        finally:
            executor.shutdown()

    exec_thread = threading.Thread(target=spin_ros, daemon=True)
    exec_thread.start()

    tk_img = None

    def update_image():
        nonlocal tk_img
        try:
            img = node._img_queue.get_nowait()
            tk_img = ImageTk.PhotoImage(img)
            img_label.configure(image=tk_img)
        except queue.Empty:
            pass
        root.after(16, update_image)

    update_image()

    def update_scene_view():
        scene_lines = node.scene_lines()
        scene_view.configure(state=tk.NORMAL)
        scene_view.delete("1.0", tk.END)
        for line in scene_lines:
            scene_view.insert(tk.END, line + "\n", ("line",))
        scene_view.configure(state=tk.DISABLED)
        root.after(100, update_scene_view)

    update_scene_view()

    def update_plan_view():
        plan_lines = node.planner_panel_lines()
        plan_view.configure(state=tk.NORMAL)
        plan_view.delete("1.0", tk.END)
        for line in plan_lines:
            plan_view.insert(tk.END, line + "\n", ("line",))
        plan_view.configure(state=tk.DISABLED)
        root.after(100, update_plan_view)

    update_plan_view()

    def update_result_view():
        result_lines = node.result_lines()
        result_view.configure(state=tk.NORMAL)
        result_view.delete("1.0", tk.END)
        for line in result_lines:
            result_view.insert(tk.END, line + "\n", ("line",))
        result_view.configure(state=tk.DISABLED)
        root.after(100, update_result_view)

    update_result_view()

    def update_terminal():
        events = node.poll_events(max_items=64)
        if events:
            terminal.configure(state=tk.NORMAL)
            for line in events:
                terminal.insert(tk.END, line + "\n", ("line",))
            total_lines = int(terminal.index("end-1c").split(".")[0])
            if total_lines > 400:
                terminal.delete("1.0", f"{total_lines - 400}.0")
            terminal.see(tk.END)
            terminal.configure(state=tk.DISABLED)
        root.after(100, update_terminal)

    update_terminal()

    def on_close():
        nonlocal exec_thread_running
        exec_thread_running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

    if exec_thread is not None:
        exec_thread.join(timeout=1.0)

    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
