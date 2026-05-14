from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ModuleNotFoundError:  # Allows parser unit tests without a sourced ROS environment.
    rclpy = None

    class Node:  # type: ignore[no-redef]
        pass

    class String:  # type: ignore[no-redef]
        def __init__(self, data: str = "") -> None:
            self.data = data


DEFAULT_PICK_TOLERANCE = ("--position-tolerance", "0.025", "--orientation-tolerance", "0.45")
GREEN_PICK_TOLERANCE = ("--position-tolerance", "0.035", "--orientation-tolerance", "0.55")


@dataclass(frozen=True)
class ResolvedLiveCommand:
    intent: str
    module: str
    args: tuple[str, ...]
    source_object: str = ""
    target_object: str = ""
    placement_mode: str = ""
    approach_preference: str = "auto"
    reasoning: str = ""

    def argv(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.args]


class LiveCommandRouter(Node):
    """Routes GUI natural-language commands to the live Isaac entry points."""

    def __init__(self) -> None:
        super().__init__("zero_shot_command_router")
        self.declare_parameter("command_topic", "/lvlm_agent/command")
        self.declare_parameter("result_topic", "/lvlm_agent/command_result")
        self.declare_parameter("reasoning_topic", "/lvlm_agent/reasoning_trace")

        command_topic = str(self.get_parameter("command_topic").value)
        result_topic = str(self.get_parameter("result_topic").value)
        reasoning_topic = str(self.get_parameter("reasoning_topic").value)

        self._result_pub = self.create_publisher(String, result_topic, 10)
        self._trace_pub = self.create_publisher(String, reasoning_topic, 10)
        self._sub = self.create_subscription(String, command_topic, self._on_command, 10)
        self._run_lock = threading.Lock()
        self._active_process: subprocess.Popen[str] | None = None

    def _on_command(self, msg: String) -> None:
        command = str(msg.data or "").strip()
        if not command:
            return
        with self._run_lock:
            if self._active_process is not None and self._active_process.poll() is None:
                self._publish_result("A live command is already running. Wait for it to finish before sending another.")
                return
        resolved = resolve_command(command)
        if resolved is None:
            self._publish_trace(
                {
                    "stage": "plan_structured",
                    "command": command,
                    "planner_source": "gui_command_router",
                    "planner_provider": "local_router",
                    "planner_model": "natural_language_dispatch",
                    "intent": "unsupported",
                    "reasoning_summary": "No supported live manipulation routine matched this command.",
                    "reasoning": "Use a supported pickup, cup placement, multi-cube cup, stack, or occluded green pickup request.",
                    "ts_sec": time.time(),
                }
            )
            self._publish_result("Command was not recognized by the live router.")
            return
        thread = threading.Thread(target=self._run_resolved, args=(command, resolved), daemon=True)
        thread.start()

    def _run_resolved(self, command: str, resolved: ResolvedLiveCommand) -> None:
        self._publish_trace(
            {
                "stage": "planner_call",
                "command": command,
                "text": "Routing GUI natural-language command to a live Isaac routine.",
                "planner_provider": "local_router",
                "planner_model": "natural_language_dispatch",
                "planner_call_index": 1,
                "planner_call_budget": 1,
                "planner_replan_count": 0,
                "ts_sec": time.time(),
            }
        )
        self._publish_trace(
            {
                "stage": "plan_structured",
                "command": command,
                "planner_source": "gui_command_router",
                "planner_provider": "local_router",
                "planner_model": "natural_language_dispatch",
                "planner_latency_sec": 0.0,
                "planner_call_index": 1,
                "planner_call_budget": 1,
                "planner_replan_count": 0,
                "intent": resolved.intent,
                "source_object": resolved.source_object,
                "target_object": resolved.target_object,
                "placement_mode": resolved.placement_mode,
                "approach_preference": resolved.approach_preference,
                "avoid_object_ids": ["cup"] if "cup" in resolved.target_object or resolved.intent == "green_occluded_pick" else [],
                "reasoning_summary": resolved.reasoning,
                "reasoning": resolved.reasoning,
                "raw_response_snippet": "local route selected from supported live routines",
                "ts_sec": time.time(),
            }
        )
        self._publish_trace(
            {
                "stage": "execution_mode",
                "command": command,
                "text": f"Running {resolved.module}",
                "execution_mode": "live_isaac",
                "ts_sec": time.time(),
            }
        )
        self._publish_result(f"Running: {command}")

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        started = time.time()
        output = ""
        returncode = 127
        try:
            process = subprocess.Popen(
                resolved.argv(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            with self._run_lock:
                self._active_process = process
            output, _ = process.communicate()
            returncode = int(process.returncode or 0)
        except Exception as exc:
            output = str(exc)
            returncode = 127
        finally:
            with self._run_lock:
                self._active_process = None

        elapsed = time.time() - started
        parsed = _extract_last_json(output)
        success = returncode == 0
        if isinstance(parsed, dict) and "success" in parsed:
            success = bool(parsed.get("success"))
        detail = _summarize_process_output(output, parsed=parsed, returncode=returncode)
        self._publish_trace(
            {
                "stage": "stage_timing",
                "command": command,
                "stage_name": "execution_total",
                "latency_sec": elapsed,
                "text": f"Live routine finished in {elapsed:.2f}s",
                "ts_sec": time.time(),
            }
        )
        self._publish_trace(
            {
                "stage": "result_detail",
                "command": command,
                "text": detail,
                "success": success,
                "returncode": returncode,
                "ts_sec": time.time(),
            }
        )
        self._publish_result("success" if success else f"failure: {detail}")

    def _publish_trace(self, payload: dict[str, Any]) -> None:
        self._trace_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _publish_result(self, text: str) -> None:
        self._result_pub.publish(String(data=str(text)))

    def destroy_node(self) -> bool:
        with self._run_lock:
            process = self._active_process
        if process is not None and process.poll() is None:
            process.terminate()
        return super().destroy_node()


def resolve_command(command: str) -> ResolvedLiveCommand | None:
    text = _normalize(command)
    if not text:
        return None

    if _mentions(text, "green") and (
        _mentions_any(text, ("occluded", "partially", "partial", "avoid", "cup", "blocked"))
    ):
        return ResolvedLiveCommand(
            intent="green_occluded_pick",
            module="fr3_zero_shot.live_green_occluded_pick",
            args=("--execute", "--llm-strategy", *GREEN_PICK_TOLERANCE),
            source_object="green cube",
            target_object="cup keepout",
            placement_mode="none",
            approach_preference="llm_strategy",
            reasoning=(
                "Green cube is described as occluded or near the cup, so use the green pickup routine "
                "with LLM strategy authority and the wider tolerances validated for that task."
            ),
        )

    if _is_stack_command(text):
        first, second = _stack_sources(text)
        return ResolvedLiveCommand(
            intent="stack_cubes",
            module="fr3_zero_shot.live_stack_cubes",
            args=("--execute", "--stack-sources", first, second, *DEFAULT_PICK_TOLERANCE),
            source_object=f"{first}, {second}",
            target_object="blue cube",
            placement_mode="stack",
            approach_preference="auto",
            reasoning=f"Command asks for cube stacking; stack {first} and then {second} on the blue base.",
        )

    if _mentions_any(text, ("cup", "container")):
        colors = _ordered_color_sequence(text)
        if len(colors) >= 2:
            return ResolvedLiveCommand(
                intent="multi_cube_cup_sequence",
                module="fr3_zero_shot.live_pick_place_sequence",
                args=("--execute", "--colors", *colors, *DEFAULT_PICK_TOLERANCE),
                source_object=", ".join(colors),
                target_object="cup",
                placement_mode="in",
                approach_preference="auto",
                reasoning=f"Command asks to place multiple cubes in the cup in this order: {', '.join(colors)}.",
            )
        if _mentions(text, "blue"):
            return ResolvedLiveCommand(
                intent="pick_place",
                module="fr3_zero_shot.live_pick_place",
                args=("--execute", *DEFAULT_PICK_TOLERANCE),
                source_object="blue cube",
                target_object="cup",
                placement_mode="in",
                approach_preference="auto",
                reasoning="Command asks for the calibrated blue-cube-in-cup live path.",
            )

    pick_color = _single_pick_color(text)
    if pick_color is not None and _mentions_any(text, ("pick", "pickup", "grab", "lift")):
        return ResolvedLiveCommand(
            intent="pick",
            module="fr3_zero_shot.live_pick",
            args=(f"pick up the {pick_color.replace('_', ' ')} cube", "--execute", *DEFAULT_PICK_TOLERANCE),
            source_object=f"{pick_color.replace('_', ' ')} cube",
            target_object="",
            placement_mode="none",
            approach_preference="auto",
            reasoning=f"Command asks for a visible {pick_color.replace('_', ' ')} cube pickup.",
        )

    return None


def _normalize(command: str) -> str:
    text = command.lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _mentions(text: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", text) is not None


def _mentions_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(_mentions(text, token) for token in tokens)


def _is_stack_command(text: str) -> bool:
    return _mentions_any(text, ("stack", "stacking")) or "on top of" in text or " on the " in text


def _ordered_color_sequence(text: str) -> list[str]:
    mentions: list[tuple[int, str]] = []
    for match in re.finditer(r"\b(?:large\s+red|small\s+red|red|blue|yellow|green)\b", text):
        raw = match.group(0)
        color = "red" if "red" in raw else raw
        mentions.append((match.start(), color))
    ordered: list[str] = []
    for _, color in sorted(mentions):
        if color not in ordered:
            ordered.append(color)
    return ordered


def _single_pick_color(text: str) -> str | None:
    if "small red" in text:
        return "small_red"
    if "large red" in text:
        return "large_red"
    for color in ("blue", "yellow", "green", "red"):
        if _mentions(text, color):
            return color
    return None


def _stack_sources(text: str) -> tuple[str, str]:
    has_yellow = _mentions(text, "yellow")
    has_small_red = "small red" in text
    has_large_red = "large red" in text
    if has_yellow and has_small_red:
        return ("yellow", "small_red")
    if has_large_red and has_small_red:
        return ("large_red", "small_red")
    if has_yellow and _mentions(text, "red"):
        return ("yellow", "small_red")
    if has_large_red and has_yellow:
        return ("large_red", "yellow")
    return ("large_red", "small_red")


def _extract_last_json(output: str) -> dict[str, Any] | None:
    text = str(output or "")
    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            best = value
    return best


def _summarize_process_output(
    output: str,
    *,
    parsed: dict[str, Any] | None,
    returncode: int,
) -> str:
    if isinstance(parsed, dict):
        outcome = parsed.get("outcome")
        if isinstance(outcome, dict):
            message = str(outcome.get("message", "") or "").strip()
            if message:
                return message
        if "success" in parsed:
            return f"live result success={bool(parsed.get('success'))} returncode={returncode}"
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    if not lines:
        return f"live routine exited with returncode={returncode}"
    tail = " | ".join(lines[-4:])
    if len(tail) > 500:
        tail = tail[-500:]
    return tail


def main(argv: list[str] | None = None) -> int:
    del argv
    if rclpy is None:
        raise RuntimeError("rclpy is required to run the live command router")
    rclpy.init()
    node = LiveCommandRouter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
