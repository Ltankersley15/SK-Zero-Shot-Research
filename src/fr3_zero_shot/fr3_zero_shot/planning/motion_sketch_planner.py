from __future__ import annotations

import json
from typing import Any

from fr3_zero_shot.core.types import GraspIntent, MotionSketch, ReobserveDirective, RetreatIntent
from fr3_zero_shot.planning.strategy_planner import (
    ChatClient,
    FORBIDDEN_METRIC_KEYS,
    OpenAIStrategyChatClient,
    _extract_json_object,
    _first_forbidden_metric_key,
)


MOTION_SKETCH_FORBIDDEN_KEYS = {
    *FORBIDDEN_METRIC_KEYS,
    "joint",
    "joints",
    "joint_target",
    "joint_targets",
    "angle",
    "angles",
    "yaw",
    "yaw_deg",
    "roll",
    "pitch",
    "offset",
    "offsets",
    "offset_m",
    "distance",
    "distance_m",
}


class MotionSketchPlanner:
    def __init__(
        self,
        *,
        chat_client: ChatClient | None = None,
        model: str = "gpt-5-mini",
        timeout_sec: float = 20.0,
    ) -> None:
        self._chat_client = chat_client
        self._model = str(model)
        self._timeout_sec = float(timeout_sec)
        self.last_prompt: str = ""
        self.last_raw_response: str = ""
        self.last_rejection_reason: str | None = None

    def plan(
        self,
        *,
        command: str,
        phase: str,
        scene_facts: dict[str, Any],
        use_llm: bool = True,
    ) -> MotionSketch:
        self.last_prompt = self._build_prompt(command=command, phase=phase, scene_facts=scene_facts)
        self.last_raw_response = ""
        self.last_rejection_reason = None
        if not use_llm:
            return _fallback_sketch("deterministic motion-sketch fallback")

        errors: list[str] = []
        for _ in range(3):
            try:
                raw = self._client().chat(
                    [
                        {"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": self.last_prompt},
                    ],
                    max_tokens=900,
                    temperature=0.1,
                )
                self.last_raw_response = str(raw.get("content", ""))
                payload = _extract_json_object(self.last_raw_response)
                sketch = _sketch_from_payload(payload)
                self.last_rejection_reason = None
                return sketch
            except Exception as exc:
                errors.append(str(exc))
                self.last_rejection_reason = str(exc)
        return _rejected_sketch(f"llm motion sketch rejected: {'; '.join(errors)}")

    def _client(self) -> ChatClient:
        if self._chat_client is None:
            self._chat_client = OpenAIStrategyChatClient(model=self._model, timeout_sec=self._timeout_sec)
        return self._chat_client

    def _build_prompt(self, *, command: str, phase: str, scene_facts: dict[str, Any]) -> str:
        payload = {
            "command": str(command),
            "phase": str(phase),
            "scene_facts": scene_facts,
            "planning_guidance": [
                (
                    "If the source is near a cup or other keepout, avoid a generic top_down center sketch. "
                    "Reason about the exposed object edge or side away from the keepout, then ask deterministic "
                    "motion code to compile that sketch into safe poses."
                ),
                (
                    "For a partially occluded green cube beside a cup, a good sketch usually reobserves first, "
                    "then chooses outer_edge/top_edge or exposed_side with yaw_45_away_from_keepout, "
                    "low_lateral_entry, and pull_away_before_lift. The hand must not descend vertically over "
                    "the cube near the cup rim; descend only while outside the cup-risk area, enter laterally "
                    "at grasp height, pull laterally away from the cup, then lift."
                ),
                (
                    "Use require_side_grasp only when the top/edge route is impossible; otherwise describe the "
                    "desired exposed-side strategy and let deterministic IK/keepout checks choose the final family."
                ),
            ],
            "allowed_language": {
                "approach_direction": [
                    "top_down",
                    "top_edge",
                    "outer_edge",
                    "away_from_keepout",
                    "exposed_side",
                    "left",
                    "right",
                    "front",
                    "back",
                    "side_left",
                    "side_right",
                    "side_front",
                    "side_back",
                ],
                "grasp_region": ["center", "top_edge", "outer_edge", "exposed_face", "side_face"],
                "wrist_intent": [
                    "top_down",
                    "align_to_cube_edge",
                    "yaw_45_away_from_keepout",
                    "side_face",
                ],
                "retreat_direction": ["up", "away_from_keepout", "pull_away_then_lift", "toward_pregrasp"],
                "constraints": [
                    "avoid_cup",
                    "keepout_clearance",
                    "pull_away_before_lift",
                    "low_lateral_entry",
                    "fail_closed_if_unverified",
                    "require_side_grasp",
                ],
            },
            "required_response": {
                "reobserve": {
                    "required": "boolean",
                    "view_goal": "string or null; describe an object-relative view, not coordinates",
                    "target_ref": "string or null",
                    "reason": "short reason",
                },
                "grasp_intent": {
                    "approach_direction": "one allowed symbolic phrase",
                    "grasp_region": "one allowed symbolic phrase",
                    "wrist_intent": "one allowed symbolic phrase",
                    "contact_style": "short symbolic phrase",
                },
                "retreat_intent": {
                    "primary_direction": "one allowed symbolic phrase",
                    "secondary_direction": "string or null",
                    "lift_after_clearance": "boolean",
                },
                "constraints": ["symbolic constraints only"],
                "avoid_refs": ["object names only"],
                "allow_recovery": "boolean",
                "reason": "embodiment-aware explanation",
            },
        }
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are an embodied motion-sketch planner for a Franka FR3 tabletop robot. "
            "Write an object-relative manipulation sketch. You may choose reobserve goals, "
            "approach direction, grasp region, wrist intent, retreat direction, constraints, "
            "and bounded recovery. Return only valid JSON. Do not emit coordinates, poses, "
            "quaternions, trajectories, distances, angles, gripper widths, joint targets, or "
            "low-level motion commands. Keep the JSON compact."
        )


def _sketch_from_payload(payload: dict[str, Any]) -> MotionSketch:
    forbidden = _first_forbidden_metric_key_with_motion(payload)
    if forbidden is not None:
        raise ValueError(f"LLM emitted forbidden metric-control key '{forbidden}'")
    reobserve = payload.get("reobserve")
    grasp = payload.get("grasp_intent")
    retreat = payload.get("retreat_intent")
    if not isinstance(reobserve, dict):
        raise ValueError("motion sketch requires reobserve object")
    if not isinstance(grasp, dict):
        raise ValueError("motion sketch requires grasp_intent object")
    if not isinstance(retreat, dict):
        raise ValueError("motion sketch requires retreat_intent object")
    constraints = _string_list(payload.get("constraints", []), "constraints")
    avoid_refs = _string_list(payload.get("avoid_refs", []), "avoid_refs")
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise ValueError("motion sketch requires reason")
    return MotionSketch(
        reobserve=ReobserveDirective(
            required=bool(reobserve.get("required", False)),
            view_goal=_optional_text(reobserve.get("view_goal")),
            target_ref=_optional_text(reobserve.get("target_ref")),
            reason=str(reobserve.get("reason", "")).strip(),
        ),
        grasp_intent=GraspIntent(
            approach_direction=_required_text(grasp.get("approach_direction"), "approach_direction"),
            grasp_region=_required_text(grasp.get("grasp_region"), "grasp_region"),
            wrist_intent=_required_text(grasp.get("wrist_intent"), "wrist_intent"),
            contact_style=_required_text(grasp.get("contact_style"), "contact_style"),
        ),
        retreat_intent=RetreatIntent(
            primary_direction=_required_text(retreat.get("primary_direction"), "primary_direction"),
            secondary_direction=_optional_text(retreat.get("secondary_direction")),
            lift_after_clearance=bool(retreat.get("lift_after_clearance", True)),
        ),
        constraints=constraints,
        avoid_refs=avoid_refs,
        allow_recovery=bool(payload.get("allow_recovery", False)),
        reason=reason,
        planner_source="llm_motion_sketch",
    )


def _fallback_sketch(reason: str) -> MotionSketch:
    return MotionSketch(
        reobserve=ReobserveDirective(False, None, None, "source already observed"),
        grasp_intent=GraspIntent("top_edge", "top_edge", "yaw_45_away_from_keepout", "pinch"),
        retreat_intent=RetreatIntent("up", None, True),
        constraints=["keepout_clearance", "fail_closed_if_unverified"],
        avoid_refs=[],
        allow_recovery=False,
        reason=reason,
        planner_source="fallback_motion_sketch",
    )


def _rejected_sketch(reason: str) -> MotionSketch:
    return MotionSketch(
        reobserve=ReobserveDirective(False, None, None, "invalid LLM response"),
        grasp_intent=GraspIntent("fail_closed", "none", "none", "none"),
        retreat_intent=RetreatIntent("none", None, False),
        constraints=["fail_closed_if_unverified"],
        avoid_refs=[],
        allow_recovery=False,
        reason=reason,
        planner_source="rejected",
    )


def _first_forbidden_metric_key_with_motion(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in MOTION_SKETCH_FORBIDDEN_KEYS:
                return normalized
            found = _first_forbidden_metric_key_with_motion(child)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = _first_forbidden_metric_key_with_motion(child)
            if found is not None:
                return found
    return _first_forbidden_metric_key(value)


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        raise ValueError(f"motion sketch requires {field_name}")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return [str(item).strip().lower() for item in value if str(item).strip()]
