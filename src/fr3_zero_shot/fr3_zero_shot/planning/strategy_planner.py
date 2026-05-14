from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Any, Protocol

from fr3_zero_shot.core.types import ApproachFamily, StrategyDecision, StrategyOption


FORBIDDEN_METRIC_KEYS = {
    "x",
    "y",
    "z",
    "xyz",
    "pose",
    "poses",
    "position",
    "positions",
    "coordinate",
    "coordinates",
    "world_coordinate",
    "world_coordinates",
    "quat",
    "quaternion",
    "quaternions",
    "trajectory",
    "trajectories",
    "waypoint",
    "waypoints",
    "pregrasp",
    "grasp_xyz",
    "target_xyz",
    "release_z",
    "gripper_width",
    "gripper_command",
}


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        ...


class OpenAIStrategyChatClient:
    def __init__(self, *, model: str = "gpt-5-mini", timeout_sec: float = 20.0) -> None:
        from fr3_lvlm_agent.reasoning.openai_client import OpenAIClient

        self.model = str(model)
        self._openai_cls = OpenAIClient
        self._client = OpenAIClient(model=self.model, timeout=float(timeout_sec), max_tokens=600)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        token_limit = int(kwargs.get("max_tokens", 400))
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self._openai_cls.completion_token_kwargs(self.model, token_limit),
        }
        if self._openai_cls.uses_max_completion_tokens(self.model):
            request_kwargs["reasoning_effort"] = "low"
            request_kwargs["verbosity"] = "low"
        else:
            request_kwargs["temperature"] = float(kwargs.get("temperature", 0.1))
        response = self._client.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        return {"content": _coerce_content_text(content)}


class StrategyPlanner:
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
        options: list[StrategyOption],
        use_llm: bool = True,
    ) -> StrategyDecision:
        if not options:
            raise ValueError("strategy planner requires at least one option")
        self.last_rejection_reason = None
        self.last_prompt = self._build_prompt(command=command, phase=phase, scene_facts=scene_facts, options=options)
        if not use_llm:
            return self._fallback_decision(options, reason="deterministic strategy fallback")

        try:
            raw = self._client().chat(
                [
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": self.last_prompt},
                ],
                max_tokens=400,
                temperature=0.1,
            )
            self.last_raw_response = str(raw.get("content", ""))
            payload = _extract_json_object(self.last_raw_response)
            return self._decision_from_payload(payload, options)
        except Exception as exc:
            self.last_rejection_reason = str(exc)
            return self._fail_closed_decision(options, reason=f"llm strategy rejected: {exc}")

    def _client(self) -> ChatClient:
        if self._chat_client is None:
            self._chat_client = OpenAIStrategyChatClient(model=self._model, timeout_sec=self._timeout_sec)
        return self._chat_client

    def _decision_from_payload(
        self,
        payload: dict[str, Any],
        options: list[StrategyOption],
    ) -> StrategyDecision:
        forbidden = _first_forbidden_metric_key(payload)
        if forbidden is not None:
            raise ValueError(f"LLM emitted forbidden metric-control key '{forbidden}'")
        by_id = {option.option_id: option for option in options}
        option_id = str(payload.get("selected_option_id", "")).strip()
        if option_id not in by_id:
            raise ValueError(f"LLM selected unknown option_id '{option_id}'")
        option = by_id[option_id]
        approach = _parse_approach(payload.get("approach_preference"), option)
        reobserve_view = payload.get("reobserve_view", option.reobserve_view)
        if reobserve_view is not None:
            reobserve_view = str(reobserve_view)
        avoid_refs = payload.get("avoid_refs", option.avoid_refs)
        if not isinstance(avoid_refs, list):
            raise ValueError("avoid_refs must be a list")
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise ValueError("LLM strategy response requires reason")
        return StrategyDecision(
            selected_option_id=option.option_id,
            approach_preference=approach,
            reobserve_view=reobserve_view,
            avoid_refs=[str(ref) for ref in avoid_refs],
            allow_recovery=bool(payload.get("allow_recovery", option.allow_recovery)),
            reason=reason,
            planner_source="llm",
        )

    def _fallback_decision(self, options: list[StrategyOption], *, reason: str) -> StrategyDecision:
        option = next((item for item in options if item.option_id != "fail_closed"), options[0])
        return StrategyDecision(
            selected_option_id=option.option_id,
            approach_preference=option.approach_family,
            reobserve_view=option.reobserve_view,
            avoid_refs=list(option.avoid_refs),
            allow_recovery=bool(option.allow_recovery),
            reason=reason,
            planner_source="fallback",
        )

    def _fail_closed_decision(self, options: list[StrategyOption], *, reason: str) -> StrategyDecision:
        option = next((item for item in options if item.option_id == "fail_closed"), options[0])
        return StrategyDecision(
            selected_option_id=option.option_id,
            approach_preference=option.approach_family,
            reobserve_view=option.reobserve_view,
            avoid_refs=list(option.avoid_refs),
            allow_recovery=False,
            reason=reason,
            planner_source="rejected",
        )

    def _build_prompt(
        self,
        *,
        command: str,
        phase: str,
        scene_facts: dict[str, Any],
        options: list[StrategyOption],
    ) -> str:
        payload = {
            "command": str(command),
            "phase": str(phase),
            "scene_facts": scene_facts,
            "strategy_options": [_option_payload(option) for option in options],
            "required_response": {
                "selected_option_id": "must equal one offered option_id",
                "approach_preference": [item.value for item in ApproachFamily],
                "reobserve_view": "string or null; use a named offered view only",
                "avoid_refs": ["cup"],
                "allow_recovery": "boolean",
                "reason": "short embodiment-aware explanation",
            },
        }
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are the embodied strategy planner for a Franka FR3 tabletop robot. "
            "Choose exactly one offered strategy option. You may reason about reobserving, "
            "side access, edge access, avoiding a cup, and bounded recovery. "
            "Return only valid JSON. Do not emit coordinates, poses, quaternions, trajectories, "
            "gripper widths, or low-level motion commands."
        )


def _option_payload(option: StrategyOption) -> dict[str, Any]:
    data = asdict(option)
    data["approach_family"] = option.approach_family.value
    return data


def _parse_approach(value: Any, option: StrategyOption) -> ApproachFamily:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("approach_preference list must contain exactly one value")
        value = value[0]
    text = str(value or "").strip().lower()
    if not text:
        return option.approach_family
    if text == "side" and option.approach_family.value.startswith("side"):
        return option.approach_family
    if text == "edge":
        return ApproachFamily.TOP_DOWN_YAW_45
    try:
        return ApproachFamily(text)
    except ValueError as exc:
        raise ValueError(f"unsupported approach_preference '{text}'") from exc


def _extract_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if "```json" in value:
        value = value.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in value:
        value = value.split("```", 1)[1].split("```", 1)[0].strip()
    else:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if match:
            value = match.group(0)
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("strategy response must be a JSON object")
    return payload


def _first_forbidden_metric_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_METRIC_KEYS:
                return normalized
            found = _first_forbidden_metric_key(child)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = _first_forbidden_metric_key(child)
            if found is not None:
                return found
    return None


def _coerce_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif getattr(item, "text", None):
                parts.append(str(item.text))
        return "".join(parts)
    return str(content or "")
