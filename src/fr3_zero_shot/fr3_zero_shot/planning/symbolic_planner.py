from __future__ import annotations

import re

from fr3_zero_shot.core.types import ApproachFamily, SceneSummary, SkillName, SymbolicPlan


class SymbolicPlanner:
    def plan(self, command: str, scene: SceneSummary | None = None) -> SymbolicPlan:
        text = " ".join(str(command or "").lower().split())
        if not text:
            return self._stop(command, "empty command")

        source_ref = self._extract_source_ref(text)
        target_ref = self._extract_target_ref(text)
        relation = self._extract_relation(text)
        avoid_refs = self._extract_avoid_refs(text)

        if target_ref and relation == "in":
            sequence = [SkillName.PICK, SkillName.PLACE_IN]
        elif target_ref:
            sequence = [SkillName.PICK, SkillName.PLACE_RELATIVE]
        elif "pick" in text or "grab" in text or "lift" in text:
            sequence = [SkillName.PICK]
        else:
            return self._stop(command, "unsupported command family")

        approach = self._choose_approach(text, source_ref, avoid_refs, scene)
        return SymbolicPlan(
            command=str(command),
            skill_sequence=sequence,
            source_ref=source_ref,
            target_ref=target_ref,
            relation=relation,
            approach_preference=approach,
            avoid_refs=avoid_refs,
            reason=self._reason(sequence, approach, avoid_refs, target_ref),
        )

    def _stop(self, command: str, reason: str) -> SymbolicPlan:
        return SymbolicPlan(
            command=str(command),
            skill_sequence=[SkillName.STOP],
            source_ref=None,
            target_ref=None,
            relation=None,
            approach_preference=ApproachFamily.AUTO,
            avoid_refs=[],
            reason=reason,
        )

    def _extract_source_ref(self, text: str) -> str | None:
        match = re.search(r"(?:pick up|pick|grab|lift)\s+(?:the\s+)?(.+?)(?:\s+(?:and|while|then|in|into|on|near|beside|left|right)\b|$)", text)
        if not match:
            return None
        ref = match.group(1).strip()
        ref = re.sub(r"^(?:object|thing)\s+", "", ref).strip()
        return ref or None

    def _extract_target_ref(self, text: str) -> str | None:
        patterns = [
            r"(?:in|into|inside)\s+(?:the\s+)?(.+?)(?:\s+then\b|$)",
            r"(?:on|onto)\s+(?:the\s+)?(.+?)(?:\s+then\b|$)",
            r"(?:near|beside|next to|left of|right of)\s+(?:the\s+)?(.+?)(?:\s+then\b|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip() or None
        return None

    def _extract_relation(self, text: str) -> str | None:
        if re.search(r"\b(?:in|into|inside)\b", text):
            return "in"
        if re.search(r"\b(?:on|onto)\b", text):
            return "on"
        if "left of" in text:
            return "left_of"
        if "right of" in text:
            return "right_of"
        if re.search(r"\b(?:near|beside|next to)\b", text):
            return "near"
        return None

    def _extract_avoid_refs(self, text: str) -> list[str]:
        match = re.search(r"(?:avoid|avoiding)\s+(?:the\s+)?(.+?)(?:\s+then\b|$)", text)
        if not match:
            return []
        return [match.group(1).strip()]

    def _choose_approach(
        self,
        text: str,
        source_ref: str | None,
        avoid_refs: list[str],
        scene: SceneSummary | None,
    ) -> ApproachFamily:
        if "side" in text:
            return ApproachFamily.SIDE_LEFT
        if avoid_refs or "avoid" in text:
            return ApproachFamily.SIDE_LEFT
        if source_ref and scene is not None:
            source_tokens = set(source_ref.split())
            for obj in scene.objects:
                obj_tokens = {str(obj.color or ""), str(obj.shape or ""), *obj.label.lower().split()}
                if source_tokens and source_tokens.issubset(obj_tokens):
                    rel = next((rel for rel in scene.relations if rel.object_id == obj.object_id), None)
                    if rel is not None and not rel.top_down_clear:
                        return ApproachFamily.SIDE_LEFT
        return ApproachFamily.TOP_DOWN

    def _reason(
        self,
        sequence: list[SkillName],
        approach: ApproachFamily,
        avoid_refs: list[str],
        target_ref: str | None,
    ) -> str:
        if avoid_refs:
            return f"use {approach.value} because the command names an object to avoid"
        if SkillName.PLACE_IN in sequence:
            return f"pick source, then use container placement for {target_ref}"
        return f"use {approach.value} for a simple visible pick"

