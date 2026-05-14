from __future__ import annotations

import re

from fr3_zero_shot.core.types import FailureCode, GroundedPlan, ObjectHypothesis, SkillOutcome, SymbolicPlan


class ReferenceGrounder:
    def ground(self, plan: SymbolicPlan, objects: tuple[ObjectHypothesis, ...]) -> tuple[GroundedPlan | None, SkillOutcome]:
        source, source_status = self._resolve(plan.source_ref, objects, require_pickable=True)
        if plan.source_ref and source is None:
            return None, source_status
        target, target_status = self._resolve(plan.target_ref, objects, require_pickable=False)
        if plan.target_ref and target is None:
            return None, target_status

        avoid_ids: list[str] = []
        for avoid_ref in plan.avoid_refs:
            avoid, avoid_status = self._resolve(avoid_ref, objects, require_pickable=False)
            if avoid is None:
                return None, avoid_status
            avoid_ids.append(avoid.object_id)

        grounded = GroundedPlan(
            symbolic=plan,
            source_id=None if source is None else source.object_id,
            target_id=None if target is None else target.object_id,
            avoid_ids=avoid_ids,
        )
        return grounded, SkillOutcome(True, FailureCode.NONE, "references grounded")

    def _resolve(
        self,
        ref: str | None,
        objects: tuple[ObjectHypothesis, ...],
        *,
        require_pickable: bool,
    ) -> tuple[ObjectHypothesis | None, SkillOutcome]:
        if not ref:
            return None, SkillOutcome(True, FailureCode.NONE, "no reference")
        scored: list[tuple[float, ObjectHypothesis]] = []
        for obj in objects:
            if require_pickable and not obj.pickable:
                continue
            score = self._score(ref, obj)
            if score > 0.0:
                scored.append((score, obj))
        if not scored:
            return None, SkillOutcome(False, FailureCode.SOURCE_NOT_VISIBLE, f"no object matches '{ref}'")
        scored.sort(key=lambda item: (item[0], item[1].confidence), reverse=True)
        if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-6:
            return None, SkillOutcome(False, FailureCode.AMBIGUOUS_REFERENCE, f"ambiguous reference '{ref}'")
        return scored[0][1], SkillOutcome(True, FailureCode.NONE, "reference resolved")

    def _score(self, ref: str, obj: ObjectHypothesis) -> float:
        ref_tokens = self._tokens(ref)
        label_tokens = self._tokens(obj.label)
        if obj.color:
            label_tokens.add(obj.color.lower())
        if obj.shape:
            label_tokens.add(obj.shape.lower())
        if not ref_tokens:
            return 0.0
        overlap = len(ref_tokens & label_tokens)
        if overlap == 0:
            return 0.0
        required = len(ref_tokens - {"the", "a", "an"})
        return float(overlap) / max(1.0, float(required))

    def _tokens(self, text: str) -> set[str]:
        return {tok for tok in re.split(r"[^a-z0-9_]+", str(text).lower()) if tok and tok not in {"the", "a", "an"}}

