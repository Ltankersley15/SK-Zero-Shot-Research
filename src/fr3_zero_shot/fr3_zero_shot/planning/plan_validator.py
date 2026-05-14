from __future__ import annotations

from fr3_zero_shot.core.types import FailureCode, SkillName, SkillOutcome, SymbolicPlan


class PlanValidator:
    def validate(self, plan: SymbolicPlan) -> SkillOutcome:
        if SkillName.STOP in plan.skill_sequence:
            return SkillOutcome(False, FailureCode.UNSUPPORTED_COMMAND, plan.reason)
        if SkillName.PICK in plan.skill_sequence and not plan.source_ref:
            return SkillOutcome(False, FailureCode.AMBIGUOUS_REFERENCE, "missing source reference")
        if any(skill in plan.skill_sequence for skill in (SkillName.PLACE_IN, SkillName.PLACE_RELATIVE)):
            if not plan.target_ref:
                return SkillOutcome(False, FailureCode.AMBIGUOUS_REFERENCE, "missing target reference")
        return SkillOutcome(True, FailureCode.NONE, "plan accepted")

