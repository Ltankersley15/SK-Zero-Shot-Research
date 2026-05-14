from __future__ import annotations

from fr3_zero_shot.core.types import SkillOutcome


class FinalVerifier:
    def summarize(self, outcomes: list[SkillOutcome]) -> SkillOutcome:
        for outcome in outcomes:
            if not outcome.success:
                return outcome
        return outcomes[-1]

