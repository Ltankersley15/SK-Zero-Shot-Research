from __future__ import annotations

from fr3_zero_shot.core.types import SkillOutcome


def success_rate(outcomes: list[SkillOutcome]) -> float:
    if not outcomes:
        return 0.0
    return float(sum(1 for outcome in outcomes if outcome.success) / len(outcomes))

