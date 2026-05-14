from __future__ import annotations

from fr3_zero_shot.core.types import SkillOutcome


class TrialLogger:
    def __init__(self) -> None:
        self.outcomes: list[SkillOutcome] = []

    def record(self, outcome: SkillOutcome) -> None:
        self.outcomes.append(outcome)

