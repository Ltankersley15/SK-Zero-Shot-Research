from __future__ import annotations

from typing import Protocol

from fr3_zero_shot.core.types import SkillOutcome


class Skill(Protocol):
    def run(self, *args, **kwargs) -> SkillOutcome:
        ...

