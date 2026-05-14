from __future__ import annotations

from fr3_zero_shot.core.types import FailureCode, ObjectHypothesis, SkillOutcome
from fr3_zero_shot.memory.scene_memory import SceneMemory


class ObserveSkill:
    def __init__(self, memory: SceneMemory) -> None:
        self._memory = memory

    def run(self, objects: list[ObjectHypothesis]) -> SkillOutcome:
        summary = self._memory.update(objects)
        return SkillOutcome(
            success=True,
            code=FailureCode.NONE,
            message=f"observed {len(summary.objects)} objects",
            data={"scene": summary},
        )

