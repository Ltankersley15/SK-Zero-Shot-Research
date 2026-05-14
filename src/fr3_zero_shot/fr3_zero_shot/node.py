from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from typing import Any

from fr3_zero_shot.affordance.affordance_scorer import AffordanceScorer
from fr3_zero_shot.core.types import FailureCode, SkillName, SkillOutcome
from fr3_zero_shot.execution.motion_adapter import DryRunMotionAdapter
from fr3_zero_shot.grasping.grasp_planner import GraspPlanner
from fr3_zero_shot.grounding.reference_grounder import ReferenceGrounder
from fr3_zero_shot.memory.scene_memory import SceneMemory
from fr3_zero_shot.perception.scene_census import SceneCensus
from fr3_zero_shot.planning.plan_validator import PlanValidator
from fr3_zero_shot.planning.symbolic_planner import SymbolicPlanner
from fr3_zero_shot.skills.pick import PickSkill
from fr3_zero_shot.skills.place_in_container import PlaceInContainerSkill


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    return str(value)


class ZeroShotOfflinePipeline:
    def __init__(self, *, motion: DryRunMotionAdapter | None = None) -> None:
        self.memory = SceneMemory()
        self.census = SceneCensus(self.memory)
        self.planner = SymbolicPlanner()
        self.validator = PlanValidator()
        self.grounder = ReferenceGrounder()
        self.affordance = AffordanceScorer()
        self.motion = DryRunMotionAdapter() if motion is None else motion
        self.grasp_planner = GraspPlanner()
        self.pick_skill = PickSkill(grasp_planner=self.grasp_planner, motion=self.motion)
        self.place_in_skill = PlaceInContainerSkill(motion=self.motion)

    def run(self, *, command: str, scene_path: str) -> dict[str, Any]:
        scene = self.census.from_json_file(scene_path)
        plan = self.planner.plan(command, scene)
        validation = self.validator.validate(plan)
        if not validation.success:
            return {"success": False, "plan": plan, "outcome": validation}
        grounded, grounding_outcome = self.grounder.ground(plan, scene.objects)
        if grounded is None:
            return {"success": False, "plan": plan, "outcome": grounding_outcome}
        source = self.memory.get(grounded.source_id)
        target = self.memory.get(grounded.target_id)
        avoid_objects = tuple(obj for obj in scene.objects if obj.object_id in set(grounded.avoid_ids))
        if source is None:
            return {"success": False, "plan": plan, "outcome": SkillOutcome(False, FailureCode.SOURCE_NOT_VISIBLE, "source missing")}

        outcomes: list[SkillOutcome] = []
        if SkillName.PICK in plan.skill_sequence:
            pick = self.pick_skill.run(
                source=source,
                avoid_objects=avoid_objects,
                approach_preference=plan.approach_preference,
            )
            outcomes.append(pick)
            if not pick.success:
                return {"success": False, "plan": plan, "grounded": grounded, "outcomes": outcomes}
        if SkillName.PLACE_IN in plan.skill_sequence:
            if target is None:
                outcome = SkillOutcome(False, FailureCode.TARGET_NOT_VISIBLE, "target missing")
            else:
                access = self.affordance.score(source=source, target=target, keepouts=avoid_objects).container_access
                outcome = self.place_in_skill.run(held_object=source, container=target, access=access)
            outcomes.append(outcome)
        return {
            "success": bool(outcomes and outcomes[-1].success),
            "plan": plan,
            "grounded": grounded,
            "outcomes": outcomes,
            "commands": self.motion.commands,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the zero-shot pipeline offline against a scene JSON file.")
    parser.add_argument("command", help="Natural-language command to plan.")
    parser.add_argument("--scene-json", required=True, help="Path to a scene JSON file with an objects list.")
    args = parser.parse_args(argv)

    result = ZeroShotOfflinePipeline().run(command=args.command, scene_path=args.scene_json)
    print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

