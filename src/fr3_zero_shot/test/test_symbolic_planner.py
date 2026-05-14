from fr3_zero_shot.core.types import ApproachFamily, SkillName
from fr3_zero_shot.planning.symbolic_planner import SymbolicPlanner


def test_planner_keeps_llm_authority_symbolic_for_cup_place():
    plan = SymbolicPlanner().plan("pick up the blue cube and place it in the cup")

    assert plan.skill_sequence == [SkillName.PICK, SkillName.PLACE_IN]
    assert plan.source_ref == "blue cube"
    assert plan.target_ref == "cup"
    assert plan.relation == "in"
    assert plan.approach_preference == ApproachFamily.TOP_DOWN


def test_planner_prefers_side_for_explicit_avoidance():
    plan = SymbolicPlanner().plan("pick up the green cube while avoiding the cup")

    assert plan.skill_sequence == [SkillName.PICK]
    assert plan.source_ref == "green cube"
    assert plan.avoid_refs == ["cup"]
    assert plan.approach_preference == ApproachFamily.SIDE_LEFT

