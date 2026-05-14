"""Task-planning safety tests for command decomposition."""

from fr3_lvlm_agent.planning.task_planner import LLMTaskPlanner, TaskPlan, SubTask, TaskType


def test_spin_command_maps_to_safe_motion_action():
    """Non-grasp commands should map to safe executable motion primitives."""
    planner = LLMTaskPlanner(llm_reasoner=None)
    plan = planner.create_plan("spin in a circle")

    assert plan.task_type == TaskType.MOVE
    assert len(plan.sub_tasks) == 1
    assert plan.sub_tasks[0].task_type == TaskType.MOVE
    assert plan.sub_tasks[0].action == "spin_in_place"
    assert int(plan.sub_tasks[0].action_params.get("turns", 0)) >= 1


def test_unknown_command_is_not_defaulted_to_pick():
    """Ambiguous commands must not silently default to pick."""
    planner = LLMTaskPlanner(llm_reasoner=None)
    plan = planner.create_plan("do something cool")

    assert plan.task_type == TaskType.INSPECT
    assert plan.sub_tasks == []


def test_pick_command_still_generates_executable_subtask():
    """Core manipulation requests remain executable."""
    planner = LLMTaskPlanner(llm_reasoner=None)
    plan = planner.create_plan("pick up the red cube")

    assert plan.sub_tasks
    pick_steps = [st for st in plan.sub_tasks if st.task_type == TaskType.PICK]
    assert pick_steps


def test_spin_command_sanitizes_bad_llm_pick_plan():
    """LLM output is trusted without sanitization overrides."""
    planner = LLMTaskPlanner(llm_reasoner=object())

    def _bad_llm_plan(_command: str) -> TaskPlan:
        return TaskPlan(
            original_command="spin in a circle",
            task_type=TaskType.PICK,
            sub_tasks=[
                SubTask(
                    task_type=TaskType.PICK,
                    description="Pick up random object",
                    target_object="object",
                )
            ],
            llm_reasoning="incorrectly mapped spin to pick",
        )

    planner._create_plan_with_llm = _bad_llm_plan  # type: ignore[method-assign]
    plan = planner.create_plan("spin in a circle")

    assert plan.task_type == TaskType.PICK
    assert len(plan.sub_tasks) == 1
    assert plan.sub_tasks[0].task_type == TaskType.PICK


def test_rule_based_pick_preserves_object_descriptor():
    """Fallback planning should keep command object qualifiers for perception."""
    planner = LLMTaskPlanner(llm_reasoner=None)
    plan = planner.create_plan("pick up the red block")

    pick_steps = [st for st in plan.sub_tasks if st.task_type == TaskType.PICK]
    assert pick_steps
    assert pick_steps[0].target_object == "red block"


def test_inspect_command_falls_back_to_executable_survey_action() -> None:
    planner = LLMTaskPlanner(llm_reasoner=object())

    def _empty_llm_plan(_command: str) -> TaskPlan:
        return TaskPlan(
            original_command="inspect the green cube",
            task_type=TaskType.INSPECT,
            sub_tasks=[],
            llm_reasoning="returned no executable subtasks",
        )

    planner._create_plan_with_llm = _empty_llm_plan  # type: ignore[method-assign]
    plan = planner.create_plan("inspect the green cube")

    assert plan.task_type == TaskType.INSPECT
    assert len(plan.sub_tasks) == 1
    st = plan.sub_tasks[0]
    assert st.task_type == TaskType.INSPECT
    assert st.action == "survey_scene"
    assert st.action_params.get("description") == "inspect the green cube"


def test_directional_move_command_maps_to_shift_lateral_action():
    planner = LLMTaskPlanner(llm_reasoner=None)
    plan = planner.create_plan("move left")

    assert plan.task_type == TaskType.MOVE
    assert len(plan.sub_tasks) == 1
    st = plan.sub_tasks[0]
    assert st.task_type == TaskType.MOVE
    assert st.action == "shift_lateral"
    assert st.action_params.get("direction") == "left"
    assert float(st.action_params.get("distance_m", 0.0)) > 0.0


def test_llm_parse_coerces_non_executable_subtask_types():
    """LLM task_type drift should be preserved for the executor to attempt."""
    planner = LLMTaskPlanner(llm_reasoner=object())
    llm_result = {
        "task_type": "inspect",
        "sub_tasks": [
            {"task_type": "action", "description": "open gripper", "action": "open_gripper"},
            {"task_type": "stack", "description": "pick up red block", "target_object": "red block"},
        ],
        "reasoning": "normalize action and stack subtasks",
    }
    plan = planner._parse_llm_plan("pick up red block", llm_result)

    assert len(plan.sub_tasks) == 2
    assert plan.sub_tasks[0].task_type == TaskType.UNKNOWN
    assert plan.sub_tasks[1].task_type == TaskType.STACK


def test_pick_intent_with_inspect_only_llm_plan_gets_pick_step_repaired():
    """A direct pick command should collapse to a deterministic pick step."""
    planner = LLMTaskPlanner(llm_reasoner=object())

    def _inspect_only_pick_plan(_command: str) -> TaskPlan:
        return TaskPlan(
            original_command='pick up the red cube',
            task_type=TaskType.PICK,
            sub_tasks=[
                SubTask(
                    task_type=TaskType.INSPECT,
                    description='Inspect the area for the red cube',
                    action='survey_scene',
                    action_params={'description': 'locate red cube before grasping'},
                )
            ],
            llm_reasoning='inspect first, but omitted the pick step',
        )

    planner._create_plan_with_llm = _inspect_only_pick_plan  # type: ignore[method-assign]
    plan = planner.create_plan('pick up the red cube')

    assert len(plan.sub_tasks) == 1
    assert plan.sub_tasks[0].task_type == TaskType.PICK
    assert plan.sub_tasks[0].target_object == 'red cube'
    assert "Bypassed LLM planning and planner-level survey insertion for a direct pick command." in (
        plan.llm_reasoning or ""
    )


def test_pick_plan_prunes_redundant_survey_when_target_already_visible():
    planner = LLMTaskPlanner(llm_reasoner=object())
    planner.set_scene_context(
        {
            "scene": {
                "visible_objects": [
                    {"label": "red cube", "color": "red", "shape": "cube"},
                ]
            }
        }
    )

    def _survey_then_pick(_command: str) -> TaskPlan:
        return TaskPlan(
            original_command='pick up the red cube',
            task_type=TaskType.PICK,
            sub_tasks=[
                SubTask(
                    task_type=TaskType.INSPECT,
                    description='Inspect the area for the red cube',
                    action='survey_scene',
                    action_params={'description': 'locate red cube before grasping'},
                ),
                SubTask(
                    task_type=TaskType.PICK,
                    description='Pick up the red cube',
                    target_object='red cube',
                ),
            ],
            llm_reasoning='inspect first, then pick',
        )

    planner._create_plan_with_llm = _survey_then_pick  # type: ignore[method-assign]
    plan = planner.create_plan('pick up the red cube')

    assert len(plan.sub_tasks) == 1
    assert plan.sub_tasks[0].task_type == TaskType.PICK
    assert plan.sub_tasks[0].target_object == 'red cube'


def test_direct_pick_command_bypasses_llm_survey_plan():
    planner = LLMTaskPlanner(llm_reasoner=object())
    planner.set_scene_context(
        {
            "scene": {
                "visible_objects": [
                    {"label": "red cube", "color": "red", "shape": "cube"},
                ]
            }
        }
    )

    def _survey_then_pick(_command: str) -> TaskPlan:
        return TaskPlan(
            original_command='pick up the red cube',
            task_type=TaskType.PICK,
            sub_tasks=[
                SubTask(
                    task_type=TaskType.INSPECT,
                    description='Survey the workspace to locate red cube',
                    action='survey_scene',
                    action_params={'description': 'locate red cube before grasping'},
                ),
                SubTask(
                    task_type=TaskType.PICK,
                    description='Pick up the red cube',
                    target_object='red cube',
                ),
            ],
            llm_reasoning='inspect first, then pick',
        )

    planner._create_plan_with_llm = _survey_then_pick  # type: ignore[method-assign]
    plan = planner.create_plan('pick up the red cube')

    assert len(plan.sub_tasks) == 1
    assert plan.sub_tasks[0].task_type == TaskType.PICK
    assert plan.sub_tasks[0].target_object == 'red cube'
    assert "Bypassed LLM planning and planner-level survey insertion for a direct pick command." in (
        plan.llm_reasoning or ""
    )


def test_direct_pick_command_skips_planner_survey_when_target_not_in_scene_cache():
    planner = LLMTaskPlanner(llm_reasoner=object())
    planner.set_scene_context(
        {
            "scene": {
                "visible_objects": [
                    {"label": "background red panel", "color": "red", "shape": "panel"},
                ]
            }
        }
    )

    def _survey_then_pick(_command: str) -> TaskPlan:
        return TaskPlan(
            original_command='pick up the red cube',
            task_type=TaskType.PICK,
            sub_tasks=[
                SubTask(
                    task_type=TaskType.INSPECT,
                    description='Survey the workspace to locate red cube',
                    action='survey_scene',
                    action_params={'description': 'locate red cube before grasping'},
                ),
                SubTask(
                    task_type=TaskType.PICK,
                    description='Pick up the red cube',
                    target_object='red cube',
                ),
            ],
            llm_reasoning='inspect first, then pick',
        )

    planner._create_plan_with_llm = _survey_then_pick  # type: ignore[method-assign]
    plan = planner.create_plan('pick up the red cube')

    assert len(plan.sub_tasks) == 1
    assert plan.sub_tasks[0].task_type == TaskType.PICK
    assert plan.sub_tasks[0].target_object == 'red cube'
