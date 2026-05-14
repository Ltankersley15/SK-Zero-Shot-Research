import json

from fr3_zero_shot.core.types import ApproachFamily, ObjectHypothesis, StrategyOption
from fr3_zero_shot.planning.green_strategy import (
    execution_green_strategy_options,
    green_recovery_allowed,
    initial_green_strategy_options,
)
from fr3_zero_shot.planning.strategy_planner import StrategyPlanner


class FakeChat:
    def __init__(self, payload):
        self.payload = payload

    def chat(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {"content": json.dumps(self.payload)}


def _option(option_id="edge_yawed_pick_away_from_cup"):
    return StrategyOption(
        option_id=option_id,
        description="test option",
        approach_family=ApproachFamily.TOP_DOWN_YAW_45,
        allow_recovery=True,
        avoid_refs=["cup"],
    )


def _obj(object_id, xyz, color, container):
    return ObjectHypothesis(
        object_id=object_id,
        label="cup" if container else f"{color} cube",
        color=color,
        shape="cup" if container else "cube",
        xyz=xyz,
        footprint_xy=(0.08, 0.08) if container else (0.025, 0.025),
        height_m=0.08 if container else 0.04,
        bbox_xyxy=(0, 0, 10, 10),
        confidence=1.0,
        visible=True,
        pickable=not container,
        container_like=container,
    )


def test_strategy_planner_accepts_known_option_id():
    planner = StrategyPlanner(
        chat_client=FakeChat(
            {
                "selected_option_id": "edge_yawed_pick_away_from_cup",
                "approach_preference": "top_down_yaw_45",
                "reobserve_view": None,
                "avoid_refs": ["cup"],
                "allow_recovery": True,
                "reason": "Reobserve showed the cup near the cube, so use edge access while avoiding the cup.",
            }
        )
    )

    decision = planner.plan(
        command="pick up the green cube while avoiding the cup",
        phase="post_reobserve_strategy",
        scene_facts={},
        options=[_option()],
    )

    assert decision.planner_source == "llm"
    assert decision.selected_option_id == "edge_yawed_pick_away_from_cup"
    assert decision.allow_recovery


def test_strategy_planner_accepts_single_item_approach_list():
    planner = StrategyPlanner(
        chat_client=FakeChat(
            {
                "selected_option_id": "reobserve_green_side",
                "approach_preference": ["side_right"],
                "reobserve_view": "green_reobserve",
                "avoid_refs": ["cup"],
                "allow_recovery": False,
                "reason": "Reobserve from the side because the cup partly occludes the green cube.",
            }
        )
    )

    option = StrategyOption(
        "reobserve_green_side",
        "side reobserve",
        ApproachFamily.SIDE_RIGHT,
        reobserve_view="green_reobserve",
        avoid_refs=["cup"],
    )
    decision = planner.plan(
        command="pick up the green cube while avoiding the cup",
        phase="initial_observation",
        scene_facts={},
        options=[option, _option("fail_closed")],
    )

    assert decision.planner_source == "llm"
    assert decision.selected_option_id == "reobserve_green_side"
    assert decision.approach_preference == ApproachFamily.SIDE_RIGHT


def test_strategy_planner_rejects_unknown_option_id():
    planner = StrategyPlanner(
        chat_client=FakeChat(
            {
                "selected_option_id": "invent_new_side_pose",
                "approach_preference": "side_right",
                "reason": "try a new side pose",
            }
        )
    )

    decision = planner.plan(command="pick", phase="test", scene_facts={}, options=[_option(), _option("fail_closed")])

    assert decision.planner_source == "rejected"
    assert decision.selected_option_id == "fail_closed"
    assert "unknown option_id" in planner.last_rejection_reason


def test_strategy_planner_rejects_metric_pose_authority():
    planner = StrategyPlanner(
        chat_client=FakeChat(
            {
                "selected_option_id": "edge_yawed_pick_away_from_cup",
                "approach_preference": "top_down_yaw_45",
                "pose": {"x": 0.6, "y": 0.0, "z": 0.05},
                "reason": "move here",
            }
        )
    )

    decision = planner.plan(command="pick", phase="test", scene_facts={}, options=[_option(), _option("fail_closed")])

    assert decision.planner_source == "rejected"
    assert decision.selected_option_id == "fail_closed"
    assert "forbidden metric-control" in planner.last_rejection_reason


def test_green_near_cup_strategy_options_include_reobserve_and_side_or_edge():
    green = _obj("green", xyz=(0.62, -0.02, 0.04), color="green", container=False)
    cup = _obj("cup", xyz=(0.50, -0.11, 0.06), color=None, container=True)

    initial = initial_green_strategy_options(green, cup)
    execution = execution_green_strategy_options(green, cup)

    assert {option.option_id for option in initial} >= {"reobserve_green_side", "edge_yawed_pick_away_from_cup"}
    assert "edge_yawed_pick_away_from_cup" in {option.option_id for option in execution}
    assert any(option.option_id.startswith("side_") for option in execution)


def test_deterministic_strategy_fallback_is_no_llm_only_path():
    decision = StrategyPlanner().plan(
        command="pick",
        phase="test",
        scene_facts={},
        options=[_option("reobserve_green_side"), _option("fail_closed")],
        use_llm=False,
    )

    assert decision.planner_source == "fallback"
    assert decision.selected_option_id == "reobserve_green_side"


def test_mocked_llm_can_choose_reobserve_then_edge_access():
    green = _obj("green", xyz=(0.62, -0.02, 0.04), color="green", container=False)
    cup = _obj("cup", xyz=(0.50, -0.11, 0.06), color=None, container=True)
    first = StrategyPlanner(
        chat_client=FakeChat(
            {
                "selected_option_id": "reobserve_green_side",
                "approach_preference": "side_back",
                "reobserve_view": "green_reobserve",
                "avoid_refs": ["cup"],
                "allow_recovery": False,
                "reason": "The cup partially blocks the green cube, so reobserve from the side before choosing a grasp.",
            }
        )
    )
    second = StrategyPlanner(
        chat_client=FakeChat(
            {
                "selected_option_id": "edge_yawed_pick_away_from_cup",
                "approach_preference": "top_down_yaw_45",
                "reobserve_view": None,
                "avoid_refs": ["cup"],
                "allow_recovery": True,
                "reason": "After reobserve, use the validated edge approach away from the cup and allow bounded recovery if separated.",
            }
        )
    )

    first_decision = first.plan(
        command="pick up the green cube while avoiding the cup",
        phase="initial_observation",
        scene_facts={},
        options=initial_green_strategy_options(green, cup),
    )
    second_decision = second.plan(
        command="pick up the green cube while avoiding the cup",
        phase="post_reobserve_strategy",
        scene_facts={},
        options=execution_green_strategy_options(green, cup),
    )

    assert first_decision.selected_option_id == "reobserve_green_side"
    assert second_decision.selected_option_id == "edge_yawed_pick_away_from_cup"
    assert second_decision.allow_recovery


def test_recovery_requires_llm_authorization_and_clearance():
    cup = _obj("cup", xyz=(0.50, -0.11, 0.06), color=None, container=True)
    separated_green = _obj("green", xyz=(0.65, 0.03, 0.04), color="green", container=False)
    near_green = _obj("green", xyz=(0.54, -0.08, 0.04), color="green", container=False)
    authorized = StrategyPlanner(chat_client=FakeChat({}))._fallback_decision(
        [_option("edge_yawed_pick_away_from_cup")],
        reason="test",
    )
    blocked = StrategyPlanner(chat_client=FakeChat({}))._fallback_decision(
        [StrategyOption("side_right_pull_away", "side", ApproachFamily.SIDE_RIGHT, allow_recovery=False)],
        reason="test",
    )

    assert green_recovery_allowed(authorized, separated_green, cup)
    assert not green_recovery_allowed(authorized, near_green, cup)
    assert not green_recovery_allowed(blocked, separated_green, cup)
