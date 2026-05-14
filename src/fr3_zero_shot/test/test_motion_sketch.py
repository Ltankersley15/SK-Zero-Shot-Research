import json

from fr3_zero_shot.core.types import ApproachFamily, ObjectHypothesis
from fr3_zero_shot.grasping.grasp_planner import GraspPlanner
from fr3_zero_shot.planning.motion_sketch_compiler import MotionSketchCompiler
from fr3_zero_shot.planning.motion_sketch_planner import MotionSketchPlanner


class FakeChat:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        payload = self.payloads.pop(0)
        if isinstance(payload, str):
            return {"content": payload}
        return {"content": json.dumps(payload)}


def _obj(object_id, xyz, color=None, container=False):
    return ObjectHypothesis(
        object_id=object_id,
        label="cup" if container else f"{color or object_id} cube",
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


def _valid_payload(**overrides):
    payload = {
        "reobserve": {
            "required": True,
            "view_goal": "side view of the exposed green face",
            "target_ref": "green cube",
            "reason": "cup partially occludes the source",
        },
        "grasp_intent": {
            "approach_direction": "away_from_keepout",
            "grasp_region": "outer_edge",
            "wrist_intent": "yaw_45_away_from_keepout",
            "contact_style": "pinch",
        },
        "retreat_intent": {
            "primary_direction": "away_from_keepout",
            "secondary_direction": "up",
            "lift_after_clearance": True,
        },
        "constraints": ["avoid_cup", "keepout_clearance", "pull_away_before_lift"],
        "avoid_refs": ["cup"],
        "allow_recovery": True,
        "reason": "Reobserve, then grasp the exposed outer edge while avoiding the cup.",
    }
    payload.update(overrides)
    return payload


def test_motion_sketch_planner_accepts_valid_relative_sketch():
    planner = MotionSketchPlanner(chat_client=FakeChat(_valid_payload()))

    sketch = planner.plan(command="pick green", phase="test", scene_facts={})

    assert sketch.planner_source == "llm_motion_sketch"
    assert sketch.reobserve.required
    assert sketch.grasp_intent.approach_direction == "away_from_keepout"
    assert sketch.allow_recovery


def test_motion_sketch_planner_rejects_raw_metric_pose():
    planner = MotionSketchPlanner(
        chat_client=FakeChat(
            _valid_payload(pose={"x": 0.6}),
            _valid_payload(pose={"x": 0.6}),
            _valid_payload(pose={"x": 0.6}),
        )
    )

    sketch = planner.plan(command="pick green", phase="test", scene_facts={})

    assert sketch.planner_source == "rejected"
    assert "forbidden metric-control" in planner.last_rejection_reason


def test_motion_sketch_planner_retries_empty_response_once():
    planner = MotionSketchPlanner(chat_client=FakeChat("", _valid_payload()))

    sketch = planner.plan(command="pick green", phase="test", scene_facts={})

    assert planner._chat_client.calls == 2
    assert sketch.planner_source == "llm_motion_sketch"


def test_motion_sketch_planner_retries_empty_response_twice():
    planner = MotionSketchPlanner(chat_client=FakeChat("", "", _valid_payload()))

    sketch = planner.plan(command="pick green", phase="test", scene_facts={})

    assert planner._chat_client.calls == 3
    assert sketch.planner_source == "llm_motion_sketch"


def test_motion_sketch_prompt_gives_near_keepout_headroom_without_coordinates():
    planner = MotionSketchPlanner(chat_client=FakeChat(_valid_payload()))

    sketch = planner.plan(command="pick green", phase="post_reobserve_strategy", scene_facts={"near_keepout": True})

    assert sketch.planner_source == "llm_motion_sketch"
    assert "outer_edge/top_edge or exposed_side" in planner.last_prompt
    assert "must not descend vertically over" in planner.last_prompt
    assert "enter laterally at grasp height" in planner.last_prompt
    assert "coordinates" in planner.last_prompt


def test_motion_sketch_compiler_prefers_yawed_edge_near_cup():
    green = _obj("green", (0.62, 0.0, 0.04), "green")
    cup = _obj("cup", (0.50, -0.10, 0.06), container=True)
    sketch = MotionSketchPlanner(chat_client=FakeChat(_valid_payload())).plan(
        command="pick green",
        phase="test",
        scene_facts={},
    )

    result = MotionSketchCompiler(grasp_planner=GraspPlanner(top_down_x_bias_m=0.0)).compile(
        sketch=sketch,
        source=green,
        avoid_objects=(cup,),
    )

    assert result.candidate is not None
    assert result.candidate.family == ApproachFamily.TOP_DOWN_YAW_45
    assert any("yawed_edge_intent" in item.reason for item in result.trace)


def test_side_sketch_is_penalized_unless_side_is_required():
    green = _obj("green", (0.62, 0.0, 0.04), "green")
    cup = _obj("cup", (0.50, -0.10, 0.06), container=True)
    side_payload = _valid_payload(
        grasp_intent={
            "approach_direction": "side_right",
            "grasp_region": "side_face",
            "wrist_intent": "side_face",
            "contact_style": "side pinch",
        },
        constraints=["avoid_cup", "keepout_clearance"],
    )
    sketch = MotionSketchPlanner(chat_client=FakeChat(side_payload)).plan(command="pick green", phase="test", scene_facts={})

    risky = MotionSketchCompiler(grasp_planner=GraspPlanner(top_down_x_bias_m=0.0)).compile(
        sketch=sketch,
        source=green,
        avoid_objects=(cup,),
    )
    required = MotionSketchCompiler(grasp_planner=GraspPlanner(top_down_x_bias_m=0.0)).compile(
        sketch=MotionSketchPlanner(chat_client=FakeChat({**side_payload, "constraints": ["require_side_grasp"]})).plan(
            command="pick green",
            phase="test",
            scene_facts={},
        ),
        source=green,
        avoid_objects=(cup,),
    )

    assert risky.candidate is not None
    assert risky.candidate.family == ApproachFamily.TOP_DOWN_YAW_45
    assert required.candidate is not None
    assert required.candidate.family.value.startswith("side")
