from fr3_zero_shot.core.types import ObjectHypothesis
from fr3_zero_shot.grounding.reference_grounder import ReferenceGrounder
from fr3_zero_shot.planning.symbolic_planner import SymbolicPlanner


def _obj(object_id, label, color, shape, xyz=(0.5, 0.0, 0.02), pickable=True, container=False):
    return ObjectHypothesis(
        object_id=object_id,
        label=label,
        color=color,
        shape=shape,
        xyz=xyz,
        footprint_xy=(0.04, 0.04),
        height_m=0.04,
        bbox_xyxy=(0, 0, 10, 10),
        confidence=0.9,
        visible=True,
        pickable=pickable,
        container_like=container,
    )


def test_grounder_rejects_ambiguous_generic_reference():
    plan = SymbolicPlanner().plan("pick up the cube")
    grounded, outcome = ReferenceGrounder().ground(
        plan,
        (
            _obj("blue", "blue cube", "blue", "cube"),
            _obj("green", "green cube", "green", "cube"),
        ),
    )

    assert grounded is None
    assert not outcome.success
    assert outcome.code.value == "ambiguous_reference"


def test_grounder_resolves_color_and_container_avoidance():
    plan = SymbolicPlanner().plan("pick up the green cube while avoiding the cup")
    grounded, outcome = ReferenceGrounder().ground(
        plan,
        (
            _obj("green", "green cube", "green", "cube"),
            _obj("cup", "cup", None, "cup", pickable=False, container=True),
        ),
    )

    assert outcome.success
    assert grounded is not None
    assert grounded.source_id == "green"
    assert grounded.avoid_ids == ["cup"]

