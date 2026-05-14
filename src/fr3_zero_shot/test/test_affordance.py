from fr3_zero_shot.affordance.source_access import evaluate_source_access
from fr3_zero_shot.core.types import ObjectHypothesis


def _obj(object_id, xyz, footprint=(0.04, 0.04), container=False):
    return ObjectHypothesis(
        object_id=object_id,
        label=object_id,
        color=None,
        shape="cube",
        xyz=xyz,
        footprint_xy=footprint,
        height_m=0.04,
        bbox_xyxy=(0, 0, 10, 10),
        confidence=0.9,
        visible=True,
        pickable=not container,
        container_like=container,
    )


def test_source_access_marks_near_container_as_side_case():
    access = evaluate_source_access(
        _obj("cube", (0.5, 0.0, 0.02)),
        (_obj("cup", (0.58, 0.0, 0.04), footprint=(0.07, 0.07), container=True),),
    )

    assert not access.top_down_clear
    assert access.side_clear
    assert access.nearest_keepout_clearance_m < 0.05
