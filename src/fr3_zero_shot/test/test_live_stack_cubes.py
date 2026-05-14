import pytest

from fr3_zero_shot.core.types import ApproachFamily, GraspCandidate, ObjectHypothesis

pytest.importorskip("rclpy")

from fr3_zero_shot.live_stack_cubes import (
    _candidate_with_xy_offset,
    _prefer_yawed_top_down_for_stack,
    _select_stack_source,
)


def _obj(object_id, *, color, xyz=(0.5, 0.0, 0.04), area_px=1000):
    return ObjectHypothesis(
        object_id=object_id,
        label=f"{color} cube",
        color=color,
        shape="cube",
        xyz=xyz,
        footprint_xy=(0.025, 0.025),
        height_m=0.04,
        bbox_xyxy=(0, 0, 20, 20),
        confidence=1.0,
        visible=True,
        pickable=True,
        container_like=False,
        metadata={"area_px": area_px},
    )


def test_select_stack_source_supports_yellow_then_small_red():
    yellow = _obj("yellow", color="yellow", area_px=6000)
    large_red = _obj("large_red", color="red", xyz=(0.4, 0.2, 0.04), area_px=9000)
    small_red = _obj("small_red", color="red", xyz=(0.4, -0.25, 0.04), area_px=2500)

    objects = [small_red, yellow, large_red]

    assert _select_stack_source(objects, "yellow") == yellow
    assert _select_stack_source(objects, "small_red") == small_red


def test_stack_prefers_yawed_top_down_alignment():
    plain = _candidate(ApproachFamily.TOP_DOWN)
    yawed = _candidate(ApproachFamily.TOP_DOWN_YAW_45)

    assert _prefer_yawed_top_down_for_stack(plain, [plain, yawed]) == yawed


def test_stack_candidate_xy_nudge_preserves_orientation():
    candidate = _candidate(ApproachFamily.TOP_DOWN_YAW_45)

    nudged = _candidate_with_xy_offset(candidate, dx=-0.012, dy=0.006)

    assert nudged.grasp_xyz == pytest.approx((0.488, 0.006, 0.045))
    assert nudged.pregrasp_xyz == pytest.approx((0.488, 0.006, 0.15))
    assert nudged.retreat_xyz == pytest.approx((0.488, 0.006, 0.25))
    assert nudged.quat_xyzw == candidate.quat_xyzw


def _candidate(family):
    return GraspCandidate(
        family=family,
        pregrasp_xyz=(0.5, 0.0, 0.15),
        grasp_xyz=(0.5, 0.0, 0.045),
        retreat_xyz=(0.5, 0.0, 0.25),
        quat_xyzw=(1.0, 0.0, 0.0, 0.0),
        clearance_score=1.0,
        ik_feasible=True,
        collision_free=True,
        keepout_clearance_m=float("inf"),
        reason="test",
    )
