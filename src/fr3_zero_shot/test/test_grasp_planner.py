import pytest

from fr3_zero_shot.core.transforms import grasp_quaternion
from fr3_zero_shot.core.types import ApproachFamily, ObjectHypothesis
from fr3_zero_shot.grasping.grasp_planner import GraspPlanner


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


def test_grasp_planner_generates_side_family_when_preferred_near_keepout():
    source = _obj("green_cube", (0.5, 0.0, 0.02))
    cup = _obj("cup", (0.57, 0.0, 0.04), footprint=(0.07, 0.07), container=True)

    chosen = GraspPlanner(min_keepout_clearance_m=0.005).plan(
        source=source,
        avoid_objects=(cup,),
        approach_preference=ApproachFamily.SIDE_LEFT,
    )

    assert chosen is not None
    assert chosen.family.value.startswith("side")
    assert chosen.keepout_clearance_m >= 0.005


def test_grasp_planner_fails_closed_when_all_candidates_fail_ik():
    source = _obj("blue_cube", (0.5, 0.0, 0.02))
    planner = GraspPlanner(ik_checker=lambda candidate: False)

    assert planner.plan(source=source, avoid_objects=(), approach_preference=ApproachFamily.TOP_DOWN) is None


def test_side_back_quaternion_points_tcp_toward_negative_x():
    rotated_z = _rotate_vector(grasp_quaternion(ApproachFamily.SIDE_BACK), (0.0, 0.0, 1.0))

    assert rotated_z[0] == pytest.approx(-1.0, abs=1e-3)
    assert rotated_z[1] == pytest.approx(0.0, abs=1e-3)
    assert rotated_z[2] == pytest.approx(0.0, abs=1e-3)


def _rotate_vector(q, vector):
    x, y, z, w = q
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )
