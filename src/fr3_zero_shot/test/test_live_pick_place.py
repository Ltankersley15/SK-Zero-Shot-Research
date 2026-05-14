import numpy as np

from fr3_zero_shot.core.types import ObjectHypothesis
from fr3_zero_shot.execution.gripper_adapter import contact_detected_from_width
from fr3_zero_shot.live_pick_place_utils import (
    best_colored_pickable,
    colored_pickables_by_area,
    color_visible_outside_anchor,
    normalize_color_sequence,
    release_drop_for_sequence_step,
    select_cup,
    side_family_away_from_keepout,
    source_still_on_table,
)
from fr3_zero_shot.perception.container_detector import detect_dark_neutral_container_blobs


def _obj(
    object_id="obj_cup",
    xyz=(0.56, 0.0, 0.06),
    confidence=0.9,
    color=None,
    container=True,
    metadata=None,
):
    return ObjectHypothesis(
        object_id=object_id,
        label="cup" if container else f"{color} cube",
        color=color,
        shape="cup" if container else "cube",
        xyz=xyz,
        footprint_xy=(0.12, 0.12),
        height_m=0.08 if container else 0.04,
        bbox_xyxy=(0, 0, 20, 20),
        confidence=confidence,
        visible=True,
        pickable=not container,
        container_like=container,
        metadata={} if metadata is None else metadata,
    )


def _cup(object_id="obj_cup", xyz=(0.56, 0.0, 0.06), confidence=0.9):
    return _obj(object_id=object_id, xyz=xyz, confidence=confidence, container=True)


def test_neutral_dark_container_candidate_becomes_cup():
    rgb = np.full((160, 160, 3), 240, dtype=np.uint8)
    rgb[55:105, 50:110] = (25, 25, 25)

    containers = detect_dark_neutral_container_blobs(rgb)

    assert len(containers) == 1
    assert containers[0].bbox_xyxy == (50, 55, 110, 105)
    assert containers[0].area_px >= 2500


def test_tiny_dark_container_fragment_is_ignored():
    rgb = np.full((160, 160, 3), 240, dtype=np.uint8)
    rgb[20:28, 20:28] = (20, 20, 20)

    assert detect_dark_neutral_container_blobs(rgb) == []


def test_two_view_cup_agreement_merges_duplicate_observations():
    cup, reason = select_cup(
        [
            ("cup_oblique", [_cup(xyz=(0.54, -0.02, 0.06), confidence=0.7)]),
            ("cup_top", []),
            ("cup_side", [_cup(xyz=(0.57, 0.01, 0.06), confidence=0.9)]),
        ]
    )

    assert cup is not None
    assert cup.container_like
    assert "agreement" in reason
    assert cup.xyz[0] == (0.54 + 0.57) / 2.0


def test_top_view_cup_detection_is_preferred():
    cup, reason = select_cup(
        [
            ("cup_oblique", [_cup(xyz=(0.30, 0.30, 0.06), confidence=1.0)]),
            ("cup_top", [_cup(xyz=(0.56, 0.0, 0.06), confidence=0.5)]),
        ]
    )

    assert cup is not None
    assert cup.xyz == (0.56, 0.0, 0.06)
    assert reason == "top-view cup detection"


def test_gripper_contact_passes_when_width_stays_above_close_command():
    assert contact_detected_from_width(0.018, commanded_width_m=0.014, margin_m=0.002)


def test_gripper_contact_fails_when_width_reaches_close_command():
    assert not contact_detected_from_width(0.014, commanded_width_m=0.014, margin_m=0.002)


def test_best_colored_pickable_prefers_high_confidence_match():
    low = _obj("low_blue", color="blue", confidence=0.3, container=False)
    high = _obj("high_blue", color="blue", confidence=0.9, container=False)
    red = _obj("red", color="red", confidence=1.0, container=False)

    assert best_colored_pickable([low, high, red], "blue") == high


def test_colored_pickables_by_area_merges_duplicate_views_and_sorts_size():
    large = _obj(
        "large_red",
        xyz=(0.40, 0.20, 0.04),
        color="red",
        container=False,
        metadata={"area_px": 9000},
    )
    duplicate_large = _obj(
        "large_red_alt",
        xyz=(0.43, 0.19, 0.04),
        color="red",
        container=False,
        metadata={"area_px": 7000},
    )
    small = _obj(
        "small_red",
        xyz=(0.55, -0.10, 0.04),
        color="red",
        container=False,
        metadata={"area_px": 2500},
    )

    ranked = colored_pickables_by_area([small, duplicate_large, large], "red")

    assert ranked == [large, small]


def test_source_still_on_table_matches_source_position_not_only_color():
    source = _obj("small_red", xyz=(0.40, -0.26, 0.04), color="red", container=False)
    placed_stack_red = _obj(
        "placed_red",
        xyz=(0.49, 0.43, 0.04),
        color="red",
        container=False,
        metadata={"raw_depth_base_xyz": (0.70, -0.10, 0.11)},
    )
    same_spot_red = _obj(
        "same_spot_red",
        xyz=(0.42, -0.25, 0.04),
        color="red",
        container=False,
        metadata={"raw_depth_base_xyz": (0.68, -0.15, 0.07)},
    )

    assert not source_still_on_table(source, [placed_stack_red], table_z=0.02)
    assert source_still_on_table(source, [same_spot_red], table_z=0.02)


def test_side_family_away_from_keepout_points_pregrasp_away_from_cup():
    cup = _cup(xyz=(0.56, -0.12, 0.06))
    green = _obj("green", xyz=(0.62, -0.02, 0.04), color="green", container=False)
    left_green = _obj("green", xyz=(0.50, -0.12, 0.04), color="green", container=False)

    assert side_family_away_from_keepout(green, cup).value == "side_right"
    assert side_family_away_from_keepout(left_green, cup).value == "side_front"


def test_color_visible_outside_anchor_ignores_inside_footprint():
    cup = _cup(xyz=(0.5, 0.0, 0.06))
    inside = _obj(
        "blue_inside",
        xyz=(0.52, 0.01, 0.04),
        color="blue",
        container=False,
        metadata={"raw_depth_base_xyz": (0.52, 0.01, 0.05)},
    )

    assert not color_visible_outside_anchor([inside], color="blue", anchor=cup, table_z=0.02)


def test_color_visible_outside_anchor_rejects_tabletop_cube_away_from_cup():
    cup = _cup(xyz=(0.5, 0.0, 0.06))
    outside = _obj(
        "blue_outside",
        xyz=(0.70, 0.20, 0.04),
        color="blue",
        container=False,
        metadata={"raw_depth_base_xyz": (0.70, 0.20, 0.05)},
    )

    assert color_visible_outside_anchor([outside], color="blue", anchor=cup, table_z=0.02)


def test_normalize_color_sequence_rejects_unsupported_color():
    try:
        normalize_color_sequence(["red", "purple"])
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("expected unsupported color")


def test_sequence_release_profile_stops_five_cm_higher():
    assert release_drop_for_sequence_step(0) == 0.095
    assert release_drop_for_sequence_step(1) == 0.10
    assert release_drop_for_sequence_step(3) == 0.10
