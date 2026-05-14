from fr3_lvlm_agent.perception.detection_policy import should_skip_sam2_for_yolo


def test_skip_sam2_when_yolo_attr_match():
    assert should_skip_sam2_for_yolo(
        attribute_required=True,
        prefer_yolo_fast_path=True,
        skip_sam2_when_yolo_present=True,
        yolo_props_count=2,
        yolo_attr_count=1,
    )


def test_do_not_skip_sam2_when_no_attr_and_flag_false():
    assert not should_skip_sam2_for_yolo(
        attribute_required=True,
        prefer_yolo_fast_path=True,
        skip_sam2_when_yolo_present=False,
        yolo_props_count=2,
        yolo_attr_count=0,
    )


def test_skip_sam2_when_no_attr_and_flag_true():
    assert should_skip_sam2_for_yolo(
        attribute_required=True,
        prefer_yolo_fast_path=True,
        skip_sam2_when_yolo_present=True,
        yolo_props_count=2,
        yolo_attr_count=0,
    )


def test_no_skip_when_attribute_not_required():
    assert not should_skip_sam2_for_yolo(
        attribute_required=False,
        prefer_yolo_fast_path=True,
        skip_sam2_when_yolo_present=True,
        yolo_props_count=2,
        yolo_attr_count=1,
    )


def test_do_not_skip_sam2_when_multiple_attribute_matches_make_yolo_ambiguous():
    assert not should_skip_sam2_for_yolo(
        attribute_required=True,
        prefer_yolo_fast_path=True,
        skip_sam2_when_yolo_present=True,
        yolo_props_count=3,
        yolo_attr_count=2,
    )
