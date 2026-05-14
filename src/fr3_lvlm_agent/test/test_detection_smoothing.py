from fr3_lvlm_agent.perception.detection_smoothing import DetectionUVSmoother


def test_detection_uv_smoothing_applies_ema():
    smoother = DetectionUVSmoother(alpha=0.5, max_jump_px=50.0, max_age_sec=1.0)
    assert smoother.update("red cube", (100, 100), 0.0) == (100, 100)
    assert smoother.update("red cube", (110, 90), 0.1) == (105, 95)


def test_detection_uv_smoothing_resets_on_target_change():
    smoother = DetectionUVSmoother(alpha=0.2, max_jump_px=50.0, max_age_sec=1.0)
    assert smoother.update("red cube", (100, 100), 0.0) == (100, 100)
    assert smoother.update("blue cube", (200, 200), 0.1) == (200, 200)


def test_detection_uv_smoothing_resets_on_jump():
    smoother = DetectionUVSmoother(alpha=0.5, max_jump_px=20.0, max_age_sec=1.0)
    assert smoother.update("red cube", (100, 100), 0.0) == (100, 100)
    assert smoother.update("red cube", (200, 200), 0.1) == (200, 200)


def test_detection_uv_smoothing_resets_on_age():
    smoother = DetectionUVSmoother(alpha=0.5, max_jump_px=50.0, max_age_sec=0.2)
    assert smoother.update("red cube", (100, 100), 0.0) == (100, 100)
    assert smoother.update("red cube", (110, 110), 1.0) == (110, 110)


def test_detection_uv_smoothing_alpha_one_no_smooth():
    smoother = DetectionUVSmoother(alpha=1.0, max_jump_px=50.0, max_age_sec=1.0)
    assert smoother.update("red cube", (100, 100), 0.0) == (100, 100)
    assert smoother.update("red cube", (110, 90), 0.1) == (110, 90)
