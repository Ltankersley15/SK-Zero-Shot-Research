from __future__ import annotations

import numpy as np

from fr3_lvlm_agent.perception.proposal_sam2_masks import (
    adaptive_edge_margin,
    classify_shape_from_mask,
    infer_canonical_color_from_hsv,
)


def test_infer_canonical_color_from_hsv_red() -> None:
    color, score = infer_canonical_color_from_hsv(2.0, 220.0, 200.0)
    assert color == "red"
    assert score > 0.5


def test_classify_shape_from_mask_cube_like() -> None:
    m = np.zeros((80, 80), dtype=np.uint8)
    m[20:60, 20:60] = 255
    shape, score, features = classify_shape_from_mask(m)
    assert shape in {"cube", "cylinder", "unknown"}
    assert score > 0.45
    assert "extent" in features


def test_classify_shape_from_mask_cylinder_like() -> None:
    y, x = np.ogrid[:80, :80]
    center = np.array([40, 40])
    radius = 18
    m = (((x - center[1]) ** 2 + (y - center[0]) ** 2) <= radius * radius).astype(np.uint8) * 255
    shape, score, features = classify_shape_from_mask(m)
    assert shape in {"cylinder", "cube", "unknown"}
    assert score > 0.45
    assert features["circularity"] > 0.5


def test_adaptive_edge_margin_reduces_margin_for_strong_semantics() -> None:
    base = adaptive_edge_margin(
        base_margin_px=36,
        semantic_score=5.0,
        mask_stability=1.0,
        enabled=True,
        semantic_threshold=4.0,
        stability_threshold=0.5,
        override_margin_px=4,
    )
    weak = adaptive_edge_margin(
        base_margin_px=36,
        semantic_score=1.0,
        mask_stability=0.2,
        enabled=True,
        semantic_threshold=4.0,
        stability_threshold=0.5,
        override_margin_px=4,
    )
    assert base == 4
    assert weak == 36
