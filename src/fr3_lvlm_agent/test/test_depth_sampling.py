import numpy as np
import pytest

from fr3_lvlm_agent.perception.depth_sampling import sample_depth


def test_sample_depth_uses_raw_when_valid():
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 0.42
    sample = sample_depth(depth, 2, 2)
    assert sample is not None
    assert sample.depth_m == pytest.approx(0.42)
    assert sample.raw_depth_m == pytest.approx(0.42)
    assert not sample.used_patch
    assert sample.patch_valid_count == 1


def test_sample_depth_uses_patch_when_raw_invalid():
    depth = np.full((5, 5), np.nan, dtype=np.float32)
    depth[2, 2] = np.nan
    depth[1, 1] = 0.50
    depth[1, 2] = 0.60
    depth[2, 1] = 0.40
    sample = sample_depth(depth, 2, 2)
    assert sample is not None
    assert sample.used_patch
    # 25th percentile of [0.50, 0.60, 0.40] is 0.45
    assert abs(sample.depth_m - 0.45) < 1e-6
    assert np.isnan(sample.raw_depth_m)
    assert sample.patch_valid_count == 3


def test_sample_depth_returns_none_when_no_valid_patch():
    depth = np.full((5, 5), np.nan, dtype=np.float32)
    sample = sample_depth(depth, 2, 2)
    assert sample is None
