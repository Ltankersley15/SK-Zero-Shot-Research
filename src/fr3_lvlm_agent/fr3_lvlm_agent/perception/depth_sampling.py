from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DepthSample:
    depth_m: float
    raw_depth_m: float
    used_patch: bool
    patch_valid_count: int


def sample_depth(depth_arr: np.ndarray, u: int, v: int) -> Optional[DepthSample]:
    raw = float(depth_arr[v, u])
    raw_valid = bool(np.isfinite(raw) and (0.01 <= raw <= 10.0))
    if raw_valid:
        return DepthSample(depth_m=raw, raw_depth_m=raw, used_patch=False, patch_valid_count=1)

    y0 = max(0, v - 2)
    y1 = min(depth_arr.shape[0], v + 3)
    x0 = max(0, u - 2)
    x1 = min(depth_arr.shape[1], u + 3)
    patch = depth_arr[y0:y1, x0:x1].astype(np.float32)
    valid = patch[np.isfinite(patch) & (patch >= 0.01) & (patch <= 10.0)]
    if valid.size <= 0:
        return None
    valid_sorted = np.sort(valid, axis=None)
    q = 0.25 * float(valid_sorted.size - 1)
    lo = int(math.floor(q))
    hi = int(math.ceil(q))
    if lo == hi:
        depth = float(valid_sorted[lo])
    else:
        frac = q - float(lo)
        depth = float(valid_sorted[lo] + (valid_sorted[hi] - valid_sorted[lo]) * frac)
    return DepthSample(
        depth_m=depth,
        raw_depth_m=raw,
        used_patch=True,
        patch_valid_count=int(valid.size),
    )
