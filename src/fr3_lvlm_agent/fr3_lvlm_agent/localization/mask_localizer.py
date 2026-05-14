from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class LocalizedTarget:
    sample_uv: tuple[int, int]
    center_uv: tuple[int, int]
    depth_m: float | None
    valid_px: int
    valid_depth: bool


class MaskLocalizer:
    """Depth localization from segmentation masks."""

    def __init__(self, depth_from_mask_fn: Callable[[np.ndarray, int, int], tuple[int, int, float, int]]) -> None:
        self._depth_from_mask_fn = depth_from_mask_fn

    def localize(self, mask_u8: np.ndarray, rgb_w: int, rgb_h: int, center_uv: tuple[int, int]) -> LocalizedTarget:
        u, v, d, valid_px = self._depth_from_mask_fn(mask_u8, int(rgb_w), int(rgb_h))
        valid_depth = bool(np.isfinite(d))
        return LocalizedTarget(
            sample_uv=(int(u), int(v)),
            center_uv=(int(center_uv[0]), int(center_uv[1])),
            depth_m=(float(d) if valid_depth else None),
            valid_px=int(valid_px),
            valid_depth=valid_depth,
        )
