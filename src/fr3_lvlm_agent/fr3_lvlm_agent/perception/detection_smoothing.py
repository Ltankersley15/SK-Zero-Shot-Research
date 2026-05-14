from __future__ import annotations

import math
from typing import Optional, Tuple


class DetectionUVSmoother:
    """Temporal smoothing for detection center coordinates."""

    def __init__(self, *, alpha: float, max_jump_px: float, max_age_sec: float) -> None:
        self.alpha = float(min(1.0, max(0.0, alpha)))
        self.max_jump_px = float(max(0.0, max_jump_px))
        self.max_age_sec = float(max(0.0, max_age_sec))
        self._last_target: Optional[str] = None
        self._last_uv: Optional[Tuple[float, float]] = None
        self._last_stamp_sec: Optional[float] = None

    def reset(self) -> None:
        self._last_target = None
        self._last_uv = None
        self._last_stamp_sec = None

    def update(self, target_desc: str, uv: Tuple[int, int], stamp_sec: float) -> Tuple[int, int]:
        target = str(target_desc or "").strip().lower()
        u, v = int(uv[0]), int(uv[1])
        if not target:
            self.reset()
            return u, v

        if self._last_target != target or self._last_uv is None:
            self._set_state(target, (float(u), float(v)), stamp_sec)
            return u, v

        if self.max_age_sec > 0.0 and self._last_stamp_sec is not None:
            if (stamp_sec - self._last_stamp_sec) > self.max_age_sec:
                self._set_state(target, (float(u), float(v)), stamp_sec)
                return u, v

        du = float(u) - float(self._last_uv[0])
        dv = float(v) - float(self._last_uv[1])
        if self.max_jump_px > 0.0 and math.hypot(du, dv) > self.max_jump_px:
            self._set_state(target, (float(u), float(v)), stamp_sec)
            return u, v

        if self.alpha >= 1.0:
            smoothed = (float(u), float(v))
        elif self.alpha <= 0.0:
            smoothed = (float(self._last_uv[0]), float(self._last_uv[1]))
        else:
            smoothed = (
                (1.0 - self.alpha) * float(self._last_uv[0]) + self.alpha * float(u),
                (1.0 - self.alpha) * float(self._last_uv[1]) + self.alpha * float(v),
            )

        self._set_state(target, smoothed, stamp_sec)
        return int(round(smoothed[0])), int(round(smoothed[1]))

    def _set_state(self, target: str, uv: Tuple[float, float], stamp_sec: float) -> None:
        self._last_target = target
        self._last_uv = uv
        self._last_stamp_sec = float(stamp_sec)
