from __future__ import annotations

from fr3_zero_shot.core.types import ObjectHypothesis


class PickVerifier:
    def verify(self, *, motion, source: ObjectHypothesis) -> bool:
        return bool(motion.verify_carry(source))

