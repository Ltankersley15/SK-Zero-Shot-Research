from __future__ import annotations

from fr3_zero_shot.core.types import ObjectHypothesis


class PlaceVerifier:
    def verify(self, *, motion, held_object: ObjectHypothesis, container: ObjectHypothesis) -> bool:
        return bool(motion.verify_place_in(held_object, container))

