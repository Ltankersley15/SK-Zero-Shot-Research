from __future__ import annotations

from fr3_zero_shot.core.types import GraspCandidate


def is_valid(candidate: GraspCandidate, *, min_keepout_clearance_m: float) -> bool:
    return bool(
        candidate.ik_feasible
        and candidate.collision_free
        and candidate.keepout_clearance_m >= float(min_keepout_clearance_m)
    )

