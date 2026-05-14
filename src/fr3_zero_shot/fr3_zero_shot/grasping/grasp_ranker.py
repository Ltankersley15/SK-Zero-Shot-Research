from __future__ import annotations

from fr3_zero_shot.core.types import ApproachFamily, GraspCandidate


def rank_candidates(
    candidates: list[GraspCandidate],
    *,
    approach_preference: ApproachFamily,
) -> list[GraspCandidate]:
    def preference_score(candidate: GraspCandidate) -> float:
        if approach_preference == ApproachFamily.AUTO:
            return 0.0
        if approach_preference == ApproachFamily.TOP_DOWN:
            return 2.0 if candidate.family in {ApproachFamily.TOP_DOWN, ApproachFamily.TOP_DOWN_YAW_45} else 0.0
        if approach_preference == candidate.family:
            return 3.0
        if approach_preference.value.startswith("side") and candidate.family.value.startswith("side"):
            return 1.5
        return 0.0

    return sorted(
        candidates,
        key=lambda cand: (
            preference_score(cand),
            cand.clearance_score,
            cand.keepout_clearance_m,
            1.0 if cand.family != ApproachFamily.TOP_DOWN_YAW_45 else 0.5,
        ),
        reverse=True,
    )

