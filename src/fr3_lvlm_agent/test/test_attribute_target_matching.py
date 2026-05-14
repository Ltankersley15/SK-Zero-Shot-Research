from fr3_lvlm_agent.perception.attribute_matching import proposal_matches_target_attributes
from fr3_lvlm_agent.perception.proposal import Proposal
from fr3_lvlm_agent.reasoning.command_reasoner import CommandReasoner


def _proposal(
    *,
    label: str,
    canonical_label: str,
    color_guess: str,
    color_score: float,
    shape_guess: str = "cube",
) -> Proposal:
    return Proposal(
        bbox_xyxy=(10, 10, 30, 30),
        center_uv=(20, 20),
        label=label,
        confidence=0.9,
        source="yolo_world",
        score=0.9,
        matched=True,
        semantic_score=0.9,
        semantic_pass=True,
        canonical_label=canonical_label,
        proposal_color_guess=color_guess,
        proposal_shape_guess=shape_guess,
        proposal_color_score=float(color_score),
        proposal_shape_score=0.5,
    )


def test_attribute_match_rejects_conflicting_pixel_color_even_when_label_matches() -> None:
    target = CommandReasoner().parse("pick up the red cube")
    prop = _proposal(
        label="red cube",
        canonical_label="red cube",
        color_guess="blue",
        color_score=0.20,
    )
    assert proposal_matches_target_attributes(prop, target, generic_tokens={"object", "item"}) is False


def test_attribute_match_allows_label_when_color_hint_unknown() -> None:
    target = CommandReasoner().parse("pick up the red cube")
    prop = _proposal(
        label="red cube",
        canonical_label="red cube",
        color_guess="unknown",
        color_score=0.0,
    )
    assert proposal_matches_target_attributes(prop, target, generic_tokens={"object", "item"}) is True
