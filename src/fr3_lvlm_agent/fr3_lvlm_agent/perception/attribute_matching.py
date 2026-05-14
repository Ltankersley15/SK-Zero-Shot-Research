from __future__ import annotations

import re

from .proposal import Proposal


def proposal_label_tokens(proposal: Proposal) -> set[str]:
    label_text = f"{getattr(proposal, 'label', '')} {getattr(proposal, 'canonical_label', '')}".lower()
    tokens = set(re.findall(r"[a-z0-9]+", label_text))
    shape_alias = {
        "block": "cube",
        "box": "cube",
        "tube": "cylinder",
        "can": "cylinder",
        "bottle": "cylinder",
    }
    return {shape_alias.get(tok, tok) for tok in tokens}


def proposal_has_specific_label(proposal: Proposal, *, generic_tokens: set[str]) -> bool:
    tokens = proposal_label_tokens(proposal)
    return bool(tokens and any(tok not in generic_tokens for tok in tokens))


def proposal_matches_target_attributes(
    proposal: Proposal,
    target: object,
    *,
    generic_tokens: set[str],
) -> bool:
    """Check if proposal matches target attributes.
    
    RELAXED matching: Accept proposals when color/shape detected even if not explicitly labeled.
    """
    target_color = str(getattr(target, "target_color", "") or "").strip().lower()
    target_shape = str(getattr(target, "target_shape", "") or "").strip().lower()
    proposal_color = str(getattr(proposal, "proposal_color_guess", "") or "").strip().lower()
    proposal_shape = str(getattr(proposal, "proposal_shape_guess", "") or "").strip().lower()
    tokens = proposal_label_tokens(proposal)

    # If no specific attributes requested, accept any proposal
    if not target_color and not target_shape:
        return True

    # RELAXED Color gating: Accept if detected color matches OR label contains color
    if target_color:
        # Only reject if proposal explicitly claims DIFFERENT color
        if proposal_color and proposal_color != "unknown" and proposal_color != target_color:
            return False
        # Accept if color matches from any source
        color_match = (
            proposal_color == target_color or  # Detected color matches
            target_color in tokens or  # Label contains color
            proposal.source == "hsv_segmentation"  # Trust HSV detection
        )
        if not color_match:
            return False

    # RELAXED Shape gating: Accept unknown shapes, match on label or detected shape
    if target_shape:
        shape_match = (
            target_shape in tokens or
            proposal_shape == target_shape or
            proposal_shape == "unknown" or
            not proposal_shape
        )
        if not shape_match:
            return False

    # RELAXED: Accept if we have a detection, even without specific label
    return True
