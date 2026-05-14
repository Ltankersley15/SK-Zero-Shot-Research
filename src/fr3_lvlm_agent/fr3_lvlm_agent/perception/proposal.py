from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..reasoning.command_reasoner import TargetSpec


@dataclass
class Proposal:
    bbox_xyxy: tuple[int, int, int, int]
    center_uv: tuple[int, int]
    label: str
    confidence: float
    source: str
    score: float
    matched: bool
    semantic_score: float = 0.0
    semantic_pass: bool = False
    canonical_label: str = ""
    proposal_color_guess: str = ""
    proposal_shape_guess: str = ""
    proposal_color_score: float = 0.0
    proposal_shape_score: float = 0.0
    proposal_shape_features: dict[str, float] | None = None
    yaw_rad: float | None = None


class ProposalDetector:
    """Detector abstraction that returns ranked proposal candidates."""

    def __init__(self, proposal_fn: Callable[[object, TargetSpec], list[Proposal]]) -> None:
        self._proposal_fn = proposal_fn

    def propose(self, image: object, target: TargetSpec) -> list[Proposal]:
        return list(self._proposal_fn(image, target))
