from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..localization.mask_localizer import LocalizedTarget, MaskLocalizer
from ..perception.proposal import Proposal, ProposalDetector
from ..perception.segmentation_sam2_video import SegmentTracker, TrackResult
from ..reasoning.command_reasoner import CommandReasoner, TargetSpec


@dataclass
class PerceptionCandidate:
    target: TargetSpec
    proposal: Proposal
    track: TrackResult
    localization: LocalizedTarget
    selection_rank: int
    llm_score: Optional[float] = None
    llm_reasoning: Optional[str] = None


class PerceptionOrchestrator:
    """End-to-end generic perception orchestration with optional LLM-enhanced scoring."""

    def __init__(
        self,
        reasoner: CommandReasoner,
        proposal_detector: ProposalDetector,
        segment_tracker: SegmentTracker,
        mask_localizer: MaskLocalizer,
        llm_reasoner: Optional[Any] = None,
    ) -> None:
        self._reasoner = reasoner
        self._proposal_detector = proposal_detector
        self._segment_tracker = segment_tracker
        self._mask_localizer = mask_localizer
        self._llm_reasoner = llm_reasoner  # Optional LLM reasoner for enhanced scoring
        self._last_attempts: list[dict[str, Any]] = []
        self._last_reject_reason: str = ""
        self._last_target: TargetSpec | None = None
        self._last_llm_command_reasoning: Optional[dict] = None  # Store LLM reasoning from command

    def get_last_attempts(self) -> list[dict[str, Any]]:
        return [dict(a) for a in self._last_attempts]

    def get_last_reject_reason(self) -> str:
        return str(self._last_reject_reason)

    def get_last_target(self) -> TargetSpec | None:
        return self._last_target

    def detect_and_localize(
        self,
        command: str,
        image: object,
        rgb_w: int,
        rgb_h: int,
        track_frames: int,
        track_iou_min: float,
        stability_min: float,
        require_valid_depth: bool,
        top_k_semantic: int,
        require_semantic_pass: bool,
        semantic_min_score: float,
        llm_command_reasoning: Optional[dict] = None,
    ) -> PerceptionCandidate | None:
        """
        Detect and localize target object with optional LLM-enhanced scoring.
        
        Args:
            command: Natural language command
            image: RGB image
            rgb_w, rgb_h: Image dimensions
            track_frames: Number of frames for tracking stabilization
            track_iou_min: Minimum IoU for tracking
            stability_min: Minimum mask stability
            require_valid_depth: Require valid depth measurement
            top_k_semantic: Number of top semantic candidates to consider
            require_semantic_pass: Require semantic score to pass threshold
            semantic_min_score: Minimum semantic score
            llm_command_reasoning: Optional LLM reasoning about the command
            
        Returns:
            PerceptionCandidate or None
        """
        self._last_attempts = []
        self._last_reject_reason = ""
        self._last_llm_command_reasoning = llm_command_reasoning
        target = self._reasoner.parse(command)
        self._last_target = target
        proposals = self._proposal_detector.propose(image, target)
        if not proposals:
            self._last_reject_reason = "no_proposals"
            return None

        proposals = sorted(proposals, key=lambda p: float(p.score), reverse=True)
        limit = max(1, int(top_k_semantic))
        candidates = proposals[:limit]
        
        # Score candidates with LLM if available
        llm_scores = {}
        if self._llm_reasoner is not None and llm_command_reasoning is not None:
            try:
                for proposal in candidates:
                    llm_result = self._llm_reasoner.score_object_with_reasoning(
                        command=command,
                        object_label=proposal.label,
                        object_attributes={
                            "color": proposal.proposal_color_guess,
                            "shape": proposal.proposal_shape_guess,
                            "position": f"({proposal.center_uv[0]}, {proposal.center_uv[1]})",
                        },
                        reasoning_context=llm_command_reasoning,
                    )
                    llm_scores[proposal.label] = {
                        "score": llm_result.get("score", 0.5),
                        "reasoning": llm_result.get("reasoning_trace", ""),
                    }
            except Exception:
                # LLM scoring failed, continue with rule-based scoring
                pass
        
        for rank, proposal in enumerate(candidates, start=1):
            semantic_score = float(getattr(proposal, "semantic_score", 0.0))
            semantic_pass = bool(getattr(proposal, "semantic_pass", semantic_score >= float(semantic_min_score)))
            
            # Apply LLM score boost if available
            llm_score = None
            llm_reasoning = None
            if proposal.label in llm_scores:
                llm_score = llm_scores[proposal.label]["score"]
                llm_reasoning = llm_scores[proposal.label]["reasoning"]
                # Blend LLM score with semantic score (70% LLM, 30% rule-based)
                boosted_score = 0.7 * llm_score + 0.3 * semantic_score / 5.0
                proposal.score = boosted_score
            
            attempt: dict[str, Any] = {
                "rank": int(rank),
                "proposal_source": str(proposal.source),
                "proposal_label": str(proposal.label),
                "proposal_conf": float(proposal.confidence),
                "proposal_score": float(proposal.score),
                "proposal_semantic_score": float(semantic_score),
                "proposal_semantic_pass": bool(semantic_pass),
                "proposal_color_guess": str(getattr(proposal, "proposal_color_guess", "")),
                "proposal_shape_guess": str(getattr(proposal, "proposal_shape_guess", "")),
                "proposal_color_score": float(getattr(proposal, "proposal_color_score", 0.0)),
                "proposal_shape_score": float(getattr(proposal, "proposal_shape_score", 0.0)),
                "proposal_shape_features": dict(getattr(proposal, "proposal_shape_features", {}) or {}),
                "llm_score": llm_score,
                "llm_reasoning": llm_reasoning,
                "reason": "",
                "accepted": False,
            }
            if require_semantic_pass and (not semantic_pass):
                attempt["reason"] = "semantic_reject"
                self._last_attempts.append(attempt)
                continue

            track = self._segment_tracker.track(
                image=image,
                proposal=proposal,
                target=target,
                max_frames=int(track_frames),
                iou_min=float(track_iou_min),
                stability_min=float(stability_min),
            )
            if track is None:
                attempt["reason"] = "track_failed"
                self._last_attempts.append(attempt)
                continue

            loc = self._mask_localizer.localize(
                mask_u8=track.mask_u8,
                rgb_w=int(rgb_w),
                rgb_h=int(rgb_h),
                center_uv=track.center_uv,
            )
            attempt["mask_area_frac"] = float(track.mask_area_frac)
            attempt["mask_stability"] = float(track.stability)
            attempt["mask_track_iou"] = float(track.track_iou)
            attempt["depth_valid_px"] = int(loc.valid_px)
            attempt["candidate_valid_depth"] = bool(loc.valid_depth)
            if require_valid_depth and (not loc.valid_depth):
                attempt["reason"] = "invalid_depth"
                self._last_attempts.append(attempt)
                continue

            attempt["reason"] = "accepted"
            attempt["accepted"] = True
            self._last_attempts.append(attempt)
            self._last_reject_reason = ""
            return PerceptionCandidate(
                target=target,
                proposal=proposal,
                track=track,
                localization=loc,
                selection_rank=int(rank),
                llm_score=llm_score,
                llm_reasoning=llm_reasoning,
            )

        self._last_reject_reason = str(self._last_attempts[-1]["reason"]) if self._last_attempts else "candidate_none"
        return None
