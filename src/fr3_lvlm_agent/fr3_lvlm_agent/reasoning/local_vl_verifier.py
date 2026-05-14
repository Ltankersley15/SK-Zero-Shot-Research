from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import threading
import time
from typing import Any, Sequence

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class LocalVLCandidate:
    candidate_id: str
    label: str
    bbox_xyxy: tuple[int, int, int, int] | None = None
    center_uv: tuple[int, int] | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalVLCandidateScore:
    candidate_id: str
    contains_target: bool
    confidence: float
    status: str
    reason: str = ""
    raw_response: str = ""
    latency_sec: float = 0.0


@dataclass(frozen=True)
class LocalVLDecision:
    selected_candidate_id: str | None
    confidence: float
    status: str
    reason: str = ""
    raw_response: str = ""
    latency_sec: float = 0.0
    candidate_scores: tuple[LocalVLCandidateScore, ...] = ()


@dataclass(frozen=True)
class LocalVLPostPlaceCheck:
    object_in_container: bool | None
    confidence: float
    status: str
    reason: str = ""
    raw_response: str = ""
    latency_sec: float = 0.0


@dataclass(frozen=True)
class LocalVLHypothesisPack:
    target_text: str
    role: str
    status: str
    confidence: float
    query_terms: tuple[str, ...] = ()
    distractor_terms: tuple[str, ...] = ()
    occlusion_hints: tuple[str, ...] = ()
    geometry_hints: tuple[str, ...] = ()
    approach_hints: tuple[str, ...] = ()
    reason: str = ""
    raw_response: str = ""
    latency_sec: float = 0.0


@dataclass(frozen=True)
class LocalVLSceneObjectAnnotation:
    candidate_id: str
    refined_label: str = ""
    visibility: str = "unknown"
    occluded_by_ids: tuple[str, ...] = ()
    access_hints: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class LocalVLSceneRelation:
    subject_id: str
    relation: str
    object_id: str
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class LocalVLSceneCensus:
    status: str
    confidence: float
    reason: str = ""
    objects: tuple[LocalVLSceneObjectAnnotation, ...] = ()
    relations: tuple[LocalVLSceneRelation, ...] = ()
    raw_response: str = ""
    latency_sec: float = 0.0


@dataclass(frozen=True)
class LocalVLRoleSelection:
    role: str
    candidate_id: str | None
    confidence: float
    status: str
    reason: str = ""


@dataclass(frozen=True)
class LocalVLSceneTargetDecision:
    status: str
    confidence: float
    selections: tuple[LocalVLRoleSelection, ...] = ()
    reason: str = ""
    raw_response: str = ""
    latency_sec: float = 0.0

    def selection_for_role(self, role: str) -> LocalVLRoleSelection | None:
        normalized = " ".join(str(role or "").strip().lower().split())
        for selection in self.selections:
            if selection.role == normalized:
                return selection
        return None


class LocalVisionVerifier:
    _GLOBAL_CALL_LOCK = threading.Lock()
    _SCENE_RELATIONS = {"left_of", "right_of", "near", "occludes", "blocks_top_access", "inside"}
    _SCENE_ACCESS_HINTS = {
        "favor_side_access",
        "top_access_occluded",
        "keep_clear_of_occluder",
        "reobserve_before_descent",
    }

    def __init__(
        self,
        *,
        client,
        max_crop_px: int = 512,
        default_timeout_sec: float = 8.0,
    ) -> None:
        self._client = client
        self._max_crop_px = max(64, int(max_crop_px))
        self._default_timeout_sec = max(0.1, float(default_timeout_sec))

    def choose_scene_targets(
        self,
        *,
        image: Image.Image,
        command_text: str,
        target_text: str,
        roles: Sequence[str],
        candidates: Sequence[LocalVLCandidate],
        confidence_threshold: float,
    ) -> LocalVLSceneTargetDecision:
        if image is None or not isinstance(image, Image.Image):
            return LocalVLSceneTargetDecision(status="missing_image", confidence=0.0, reason="missing image")
        normalized_roles = tuple(
            role
            for role in (
                " ".join(str(item or "").strip().lower().split())
                for item in roles
            )
            if role
        )
        if not normalized_roles:
            return LocalVLSceneTargetDecision(status="missing_roles", confidence=0.0, reason="missing roles")
        if not candidates:
            return LocalVLSceneTargetDecision(status="no_candidates", confidence=0.0, reason="no candidates")

        candidate_by_id = {str(candidate.candidate_id): candidate for candidate in candidates}
        annotated = self._scene_target_image(image=image, candidates=candidates)
        prompt = self._scene_target_prompt(
            command_text=command_text,
            target_text=target_text,
            roles=normalized_roles,
            candidates=candidates,
        )
        started = time.time()
        raw_content = self._run_image_prompt(image=annotated, prompt=prompt, max_tokens=360)
        latency_sec = max(0.0, time.time() - started)
        parsed = self._parse_scene_target_result(
            raw_content=raw_content,
            valid_roles=set(normalized_roles),
            valid_candidate_ids=set(candidate_by_id),
        )
        confidence = float(parsed.get("confidence", 0.0))
        status = str(parsed.get("status", "ok"))
        reason = str(parsed.get("reason", ""))
        if status != "ok":
            return LocalVLSceneTargetDecision(
                status=status,
                confidence=confidence,
                reason=reason,
                raw_response=raw_content,
                latency_sec=latency_sec,
            )

        selections: list[LocalVLRoleSelection] = []
        for selection in parsed.get("selections", ()):
            role = str(selection.role)
            candidate_id = selection.candidate_id
            selection_status = str(selection.status or "ok")
            selection_reason = str(selection.reason or "")
            if candidate_id is None:
                selections.append(selection)
                continue
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                selections.append(
                    LocalVLRoleSelection(
                        role=role,
                        candidate_id=None,
                        confidence=float(selection.confidence),
                        status="invalid_candidate",
                        reason=selection_reason or f"unknown candidate_id {candidate_id}",
                    )
                )
                continue
            if float(selection.confidence) < float(confidence_threshold):
                selections.append(
                    LocalVLRoleSelection(
                        role=role,
                        candidate_id=None,
                        confidence=float(selection.confidence),
                        status="low_confidence",
                        reason=selection_reason or "confidence below threshold",
                    )
                )
                continue
            allowed, reject_reason = self._candidate_role_selection_allowed(
                role=role,
                command_text=command_text,
                target_text=target_text,
                candidate=candidate,
            )
            if not allowed:
                selections.append(
                    LocalVLRoleSelection(
                        role=role,
                        candidate_id=None,
                        confidence=float(selection.confidence),
                        status="rejected_candidate",
                        reason=reject_reason or selection_reason,
                    )
                )
                continue
            selections.append(
                LocalVLRoleSelection(
                    role=role,
                    candidate_id=candidate_id,
                    confidence=float(selection.confidence),
                    status=selection_status,
                    reason=selection_reason,
                )
            )

        best_confidence = max((selection.confidence for selection in selections), default=confidence)
        if not any(selection.candidate_id is not None for selection in selections):
            terminal_status = "no_match"
            if any(selection.status == "rejected_candidate" for selection in selections):
                terminal_status = "rejected_candidate"
            elif any(selection.status == "low_confidence" for selection in selections):
                terminal_status = "low_confidence"
            return LocalVLSceneTargetDecision(
                status=terminal_status,
                confidence=best_confidence,
                selections=tuple(selections),
                reason=reason or "no role selection passed deterministic gates",
                raw_response=raw_content,
                latency_sec=latency_sec,
            )
        return LocalVLSceneTargetDecision(
            status="ok",
            confidence=best_confidence,
            selections=tuple(selections),
            reason=reason,
            raw_response=raw_content,
            latency_sec=latency_sec,
        )

    def choose_named_container(
        self,
        *,
        image: Image.Image,
        target_text: str,
        candidates: Sequence[LocalVLCandidate],
        confidence_threshold: float,
    ) -> LocalVLDecision:
        if image is None or not isinstance(image, Image.Image):
            return LocalVLDecision(None, 0.0, "missing_image", reason="missing image")
        if not candidates:
            return LocalVLDecision(None, 0.0, "no_candidates", reason="no candidates")

        scores: list[LocalVLCandidateScore] = []
        best_score: LocalVLCandidateScore | None = None
        best_raw = ""
        total_latency = 0.0
        for candidate in candidates:
            crop = self._candidate_crop(image=image, candidate=candidate, context_scale=1.1)
            if crop is None:
                scores.append(
                    LocalVLCandidateScore(
                        candidate_id=candidate.candidate_id,
                        contains_target=False,
                        confidence=0.0,
                        status="missing_roi",
                        reason="missing crop region",
                    )
                )
                continue
            prompt = self._named_container_prompt(target_text=target_text)
            started = time.time()
            raw_content = self._run_image_prompt(image=crop, prompt=prompt, max_tokens=220)
            latency_sec = max(0.0, time.time() - started)
            total_latency += latency_sec
            parsed = self._parse_json_result(raw_content=raw_content, yes_key="contains_target")
            score = LocalVLCandidateScore(
                candidate_id=candidate.candidate_id,
                contains_target=bool(parsed.get("flag", False)),
                confidence=float(parsed.get("confidence", 0.0)),
                status=str(parsed.get("status", "ok")),
                reason=str(parsed.get("reason", "")),
                raw_response=raw_content,
                latency_sec=latency_sec,
            )
            scores.append(score)
            if score.status != "ok" or not score.contains_target:
                continue
            if best_score is None or (score.confidence, -latency_sec) > (best_score.confidence, -best_score.latency_sec):
                best_score = score
                best_raw = raw_content

        if best_score is None:
            best_observed = max((score.confidence for score in scores), default=0.0)
            statuses = {score.status for score in scores}
            if "timeout" in statuses and statuses <= {"timeout", "missing_roi"}:
                return LocalVLDecision(
                    selected_candidate_id=None,
                    confidence=best_observed,
                    status="timeout",
                    reason="all candidate calls timed out",
                    raw_response=best_raw,
                    latency_sec=total_latency,
                    candidate_scores=tuple(scores),
                )
            if "parse_fail" in statuses and statuses <= {"parse_fail", "missing_roi"}:
                return LocalVLDecision(
                    selected_candidate_id=None,
                    confidence=best_observed,
                    status="parse_fail",
                    reason="all candidate calls failed to parse",
                    raw_response=best_raw,
                    latency_sec=total_latency,
                    candidate_scores=tuple(scores),
                )
            return LocalVLDecision(
                selected_candidate_id=None,
                confidence=best_observed,
                status="no_match",
                reason="no candidate met the requested target",
                raw_response=best_raw,
                latency_sec=total_latency,
                candidate_scores=tuple(scores),
            )
        if best_score.confidence < float(confidence_threshold):
            return LocalVLDecision(
                selected_candidate_id=None,
                confidence=best_score.confidence,
                status="low_confidence",
                reason=best_score.reason or "confidence below threshold",
                raw_response=best_score.raw_response,
                latency_sec=total_latency,
                candidate_scores=tuple(scores),
            )
        return LocalVLDecision(
            selected_candidate_id=best_score.candidate_id,
            confidence=best_score.confidence,
            status=best_score.status,
            reason=best_score.reason,
            raw_response=best_score.raw_response,
            latency_sec=total_latency,
            candidate_scores=tuple(scores),
        )

    def check_object_in_container(
        self,
        *,
        image: Image.Image,
        object_text: str,
        container_text: str,
        candidate: LocalVLCandidate,
        confidence_threshold: float,
    ) -> LocalVLPostPlaceCheck:
        if image is None or not isinstance(image, Image.Image):
            return LocalVLPostPlaceCheck(None, 0.0, "missing_image", reason="missing image")
        crop = self._candidate_crop(image=image, candidate=candidate, context_scale=1.8)
        if crop is None:
            return LocalVLPostPlaceCheck(None, 0.0, "missing_roi", reason="missing crop region")
        prompt = self._post_place_prompt(object_text=object_text, container_text=container_text)
        started = time.time()
        raw_content = self._run_image_prompt(image=crop, prompt=prompt)
        latency_sec = max(0.0, time.time() - started)
        parsed = self._parse_json_result(raw_content=raw_content, yes_key="object_in_container")
        confidence = float(parsed.get("confidence", 0.0))
        if str(parsed.get("status", "ok")) != "ok":
            return LocalVLPostPlaceCheck(
                None,
                confidence,
                str(parsed.get("status", "error")),
                reason=str(parsed.get("reason", "")),
                raw_response=raw_content,
                latency_sec=latency_sec,
            )
        if confidence < float(confidence_threshold):
            return LocalVLPostPlaceCheck(
                None,
                confidence,
                "low_confidence",
                reason=str(parsed.get("reason", "")),
                raw_response=raw_content,
                latency_sec=latency_sec,
            )
        return LocalVLPostPlaceCheck(
            bool(parsed.get("flag", False)),
            confidence,
            "ok",
            reason=str(parsed.get("reason", "")),
            raw_response=raw_content,
            latency_sec=latency_sec,
        )

    def build_hypothesis_pack(
        self,
        *,
        command_text: str,
        target_text: str,
        role: str,
        candidate_limit: int = 6,
        confidence_threshold: float = 0.0,
    ) -> LocalVLHypothesisPack:
        normalized_target = str(target_text or "").strip().lower()
        normalized_role = str(role or "object").strip().lower() or "object"
        if not normalized_target:
            return LocalVLHypothesisPack(
                target_text="",
                role=normalized_role,
                status="missing_target",
                confidence=0.0,
                reason="missing target text",
            )
        prompt = self._hypothesis_prompt(
            command_text=str(command_text or "").strip(),
            target_text=normalized_target,
            role=normalized_role,
            candidate_limit=max(2, int(candidate_limit)),
        )
        started = time.time()
        raw_content = self._run_text_prompt(prompt=prompt)
        latency_sec = max(0.0, time.time() - started)
        parsed = self._parse_hypothesis_result(
            raw_content=raw_content,
            candidate_limit=max(2, int(candidate_limit)),
        )
        confidence = float(parsed.get("confidence", 0.0))
        if str(parsed.get("status", "ok")) != "ok":
            return LocalVLHypothesisPack(
                target_text=normalized_target,
                role=normalized_role,
                status=str(parsed.get("status", "error")),
                confidence=confidence,
                reason=str(parsed.get("reason", "")),
                raw_response=raw_content,
                latency_sec=latency_sec,
            )
        if confidence < float(confidence_threshold):
            return LocalVLHypothesisPack(
                target_text=normalized_target,
                role=normalized_role,
                status="low_confidence",
                confidence=confidence,
                query_terms=tuple(parsed.get("query_terms", ())),
                distractor_terms=tuple(parsed.get("distractor_terms", ())),
                occlusion_hints=tuple(parsed.get("occlusion_hints", ())),
                geometry_hints=tuple(parsed.get("geometry_hints", ())),
                approach_hints=tuple(parsed.get("approach_hints", ())),
                reason=str(parsed.get("reason", "")),
                raw_response=raw_content,
                latency_sec=latency_sec,
            )
        return LocalVLHypothesisPack(
            target_text=normalized_target,
            role=normalized_role,
            status="ok",
            confidence=confidence,
            query_terms=tuple(parsed.get("query_terms", ())),
            distractor_terms=tuple(parsed.get("distractor_terms", ())),
            occlusion_hints=tuple(parsed.get("occlusion_hints", ())),
            geometry_hints=tuple(parsed.get("geometry_hints", ())),
            approach_hints=tuple(parsed.get("approach_hints", ())),
            reason=str(parsed.get("reason", "")),
            raw_response=raw_content,
            latency_sec=latency_sec,
        )

    def build_scene_census(
        self,
        *,
        image: Image.Image,
        candidates: Sequence[LocalVLCandidate],
        confidence_threshold: float = 0.0,
    ) -> LocalVLSceneCensus:
        if image is None or not isinstance(image, Image.Image):
            return LocalVLSceneCensus(status="missing_image", confidence=0.0, reason="missing image")
        if not candidates:
            return LocalVLSceneCensus(status="no_candidates", confidence=0.0, reason="no candidates")
        prompt = self._scene_census_prompt(candidates=candidates)
        started = time.time()
        raw_content = self._run_image_prompt(image=image, prompt=prompt)
        latency_sec = max(0.0, time.time() - started)
        parsed = self._parse_scene_census_result(
            raw_content=raw_content,
            valid_candidate_ids={str(candidate.candidate_id) for candidate in candidates},
        )
        confidence = float(parsed.get("confidence", 0.0))
        if str(parsed.get("status", "ok")) != "ok":
            return LocalVLSceneCensus(
                status=str(parsed.get("status", "error")),
                confidence=confidence,
                reason=str(parsed.get("reason", "")),
                raw_response=raw_content,
                latency_sec=latency_sec,
            )
        census = LocalVLSceneCensus(
            status="ok",
            confidence=confidence,
            reason=str(parsed.get("reason", "")),
            objects=tuple(parsed.get("objects", ())),
            relations=tuple(parsed.get("relations", ())),
            raw_response=raw_content,
            latency_sec=latency_sec,
        )
        if confidence < float(confidence_threshold):
            return LocalVLSceneCensus(
                status="low_confidence",
                confidence=confidence,
                reason=census.reason,
                objects=census.objects,
                relations=census.relations,
                raw_response=raw_content,
                latency_sec=latency_sec,
            )
        return census

    def _run_image_prompt(self, *, image: Image.Image, prompt: str, max_tokens: int = 160) -> str:
        with self._GLOBAL_CALL_LOCK:
            response = self._client.analyze_image(
                image,
                prompt,
                temperature=0.1,
                max_tokens=int(max_tokens),
            )
        return str((response or {}).get("content", "") or "").strip()

    def _run_text_prompt(self, *, prompt: str) -> str:
        with self._GLOBAL_CALL_LOCK:
            response = self._client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=220,
            )
        return str((response or {}).get("content", "") or "").strip()

    def _named_container_prompt(self, *, target_text: str) -> str:
        _ = str(target_text or "").strip().lower()
        return (
            "Does this crop show a large open-top neutral metal cup, beaker, or container?\n"
            'Return ONLY JSON like {"is_container": true, "confidence": 0.85, "reason": "short phrase"}. '
            "Use a real confidence value, not the example."
        )

    def _post_place_prompt(self, *, object_text: str, container_text: str) -> str:
        return (
            "You are checking the final state of a robot tabletop task.\n"
            f"Decide whether the object '{object_text}' appears to be inside the container '{container_text}'.\n"
            "Answer false if the image is ambiguous, the object is outside the rim, or the object cannot be seen clearly.\n"
            "Use a confidence value from 0.0 to 1.0; do not copy the example value.\n"
            "Return ONLY compact JSON with this exact schema:\n"
            '{"object_in_container": true, "confidence": 0.85, "reason": "short phrase"}'
        )

    def _scene_target_prompt(
        self,
        *,
        command_text: str,
        target_text: str,
        roles: Sequence[str],
        candidates: Sequence[LocalVLCandidate],
    ) -> str:
        candidate_lines: list[str] = []
        for index, candidate in enumerate(candidates, start=1):
            metadata = dict(candidate.metadata or {})
            candidate_lines.append(
                json.dumps(
                    {
                        "number": index,
                        "candidate_id": str(candidate.candidate_id),
                        "label": str(candidate.label or ""),
                        "source": str(candidate.source or ""),
                        "center_uv": list(candidate.center_uv) if candidate.center_uv is not None else None,
                        "bbox_xyxy": list(candidate.bbox_xyxy) if candidate.bbox_xyxy is not None else None,
                        "color": str(metadata.get("color", metadata.get("proposal_color", "")) or ""),
                        "shape": str(metadata.get("shape", metadata.get("proposal_shape", "")) or ""),
                    },
                    sort_keys=True,
                )
            )
        return (
            "You are selecting visual targets for a robot tabletop manipulation task.\n"
            f"Robot command: '{str(command_text or target_text or '').strip()}'.\n"
            f"Requested target text: '{str(target_text or '').strip()}'.\n"
            f"Roles to fill: {', '.join(str(role) for role in roles)}.\n"
            "Use the annotated full-scene image and the contact sheet. Each box/crop number maps to one candidate_id.\n"
            "Select only from the candidate_id values in the catalog. Do not invent object ids, coordinates, keepouts, or motions.\n"
            "For target_container when the request says cup, beaker, mug, glass, bowl, flask, or container: prefer a large open-top neutral metal/white/gray container; reject small colored cylinders, cubes, blocks, boxes, and generic colored tabletop props.\n"
            "For source_object or avoid_object, choose the candidate that best matches the visual object role in the command.\n"
            "Use confidence values from 0.0 to 1.0; do not copy the example values.\n"
            f"Candidate catalog:\n{chr(10).join(candidate_lines)}\n"
            "Return ONLY compact JSON with this exact schema:\n"
            '{"selections":[{"role":"target_container","candidate_id":"id","confidence":0.85,"reason":"short phrase"}],"confidence":0.85,"reason":"short phrase"}'
        )

    def _hypothesis_prompt(
        self,
        *,
        command_text: str,
        target_text: str,
        role: str,
        candidate_limit: int,
    ) -> str:
        return (
            "You are helping a robot ground a tabletop object request under uncertainty.\n"
            f"Robot command: '{command_text or target_text}'.\n"
            f"Focus object role: '{role}'.\n"
            f"Reference text: '{target_text}'.\n"
            "Generate short detector-friendly phrases that could match the same physical target, including partial-visibility wording when useful.\n"
            "You may suggest close synonyms, geometry cues, occlusion cues, and bounded approach hints.\n"
            "Approach hints must stay high level, for example: favor_side_access, top_access_occluded, reobserve_before_descent, keep_clear_of_occluder.\n"
            "Do not invent coordinates, trajectories, or low-level actions.\n"
            "Use a confidence value from 0.0 to 1.0; do not copy the example value.\n"
            f"Return at most {candidate_limit} query terms.\n"
            "Return ONLY compact JSON with this exact schema:\n"
            '{"query_terms":["term"],"distractor_terms":["term"],"occlusion_hints":["hint"],"geometry_hints":["hint"],"approach_hints":["hint"],"confidence":0.85,"reason":"short phrase"}'
        )

    def _scene_census_prompt(
        self,
        *,
        candidates: Sequence[LocalVLCandidate],
    ) -> str:
        candidate_lines: list[str] = []
        for candidate in candidates:
            bbox = (
                list(candidate.bbox_xyxy)
                if isinstance(candidate.bbox_xyxy, tuple)
                else candidate.bbox_xyxy
            )
            candidate_lines.append(
                json.dumps(
                    {
                        "candidate_id": str(candidate.candidate_id),
                        "label": str(candidate.label or ""),
                        "center_uv": list(candidate.center_uv) if candidate.center_uv is not None else None,
                        "bbox_xyxy": bbox,
                    },
                    sort_keys=True,
                )
            )
        return (
            "You are building a compact scene census for a robot tabletop task from a full camera image.\n"
            "Only use candidate_ids from the provided candidate catalog.\n"
            "You may refine object identity, report partial occlusion, and emit high-level access hints.\n"
            "Allowed relation labels: left_of, right_of, near, occludes, blocks_top_access, inside.\n"
            "Allowed access hints: favor_side_access, top_access_occluded, keep_clear_of_occluder, reobserve_before_descent.\n"
            "Do not invent coordinates, new object ids, or motion commands.\n"
            "Use confidence values from 0.0 to 1.0; do not copy the example values.\n"
            f"Candidate catalog:\n{chr(10).join(candidate_lines)}\n"
            "Return ONLY compact JSON with this exact schema:\n"
            '{"objects":[{"candidate_id":"id","refined_label":"short label","visibility":"clear|partial|occluded","occluded_by_ids":["id"],"access_hints":["hint"],"reason":"short phrase"}],"relations":[{"subject_id":"id","relation":"near","object_id":"id","confidence":0.85,"reason":"short phrase"}],"confidence":0.85,"reason":"short phrase"}'
        )

    def _scene_target_image(
        self,
        *,
        image: Image.Image,
        candidates: Sequence[LocalVLCandidate],
    ) -> Image.Image:
        base = image.convert("RGB")
        image_w, image_h = base.size
        annotated = base.copy()
        draw = ImageDraw.Draw(annotated)
        palette = (
            (255, 214, 10),
            (255, 69, 58),
            (52, 199, 89),
            (64, 156, 255),
            (191, 90, 242),
            (255, 159, 10),
        )
        for index, candidate in enumerate(candidates, start=1):
            bbox = self._candidate_bbox_for_drawing(image=base, candidate=candidate)
            if bbox is None:
                continue
            color = palette[(index - 1) % len(palette)]
            x0, y0, x1, y1 = bbox
            for inset in range(3):
                draw.rectangle((x0 - inset, y0 - inset, x1 + inset, y1 + inset), outline=color)
            label = f"#{index}"
            text_bbox = draw.textbbox((0, 0), label)
            tw = int(text_bbox[2] - text_bbox[0])
            th = int(text_bbox[3] - text_bbox[1])
            draw.rectangle((x0, max(0, y0 - th - 6), x0 + tw + 8, max(th + 6, y0)), fill=color)
            draw.text((x0 + 4, max(0, y0 - th - 4)), label, fill=(0, 0, 0))

        thumb_size = 112
        label_h = 36
        columns = min(4, max(1, len(candidates)))
        rows = int(math.ceil(float(len(candidates)) / float(columns)))
        sheet_w = max(image_w, columns * thumb_size)
        sheet_h = rows * (thumb_size + label_h)
        sheet = Image.new("RGB", (sheet_w, sheet_h), (250, 250, 250))
        sheet_draw = ImageDraw.Draw(sheet)
        for index, candidate in enumerate(candidates, start=1):
            row = (index - 1) // columns
            col = (index - 1) % columns
            left = col * thumb_size
            top = row * (thumb_size + label_h)
            crop = self._candidate_crop(image=base, candidate=candidate, context_scale=1.25)
            if crop is None:
                crop = Image.new("RGB", (thumb_size, thumb_size), (230, 230, 230))
            crop = crop.convert("RGB")
            crop.thumbnail((thumb_size, thumb_size), resample=getattr(getattr(Image, "Resampling", Image), "LANCZOS"))
            paste_x = left + max(0, (thumb_size - crop.size[0]) // 2)
            paste_y = top + max(0, (thumb_size - crop.size[1]) // 2)
            sheet.paste(crop, (paste_x, paste_y))
            sheet_draw.rectangle((left, top, left + thumb_size - 1, top + thumb_size - 1), outline=(80, 80, 80))
            label = f"#{index} {candidate.candidate_id}"
            if candidate.label:
                label = f"{label} {str(candidate.label)[:26]}"
            sheet_draw.text((left + 3, top + thumb_size + 3), label[:34], fill=(0, 0, 0))

        combined = Image.new("RGB", (max(image_w, sheet_w), image_h + sheet_h), (255, 255, 255))
        combined.paste(annotated, (0, 0))
        combined.paste(sheet, (0, image_h))
        max_dim = max(768, self._max_crop_px)
        current_max = max(combined.size)
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        if current_max == max_dim or (current_max < max_dim and max_dim <= 1280):
            return combined
        scale = float(max_dim) / float(current_max)
        return combined.resize(
            (max(1, int(round(combined.size[0] * scale))), max(1, int(round(combined.size[1] * scale)))),
            resample=resample,
        )

    def _candidate_bbox_for_drawing(
        self,
        *,
        image: Image.Image,
        candidate: LocalVLCandidate,
    ) -> tuple[int, int, int, int] | None:
        image_w, image_h = image.size
        bbox = candidate.bbox_xyxy
        if bbox is None and candidate.center_uv is not None:
            cx = int(candidate.center_uv[0])
            cy = int(candidate.center_uv[1])
            half = max(20, min(image_w, image_h) // 16)
            bbox = (cx - half, cy - half, cx + half, cy + half)
        if bbox is None:
            return None
        try:
            x0, y0, x1, y1 = [int(v) for v in bbox]
        except Exception:
            return None
        x0 = max(0, min(image_w - 1, x0))
        y0 = max(0, min(image_h - 1, y0))
        x1 = max(0, min(image_w - 1, x1))
        y1 = max(0, min(image_h - 1, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _parse_json_result(self, *, raw_content: str, yes_key: str) -> dict[str, Any]:
        content = str(raw_content or "").strip()
        lowered = content.lower()
        if not content:
            return {"status": "empty", "flag": False, "confidence": 0.0, "reason": "empty response"}
        if content.startswith("Error:"):
            if "timed out" in lowered or "timeout" in lowered:
                return {"status": "timeout", "flag": False, "confidence": 0.0, "reason": content}
            return {"status": "error", "flag": False, "confidence": 0.0, "reason": content}
        try:
            json_text = self._extract_json_text(content)
            payload = json.loads(json_text)
        except Exception as exc:
            return {"status": "parse_fail", "flag": False, "confidence": 0.0, "reason": str(exc)}
        confidence = self._safe_confidence(payload.get("confidence", 0.0))
        flag_value = payload.get(yes_key, False)
        if yes_key == "contains_target":
            flag_value = payload.get("contains_target", payload.get("is_container", False))
        return {
            "status": "ok",
            "flag": bool(flag_value),
            "confidence": confidence,
            "reason": str(payload.get("reason", "") or "").strip(),
        }

    def _parse_hypothesis_result(self, *, raw_content: str, candidate_limit: int) -> dict[str, Any]:
        content = str(raw_content or "").strip()
        lowered = content.lower()
        if not content:
            return {"status": "empty", "confidence": 0.0, "reason": "empty response"}
        if content.startswith("Error:"):
            if "timed out" in lowered or "timeout" in lowered:
                return {"status": "timeout", "confidence": 0.0, "reason": content}
            return {"status": "error", "confidence": 0.0, "reason": content}
        try:
            json_text = self._extract_json_text(content)
            payload = json.loads(json_text)
        except Exception as exc:
            return {"status": "parse_fail", "confidence": 0.0, "reason": str(exc)}
        return {
            "status": "ok",
            "confidence": self._safe_confidence(payload.get("confidence", 0.0)),
            "reason": str(payload.get("reason", "") or "").strip(),
            "query_terms": self._dedup_terms(payload.get("query_terms"), limit=candidate_limit),
            "distractor_terms": self._dedup_terms(payload.get("distractor_terms"), limit=max(2, candidate_limit // 2)),
            "occlusion_hints": self._dedup_terms(payload.get("occlusion_hints"), limit=max(2, candidate_limit // 2)),
            "geometry_hints": self._dedup_terms(payload.get("geometry_hints"), limit=max(2, candidate_limit // 2)),
            "approach_hints": self._dedup_terms(payload.get("approach_hints"), limit=max(2, candidate_limit // 2)),
        }

    def _parse_scene_target_result(
        self,
        *,
        raw_content: str,
        valid_roles: set[str],
        valid_candidate_ids: set[str],
    ) -> dict[str, Any]:
        content = str(raw_content or "").strip()
        lowered = content.lower()
        if not content:
            return {"status": "empty", "confidence": 0.0, "reason": "empty response"}
        if content.startswith("Error:"):
            if "timed out" in lowered or "timeout" in lowered:
                return {"status": "timeout", "confidence": 0.0, "reason": content}
            return {"status": "error", "confidence": 0.0, "reason": content}
        try:
            json_text = self._extract_json_text(content)
            payload = json.loads(json_text)
        except Exception as exc:
            return {"status": "parse_fail", "confidence": 0.0, "reason": str(exc)}

        parsed_selections: list[LocalVLRoleSelection] = []
        raw_selections = payload.get("selections", None)
        if raw_selections is None:
            raw_selections = []
            for role in valid_roles:
                raw_id = payload.get(role, None)
                if raw_id is not None:
                    raw_selections.append(
                        {
                            "role": role,
                            "candidate_id": raw_id,
                            "confidence": payload.get("confidence", 0.0),
                            "reason": payload.get("reason", ""),
                        }
                    )
        for item in list(raw_selections or []):
            if not isinstance(item, dict):
                continue
            role = " ".join(str(item.get("role", "") or "").strip().lower().split())
            if role not in valid_roles:
                continue
            candidate_id = str(item.get("candidate_id", "") or "").strip()
            if candidate_id in {"", "none", "null", "no_target"}:
                candidate_id = ""
            status = "ok"
            if candidate_id and candidate_id not in valid_candidate_ids:
                status = "invalid_candidate"
            parsed_selections.append(
                LocalVLRoleSelection(
                    role=role,
                    candidate_id=candidate_id or None,
                    confidence=self._safe_confidence(item.get("confidence", payload.get("confidence", 0.0))),
                    status=status,
                    reason=str(item.get("reason", "") or "").strip(),
                )
            )
        return {
            "status": "ok",
            "confidence": self._safe_confidence(payload.get("confidence", 0.0)),
            "reason": str(payload.get("reason", "") or "").strip(),
            "selections": tuple(parsed_selections),
        }

    def _parse_scene_census_result(
        self,
        *,
        raw_content: str,
        valid_candidate_ids: set[str],
    ) -> dict[str, Any]:
        content = str(raw_content or "").strip()
        lowered = content.lower()
        if not content:
            return {"status": "empty", "confidence": 0.0, "reason": "empty response"}
        if content.startswith("Error:"):
            if "timed out" in lowered or "timeout" in lowered:
                return {"status": "timeout", "confidence": 0.0, "reason": content}
            return {"status": "error", "confidence": 0.0, "reason": content}
        try:
            json_text = self._extract_json_text(content)
            payload = json.loads(json_text)
        except Exception as exc:
            return {"status": "parse_fail", "confidence": 0.0, "reason": str(exc)}

        parsed_objects: list[LocalVLSceneObjectAnnotation] = []
        for item in list(payload.get("objects") or []):
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id", "") or "").strip()
            if candidate_id not in valid_candidate_ids:
                continue
            occluded_by_ids = tuple(
                candidate
                for candidate in self._dedup_terms(item.get("occluded_by_ids"), limit=4)
                if candidate in valid_candidate_ids and candidate != candidate_id
            )
            access_hints = tuple(
                hint
                for hint in self._dedup_terms(item.get("access_hints"), limit=4)
                if hint in self._SCENE_ACCESS_HINTS
            )
            visibility = " ".join(str(item.get("visibility", "unknown") or "unknown").strip().lower().split())
            if visibility not in {"clear", "partial", "occluded"}:
                visibility = "unknown"
            parsed_objects.append(
                LocalVLSceneObjectAnnotation(
                    candidate_id=candidate_id,
                    refined_label=" ".join(str(item.get("refined_label", "") or "").strip().lower().split()),
                    visibility=visibility,
                    occluded_by_ids=occluded_by_ids,
                    access_hints=access_hints,
                    reason=str(item.get("reason", "") or "").strip(),
                )
            )

        parsed_relations: list[LocalVLSceneRelation] = []
        for item in list(payload.get("relations") or []):
            if not isinstance(item, dict):
                continue
            subject_id = str(item.get("subject_id", "") or "").strip()
            relation = " ".join(str(item.get("relation", "") or "").strip().lower().split())
            object_id = str(item.get("object_id", "") or "").strip()
            if (
                subject_id not in valid_candidate_ids
                or object_id not in valid_candidate_ids
                or relation not in self._SCENE_RELATIONS
            ):
                continue
            parsed_relations.append(
                LocalVLSceneRelation(
                    subject_id=subject_id,
                    relation=relation,
                    object_id=object_id,
                    confidence=self._safe_confidence(item.get("confidence", 0.0)),
                    reason=str(item.get("reason", "") or "").strip(),
                )
            )

        return {
            "status": "ok",
            "confidence": self._safe_confidence(payload.get("confidence", 0.0)),
            "reason": str(payload.get("reason", "") or "").strip(),
            "objects": tuple(parsed_objects),
            "relations": tuple(parsed_relations),
        }

    def _candidate_role_selection_allowed(
        self,
        *,
        role: str,
        command_text: str,
        target_text: str,
        candidate: LocalVLCandidate,
    ) -> tuple[bool, str]:
        normalized_role = " ".join(str(role or "").strip().lower().split())
        if normalized_role != "target_container":
            return True, ""
        request_text = f"{command_text or ''} {target_text or ''}".lower()
        if not any(token in request_text for token in ("cup", "mug", "glass", "beaker", "bowl", "flask", "container")):
            return True, ""
        identity_text = self._candidate_identity_text(candidate)
        exact_named = any(token in identity_text for token in ("cup", "mug", "glass", "beaker", "bowl", "flask"))
        if exact_named:
            return True, ""
        if any(token in identity_text for token in ("cube", "block", "box")):
            return False, "box-like candidate cannot satisfy explicit named container request"
        color_hint = self._candidate_color_hint(candidate)
        if color_hint and any(token in identity_text for token in ("cylinder", "object", "thing", "item", "target")):
            return False, "colored generic cylinder/object cannot satisfy explicit named container request"
        if color_hint and not any(token in identity_text for token in ("container", "jar", "can", "bottle")):
            return False, "colored generic tabletop candidate cannot satisfy explicit named container request"
        if any(token in identity_text for token in ("cylinder", "object", "container", "jar", "can", "bottle", "unknown")):
            return True, ""
        return False, "candidate lacks open-container identity"

    @staticmethod
    def _candidate_identity_text(candidate: LocalVLCandidate) -> str:
        metadata = dict(candidate.metadata or {})
        parts = [
            candidate.label,
            candidate.source,
            metadata.get("canonical_label", ""),
            metadata.get("refined_label", ""),
            metadata.get("shape", ""),
            metadata.get("proposal_shape", ""),
            metadata.get("color", ""),
            metadata.get("proposal_color", ""),
        ]
        return " ".join(str(part or "").strip().lower().replace("-", " ") for part in parts if str(part or "").strip())

    @staticmethod
    def _candidate_color_hint(candidate: LocalVLCandidate) -> str:
        metadata = dict(candidate.metadata or {})
        explicit = str(
            metadata.get("color", "")
            or metadata.get("proposal_color", "")
            or metadata.get("dominant_color", "")
            or ""
        ).strip().lower()
        if explicit and explicit not in {"unknown", "neutral", "none", "gray", "grey", "white", "black", "silver", "metal", "metallic"}:
            return explicit
        identity_text = LocalVisionVerifier._candidate_identity_text(candidate)
        for color in ("red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan"):
            if color in identity_text.split():
                return color
        return ""

    @staticmethod
    def _extract_json_text(content: str) -> str:
        if "```json" in content:
            return content.split("```json", 1)[1].split("```", 1)[0].strip()
        if "```" in content:
            return content.split("```", 1)[1].split("```", 1)[0].strip()
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return content[start : end + 1]
        return content

    @staticmethod
    def _safe_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except Exception:
            return 0.0
        if not math.isfinite(confidence):
            return 0.0
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _dedup_terms(values: Any, *, limit: int) -> tuple[str, ...]:
        if isinstance(values, str):
            items = [values]
        elif isinstance(values, Sequence):
            items = [str(v or "") for v in values]
        else:
            items = []
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = " ".join(str(item or "").strip().lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
            if len(out) >= max(1, int(limit)):
                break
        return tuple(out)

    def _candidate_crop(
        self,
        *,
        image: Image.Image,
        candidate: LocalVLCandidate,
        context_scale: float = 1.4,
    ) -> Image.Image | None:
        image_w, image_h = image.size
        bbox = candidate.bbox_xyxy
        if bbox is None and candidate.center_uv is not None:
            cx = int(candidate.center_uv[0])
            cy = int(candidate.center_uv[1])
            half = max(32, min(image_w, image_h) // 10)
            bbox = (cx - half, cy - half, cx + half, cy + half)
        if bbox is None:
            return None
        try:
            x0, y0, x1, y1 = [int(v) for v in bbox]
        except Exception:
            return None
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        pad_x = int(round(width * max(0.0, context_scale - 1.0)))
        pad_y = int(round(height * max(0.0, context_scale - 1.0)))
        left = max(0, x0 - pad_x)
        top = max(0, y0 - pad_y)
        right = min(image_w, x1 + pad_x)
        bottom = min(image_h, y1 + pad_y)
        if right <= left or bottom <= top:
            return None
        crop = image.crop((left, top, right, bottom))
        crop_w, crop_h = crop.size
        max_dim = max(crop_w, crop_h)
        if max_dim <= self._max_crop_px:
            return crop
        scale = float(self._max_crop_px) / float(max_dim)
        resized = crop.resize(
            (max(1, int(round(crop_w * scale))), max(1, int(round(crop_h * scale)))),
            resample=getattr(getattr(Image, "Resampling", Image), "LANCZOS"),
        )
        return resized
