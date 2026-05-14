from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


COLOR_SYNONYMS: dict[str, set[str]] = {
    "red": {"red", "crimson", "scarlet", "ruby", "maroon"},
    "blue": {"blue", "navy", "azure", "cobalt"},
    "green": {"green", "emerald", "lime"},
    "yellow": {"yellow", "amber", "gold"},
    "orange": {"orange", "tangerine"},
    "purple": {"purple", "violet", "lavender"},
}

SHAPE_SYNONYMS: dict[str, set[str]] = {
    "cube": {"cube", "block", "box"},
    "cylinder": {"cylinder", "tube", "can", "bottle"},
}

RELATIVE_SIZE_SYNONYMS: dict[str, set[str]] = {
    "smaller": {"small", "smaller", "smaller-sized", "smaller_size", "smallerone", "little"},
    "smallest": {"smallest", "tiny", "tiniest"},
    "larger": {"large", "larger", "big", "bigger", "larger-sized", "larger_size", "biggerone"},
    "largest": {"largest", "biggest"},
}

_STOP_WORDS = {
    "pick",
    "pickup",
    "up",
    "move",
    "grab",
    "place",
    "put",
    "find",
    "locate",
    "detect",
    "track",
    "inspect",
    "please",
    "the",
    "a",
    "an",
    "and",
    "to",
    "from",
    "on",
    "in",
    "at",
    "near",
    "next",
    "beside",
    "behind",
    "under",
    "over",
    "left",
    "right",
    "front",
    "back",
    "with",
    "of",
}


def _invert_synonyms(groups: dict[str, set[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for canonical, terms in groups.items():
        c = str(canonical).strip().lower()
        if not c:
            continue
        out[c] = c
        for t in terms:
            ts = str(t).strip().lower()
            if ts:
                out[ts] = c
    return out


_COLOR_CANONICAL = _invert_synonyms(COLOR_SYNONYMS)
_SHAPE_CANONICAL = _invert_synonyms(SHAPE_SYNONYMS)


def canonicalize_token(token: str, synonym_map_enabled: bool = True) -> str:
    t = str(token or "").strip().lower()
    if not t:
        return ""
    if not synonym_map_enabled:
        return t
    if t in _COLOR_CANONICAL:
        return _COLOR_CANONICAL[t]
    if t in _SHAPE_CANONICAL:
        return _SHAPE_CANONICAL[t]
    return t


def canonicalize_label(label: str, synonym_map_enabled: bool = True) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(label or "").lower())
    return " ".join(canonicalize_token(t, synonym_map_enabled=synonym_map_enabled) for t in tokens).strip()


@dataclass
class TargetSpec:
    raw_command: str
    normalized_command: str
    noun_phrase: str
    target_terms: list[str]
    target_color: str | None
    target_shape: str | None
    canonical_terms: list[str]
    relative_size: str | None
    target_class_scope: str
    spatial_relation: str | None = None
    spatial_reference: str | None = None


@dataclass
class SemanticScore:
    score: float
    semantic_pass: bool
    color_match: bool
    shape_match: bool
    partial_match: bool
    canonical_label: str


class CommandReasoner:
    """Parse natural-language commands into object-centric target terms."""

    def __init__(
        self,
        default_terms: Iterable[str] | None = None,
        max_terms: int = 16,
        synonym_map_enabled: bool = True,
    ) -> None:
        self._default_terms = [str(t).strip().lower() for t in (default_terms or ["object"]) if str(t).strip()]
        if not self._default_terms:
            self._default_terms = ["object"]
        self._max_terms = max(1, int(max_terms))
        self._synonym_map_enabled = bool(synonym_map_enabled)

    def _dedup_limit(self, values: list[str]) -> list[str]:
        out: list[str] = []
        seen = set()
        for v in values:
            vs = str(v).strip().lower()
            if not vs or vs in seen:
                continue
            seen.add(vs)
            out.append(vs)
            if len(out) >= self._max_terms:
                break
        return out

    def _infer_color(self, tokens: list[str]) -> str | None:
        for t in tokens:
            ct = canonicalize_token(t, synonym_map_enabled=self._synonym_map_enabled)
            if ct in COLOR_SYNONYMS:
                return ct
        return None

    def _infer_shape(self, tokens: list[str]) -> str | None:
        for t in tokens:
            ct = canonicalize_token(t, synonym_map_enabled=self._synonym_map_enabled)
            if ct in SHAPE_SYNONYMS:
                return ct
        return None

    def _infer_relative_size(self, tokens: list[str]) -> str | None:
        normalized = [str(t or "").strip().lower() for t in tokens]
        for canonical, synonyms in RELATIVE_SIZE_SYNONYMS.items():
            if any(token in synonyms for token in normalized):
                return canonical
        return None

    def _infer_spatial_relation(self, clean_command: str) -> tuple[str | None, str | None]:
        text = str(clean_command or "").strip().lower()
        if not text:
            return None, None
        match = re.search(
            r"\b(?:closest|nearest)\s+(?:object\s+)?(?:to|near)\s+(?:the\s+|a\s+|an\s+)?(?P<ref>[a-z0-9 ]+)",
            text,
        )
        if match is None:
            match = re.search(
                r"\b(?:closest|nearest)\s+(?:the\s+|a\s+|an\s+)?(?P<ref>cup|mug|beaker|glass|bowl|container)\b",
                text,
            )
        if match is None:
            return None, None
        reference = re.split(
            r"\b(?:then|and|but|while|without|with|from|on|in|at|left|right|beside|behind|under|over)\b",
            str(match.group("ref") or ""),
            maxsplit=1,
        )[0]
        reference = re.sub(r"^(?:the|a|an)\s+", "", reference.strip())
        reference_tokens = [
            canonicalize_token(token, synonym_map_enabled=self._synonym_map_enabled)
            for token in re.findall(r"[a-z0-9]+", reference)
            if token
        ]
        if not reference_tokens:
            return None, None
        return "closest_to", " ".join(reference_tokens)

    def _infer_target_class_scope(
        self,
        *,
        target_shape: str | None,
        phrase_tokens: list[str],
    ) -> str:
        if target_shape:
            return "shape"
        normalized = {canonicalize_token(token, synonym_map_enabled=self._synonym_map_enabled) for token in phrase_tokens}
        if normalized & set(SHAPE_SYNONYMS):
            return "shape"
        return "object"

    def parse(self, command: str) -> TargetSpec:
        raw = str(command or "")
        cmd = raw.strip().lower()
        if not cmd:
            defaults = list(self._default_terms)
            return TargetSpec(
                raw_command=raw,
                normalized_command="",
                noun_phrase="",
                target_terms=defaults,
                target_color=None,
                target_shape=None,
                canonical_terms=defaults,
                relative_size=None,
                target_class_scope="object",
                spatial_relation=None,
                spatial_reference=None,
            )

        clean = re.sub(r"[^a-z0-9\s]", " ", cmd)
        clean = re.sub(r"\s+", " ", clean).strip()

        phrase = ""
        m = re.search(r"(?:pick(?:\s+up)?|grab|move|place|put|find|locate|detect|track|inspect)\s+(.+)", clean)
        if m:
            phrase = m.group(1).strip()
            phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase).strip()
            phrase = re.split(r"\b(?:from|to|on|in|at|near|next|beside|behind|under|over)\b", phrase)[0].strip()

        tokens = re.findall(r"[a-z0-9]+", clean)
        content = [t for t in tokens if t not in _STOP_WORDS]
        canonical_content = [
            canonicalize_token(t, synonym_map_enabled=self._synonym_map_enabled) for t in content
        ]
        phrase_tokens = re.findall(r"[a-z0-9]+", phrase)
        canonical_phrase_tokens = [
            canonicalize_token(t, synonym_map_enabled=self._synonym_map_enabled) for t in phrase_tokens
        ]
        target_color = self._infer_color(content)
        target_shape = self._infer_shape(content + phrase_tokens)
        relative_size = self._infer_relative_size(content + phrase_tokens)
        spatial_relation, spatial_reference = self._infer_spatial_relation(clean)
        target_class_scope = self._infer_target_class_scope(
            target_shape=target_shape,
            phrase_tokens=phrase_tokens,
        )

        canonical_terms: list[str] = []
        if target_color and target_shape:
            canonical_terms.append(f"{target_color} {target_shape}")
        if target_shape:
            canonical_terms.append(target_shape)
        if target_color:
            canonical_terms.append(target_color)
        if canonical_phrase_tokens:
            canonical_terms.append(" ".join(canonical_phrase_tokens))
        for n in (3, 2):
            for i in range(0, max(0, len(canonical_content) - n + 1)):
                canonical_terms.append(" ".join(canonical_content[i : i + n]))
        canonical_terms.extend(canonical_content)
        canonical_terms.extend(
            canonicalize_token(t, synonym_map_enabled=self._synonym_map_enabled) for t in self._default_terms
        )
        canonical_terms_out = self._dedup_limit(canonical_terms)

        terms: list[str] = list(canonical_terms_out)
        if phrase:
            terms.append(phrase)
        for n in (3, 2):
            for i in range(0, max(0, len(content) - n + 1)):
                terms.append(" ".join(content[i : i + n]))
        terms.extend(content)
        terms.extend(self._default_terms)
        out = self._dedup_limit(terms)

        return TargetSpec(
            raw_command=raw,
            normalized_command=clean,
            noun_phrase=phrase,
            target_terms=out,
            target_color=target_color,
            target_shape=target_shape,
            canonical_terms=canonical_terms_out,
            relative_size=relative_size,
            target_class_scope=target_class_scope,
            spatial_relation=spatial_relation,
            spatial_reference=spatial_reference,
        )


def score_label_against_target(
    label: str,
    target: TargetSpec,
    *,
    min_score: float = 1.0,
    synonym_map_enabled: bool = True,
) -> SemanticScore:
    canonical_label = canonicalize_label(label, synonym_map_enabled=synonym_map_enabled)
    label_tokens = [t for t in canonical_label.split(" ") if t]
    label_token_set = set(label_tokens)

    color_match = bool(target.target_color and target.target_color in label_token_set)
    shape_match = bool(target.target_shape and target.target_shape in label_token_set)
    label_has_explicit_shape = any(t in SHAPE_SYNONYMS for t in label_token_set)

    partial_match = False
    for term in list(target.canonical_terms or []) + list(target.target_terms or []):
        ts = canonicalize_label(term, synonym_map_enabled=synonym_map_enabled)
        if not ts:
            continue
        if ts == canonical_label or (ts in canonical_label) or (canonical_label in ts):
            partial_match = True
            break

    score = 0.0
    if shape_match:
        score += 2.0
    if color_match:
        score += 2.0
    if partial_match:
        score += 1.0
    if (not color_match) and (not shape_match) and (not partial_match):
        score -= 2.0

    # For explicit tabletop commands (e.g., "blue cube"), require color match
    # and only allow shape mismatch when the proposal shape is unknown/ambiguous.
    # This prevents near-miss labels like "blue cylinder" while avoiding false
    # rejects from noisy shape guesses.
    if target.target_color and target.target_shape:
        semantic_pass = bool(
            (score >= float(min_score))
            and color_match
            and (shape_match or (not label_has_explicit_shape))
        )
    elif target.target_shape:
        semantic_pass = bool(shape_match and (score >= float(min_score)))
    elif target.target_color:
        semantic_pass = bool(color_match and (score >= float(min_score)))
    else:
        semantic_pass = bool(score >= float(min_score))
    return SemanticScore(
        score=float(score),
        semantic_pass=semantic_pass,
        color_match=color_match,
        shape_match=shape_match,
        partial_match=partial_match,
        canonical_label=canonical_label,
    )


def is_target_supported(
    target: TargetSpec,
    *,
    supported_shapes: set[str],
    supported_colors: set[str],
    require_color: bool = True,
    require_shape: bool = True,
) -> bool:
    shape = str(target.target_shape or "").strip().lower()
    color = str(target.target_color or "").strip().lower()
    if require_shape and not shape:
        return False
    if require_color and not color:
        return False
    if shape and supported_shapes and shape not in supported_shapes:
        return False
    if color and supported_colors and color not in supported_colors:
        return False
    return True
