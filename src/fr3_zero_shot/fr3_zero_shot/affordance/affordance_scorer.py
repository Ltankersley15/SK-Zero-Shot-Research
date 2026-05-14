from __future__ import annotations

from dataclasses import dataclass

from fr3_zero_shot.core.types import ObjectHypothesis

from .container_access import ContainerAccess, evaluate_container_access
from .source_access import SourceAccess, evaluate_source_access


@dataclass(frozen=True)
class AffordanceSummary:
    source_access: SourceAccess | None = None
    container_access: ContainerAccess | None = None


class AffordanceScorer:
    def score(
        self,
        *,
        source: ObjectHypothesis | None,
        target: ObjectHypothesis | None,
        keepouts: tuple[ObjectHypothesis, ...] = (),
    ) -> AffordanceSummary:
        source_access = None if source is None else evaluate_source_access(source, keepouts)
        container_access = None if target is None else evaluate_container_access(target)
        return AffordanceSummary(source_access=source_access, container_access=container_access)

