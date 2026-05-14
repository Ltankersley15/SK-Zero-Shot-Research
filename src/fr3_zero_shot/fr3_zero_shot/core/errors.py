from __future__ import annotations


class PipelineError(RuntimeError):
    """Base exception for unrecoverable pipeline errors."""


class FailClosedError(PipelineError):
    """Raised when deterministic safety gates reject a requested action."""

