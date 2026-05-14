from __future__ import annotations

from fr3_lvlm_agent.debug_overlay import should_emit_debug_overlay


def test_debug_overlay_gate_requires_debug_flag() -> None:
    assert should_emit_debug_overlay(False, False) is False


def test_debug_overlay_gate_allows_save_only() -> None:
    assert should_emit_debug_overlay(True, False) is True


def test_debug_overlay_gate_allows_publish_only() -> None:
    assert should_emit_debug_overlay(False, True) is True


def test_debug_overlay_gate_allows_both() -> None:
    assert should_emit_debug_overlay(True, True) is True
