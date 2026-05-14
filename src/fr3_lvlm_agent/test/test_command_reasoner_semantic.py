from __future__ import annotations

from fr3_lvlm_agent.reasoning.command_reasoner import (
    CommandReasoner,
    is_target_supported,
    score_label_against_target,
)


def test_reasoner_normalizes_red_block_to_cube() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    target = reasoner.parse("pick up the red block")
    assert target.target_color == "red"
    assert target.target_shape == "cube"
    assert "red cube" in target.canonical_terms


def test_reasoner_normalizes_blue_can_to_cylinder() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    target = reasoner.parse("pick up blue can")
    assert target.target_color == "blue"
    assert target.target_shape == "cylinder"
    assert "blue cylinder" in target.canonical_terms


def test_reasoner_parses_smaller_cube_relative_size_scope() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    target = reasoner.parse("pick up the smaller cube")
    assert target.target_shape == "cube"
    assert target.relative_size == "smaller"
    assert target.target_class_scope == "shape"


def test_reasoner_parses_small_and_large_relative_size_scope() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    small = reasoner.parse("pick up the small red cube")
    large = reasoner.parse("pick up the large red cube")
    assert small.target_color == "red"
    assert small.target_shape == "cube"
    assert small.relative_size == "smaller"
    assert large.target_color == "red"
    assert large.target_shape == "cube"
    assert large.relative_size == "larger"


def test_reasoner_parses_smallest_object_relative_size_scope() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    target = reasoner.parse("pick up the smallest object")
    assert target.target_shape is None
    assert target.relative_size == "smallest"
    assert target.target_class_scope == "object"


def test_reasoner_parses_closest_to_cup_relation() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    target = reasoner.parse("pick up the red cube closest to the cup")
    assert target.target_color == "red"
    assert target.target_shape == "cube"
    assert target.spatial_relation == "closest_to"
    assert target.spatial_reference == "cup"


def test_reasoner_keeps_bare_red_cube_non_size_constrained() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    target = reasoner.parse("pick up the red cube")
    assert target.target_color == "red"
    assert target.target_shape == "cube"
    assert target.relative_size is None


def test_semantic_scoring_rejects_unmatched_label() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    target = reasoner.parse("pick up red cylinder")
    score = score_label_against_target("kite", target, min_score=1.0, synonym_map_enabled=True)
    assert score.semantic_pass is False
    assert score.score < 1.0


def test_semantic_scoring_accepts_shape_synonym() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    target = reasoner.parse("pick up red cylinder")
    score_shape_only = score_label_against_target("bottle", target, min_score=1.0, synonym_map_enabled=True)
    assert score_shape_only.shape_match is True
    assert score_shape_only.semantic_pass is False
    score_full = score_label_against_target("red bottle", target, min_score=1.0, synonym_map_enabled=True)
    assert score_full.shape_match is True
    assert score_full.color_match is True
    assert score_full.semantic_pass is True


def test_target_support_requires_known_color_and_shape() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    target_ok = reasoner.parse("pick up red cube")
    target_bad = reasoner.parse("pick up spoon")
    supported = {"cube", "cylinder"}
    colors = {"red", "blue", "green", "yellow", "orange", "purple"}
    assert is_target_supported(target_ok, supported_shapes=supported, supported_colors=colors)
    assert not is_target_supported(target_bad, supported_shapes=supported, supported_colors=colors)
