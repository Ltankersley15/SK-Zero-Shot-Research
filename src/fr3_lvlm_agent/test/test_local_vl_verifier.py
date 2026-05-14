from __future__ import annotations

from PIL import Image

from fr3_lvlm_agent.reasoning.local_vl_verifier import LocalVLCandidate, LocalVisionVerifier


class _FakeClient:
    def __init__(self, responses=None, *, text_responses=None):
        self._responses = list(responses or [])
        self._text_responses = list(text_responses or [])
        self.calls = []

    def analyze_image(self, image, prompt, temperature=0.1, max_tokens=160):
        self.calls.append(
            {
                "image_size": image.size,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        content = self._responses.pop(0) if self._responses else ""
        return {"content": content}

    def chat(self, messages, temperature=0.1, max_tokens=220):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        content = self._text_responses.pop(0) if self._text_responses else ""
        return {"content": content}


def test_choose_named_container_selects_highest_confidence_candidate() -> None:
    client = _FakeClient(
        [
            '{"contains_target": false, "confidence": 0.15, "reason": "not the cup"}',
            '{"contains_target": true, "confidence": 0.87, "reason": "clear cup"}',
        ]
    )
    verifier = LocalVisionVerifier(client=client, max_crop_px=512, default_timeout_sec=8.0)
    image = Image.new("RGB", (640, 480), "white")

    decision = verifier.choose_named_container(
        image=image,
        target_text="cup",
        candidates=[
            LocalVLCandidate(candidate_id="cand_1", label="object", bbox_xyxy=(10, 10, 110, 160)),
            LocalVLCandidate(candidate_id="cand_2", label="cup", bbox_xyxy=(160, 40, 300, 260)),
        ],
        confidence_threshold=0.70,
    )

    assert decision.status == "ok"
    assert decision.selected_candidate_id == "cand_2"
    assert len(decision.candidate_scores) == 2


def test_choose_named_container_reports_timeout_when_all_candidates_timeout() -> None:
    client = _FakeClient(
        [
            "Error: request timed out",
            "Error: request timed out",
        ]
    )
    verifier = LocalVisionVerifier(client=client)
    image = Image.new("RGB", (640, 480), "white")

    decision = verifier.choose_named_container(
        image=image,
        target_text="cup",
        candidates=[
            LocalVLCandidate(candidate_id="cand_1", label="cup", bbox_xyxy=(10, 10, 110, 160)),
            LocalVLCandidate(candidate_id="cand_2", label="object", bbox_xyxy=(160, 40, 300, 260)),
        ],
        confidence_threshold=0.70,
    )

    assert decision.status == "timeout"
    assert decision.selected_candidate_id is None


def test_choose_named_container_reports_parse_fail_for_malformed_json() -> None:
    client = _FakeClient(
        [
            "definitely a cup",
            "still definitely a cup",
        ]
    )
    verifier = LocalVisionVerifier(client=client)
    image = Image.new("RGB", (640, 480), "white")

    decision = verifier.choose_named_container(
        image=image,
        target_text="cup",
        candidates=[
            LocalVLCandidate(candidate_id="cand_1", label="cup", bbox_xyxy=(10, 10, 110, 160)),
            LocalVLCandidate(candidate_id="cand_2", label="object", bbox_xyxy=(160, 40, 300, 260)),
        ],
        confidence_threshold=0.70,
    )

    assert decision.status == "parse_fail"
    assert decision.selected_candidate_id is None


def test_choose_named_container_resizes_large_crop_to_budget() -> None:
    client = _FakeClient(['{"contains_target": true, "confidence": 0.81, "reason": "clear"}'])
    verifier = LocalVisionVerifier(client=client, max_crop_px=256)
    image = Image.new("RGB", (1600, 1200), "white")

    decision = verifier.choose_named_container(
        image=image,
        target_text="cup",
        candidates=[LocalVLCandidate(candidate_id="cand_1", label="cup", bbox_xyxy=(100, 100, 1200, 900))],
        confidence_threshold=0.70,
    )

    assert decision.status == "ok"
    assert max(client.calls[0]["image_size"]) <= 256


def test_choose_scene_targets_selects_large_neutral_container_surrogate() -> None:
    client = _FakeClient(
        [
            """{
                "selections": [
                    {
                        "role": "target_container",
                        "candidate_id": "neutral_cylinder",
                        "confidence": 0.82,
                        "reason": "large open metal container"
                    }
                ],
                "confidence": 0.82,
                "reason": "best open container candidate"
            }"""
        ]
    )
    verifier = LocalVisionVerifier(client=client, max_crop_px=512)
    image = Image.new("RGB", (640, 480), "white")

    decision = verifier.choose_scene_targets(
        image=image,
        command_text="pick up the blue cube and place it in the cup",
        target_text="cup",
        roles=("target_container",),
        candidates=[
            LocalVLCandidate(
                candidate_id="yellow_cylinder",
                label="yellow cylinder",
                bbox_xyxy=(20, 20, 90, 120),
                metadata={"color": "yellow", "shape": "cylinder"},
            ),
            LocalVLCandidate(
                candidate_id="neutral_cylinder",
                label="cylinder",
                bbox_xyxy=(220, 80, 360, 300),
                metadata={"color": "silver", "shape": "cylinder"},
            ),
        ],
        confidence_threshold=0.55,
    )

    selection = decision.selection_for_role("target_container")
    assert decision.status == "ok"
    assert selection is not None
    assert selection.candidate_id == "neutral_cylinder"
    assert client.calls[0]["max_tokens"] == 360
    assert client.calls[0]["image_size"][1] > 480


def test_choose_scene_targets_rejects_colored_cylinder_for_explicit_cup() -> None:
    client = _FakeClient(
        [
            """{
                "selections": [
                    {
                        "role": "target_container",
                        "candidate_id": "yellow_cylinder",
                        "confidence": 0.91,
                        "reason": "looks cup-like"
                    }
                ],
                "confidence": 0.91,
                "reason": "selected colored cylinder"
            }"""
        ]
    )
    verifier = LocalVisionVerifier(client=client)
    image = Image.new("RGB", (640, 480), "white")

    decision = verifier.choose_scene_targets(
        image=image,
        command_text="place it in the cup",
        target_text="cup",
        roles=("target_container",),
        candidates=[
            LocalVLCandidate(
                candidate_id="yellow_cylinder",
                label="yellow cylinder",
                bbox_xyxy=(20, 20, 90, 120),
                metadata={"color": "yellow", "shape": "cylinder"},
            )
        ],
        confidence_threshold=0.55,
    )

    selection = decision.selection_for_role("target_container")
    assert decision.status == "rejected_candidate"
    assert selection is not None
    assert selection.candidate_id is None
    assert selection.status == "rejected_candidate"


def test_choose_scene_targets_rejects_box_candidate_for_explicit_cup() -> None:
    client = _FakeClient(
        [
            """{
                "selections": [
                    {
                        "role": "target_container",
                        "candidate_id": "green_box",
                        "confidence": 0.86,
                        "reason": "box is near target area"
                    }
                ],
                "confidence": 0.86
            }"""
        ]
    )
    verifier = LocalVisionVerifier(client=client)
    image = Image.new("RGB", (640, 480), "white")

    decision = verifier.choose_scene_targets(
        image=image,
        command_text="place it in the beaker",
        target_text="beaker",
        roles=("target_container",),
        candidates=[
            LocalVLCandidate(
                candidate_id="green_box",
                label="green cube",
                bbox_xyxy=(20, 20, 90, 120),
                metadata={"color": "green", "shape": "box"},
            )
        ],
        confidence_threshold=0.55,
    )

    selection = decision.selection_for_role("target_container")
    assert decision.status == "rejected_candidate"
    assert selection is not None
    assert selection.candidate_id is None
    assert "box-like" in selection.reason


def test_choose_scene_targets_returns_no_target_on_low_confidence_and_parse_failure() -> None:
    image = Image.new("RGB", (640, 480), "white")
    low_client = _FakeClient(
        [
            """{
                "selections": [
                    {
                        "role": "target_container",
                        "candidate_id": "neutral",
                        "confidence": 0.42
                    }
                ],
                "confidence": 0.42
            }"""
        ]
    )
    low_verifier = LocalVisionVerifier(client=low_client)

    low_decision = low_verifier.choose_scene_targets(
        image=image,
        command_text="place it in the cup",
        target_text="cup",
        roles=("target_container",),
        candidates=[LocalVLCandidate(candidate_id="neutral", label="cylinder", bbox_xyxy=(20, 20, 90, 120))],
        confidence_threshold=0.55,
    )

    parse_verifier = LocalVisionVerifier(client=_FakeClient(["not json"]))
    parse_decision = parse_verifier.choose_scene_targets(
        image=image,
        command_text="place it in the cup",
        target_text="cup",
        roles=("target_container",),
        candidates=[LocalVLCandidate(candidate_id="neutral", label="cylinder", bbox_xyxy=(20, 20, 90, 120))],
        confidence_threshold=0.55,
    )

    assert low_decision.status == "low_confidence"
    assert low_decision.selection_for_role("target_container").candidate_id is None
    assert parse_decision.status == "parse_fail"
    assert parse_decision.selection_for_role("target_container") is None


def test_post_place_check_returns_low_confidence_without_overclaiming() -> None:
    client = _FakeClient(['{"object_in_container": true, "confidence": 0.31, "reason": "hard to tell"}'])
    verifier = LocalVisionVerifier(client=client, max_crop_px=512)
    image = Image.new("RGB", (640, 480), "white")

    result = verifier.check_object_in_container(
        image=image,
        object_text="red cube",
        container_text="cup",
        candidate=LocalVLCandidate(candidate_id="cand_1", label="cup", bbox_xyxy=(120, 80, 260, 260)),
        confidence_threshold=0.70,
    )

    assert result.status == "low_confidence"
    assert result.object_in_container is None
    assert result.confidence == 0.31


def test_build_scene_census_parses_candidate_relations_and_access_hints() -> None:
    client = _FakeClient(
        [
            """{
                "objects": [
                    {
                        "candidate_id": "scene_obj_001",
                        "refined_label": "green cube",
                        "visibility": "partial",
                        "occluded_by_ids": ["scene_obj_002"],
                        "access_hints": ["favor_side_access", "keep_clear_of_occluder"],
                        "reason": "cup covers top"
                    }
                ],
                "relations": [
                    {
                        "subject_id": "scene_obj_002",
                        "relation": "blocks_top_access",
                        "object_id": "scene_obj_001",
                        "confidence": 0.82,
                        "reason": "cup overlaps target"
                    }
                ],
                "confidence": 0.84,
                "reason": "green cube partially blocked by cup"
            }"""
        ]
    )
    verifier = LocalVisionVerifier(client=client, max_crop_px=512, default_timeout_sec=8.0)
    image = Image.new("RGB", (640, 480), "white")

    census = verifier.build_scene_census(
        image=image,
        candidates=[
            LocalVLCandidate(candidate_id="scene_obj_001", label="green cube", bbox_xyxy=(40, 40, 120, 120)),
            LocalVLCandidate(candidate_id="scene_obj_002", label="cup", bbox_xyxy=(130, 20, 240, 180)),
        ],
        confidence_threshold=0.70,
    )

    assert census.status == "ok"
    assert census.confidence == 0.84
    assert census.objects[0].candidate_id == "scene_obj_001"
    assert census.objects[0].occluded_by_ids == ("scene_obj_002",)
    assert "favor_side_access" in census.objects[0].access_hints
    assert census.relations[0].relation == "blocks_top_access"
    assert census.relations[0].subject_id == "scene_obj_002"


def test_build_scene_census_ignores_unknown_candidate_ids() -> None:
    client = _FakeClient(
        [
            """{
                "objects": [
                    {
                        "candidate_id": "unknown_id",
                        "refined_label": "ghost cube",
                        "visibility": "clear",
                        "occluded_by_ids": [],
                        "access_hints": ["favor_side_access"]
                    }
                ],
                "relations": [
                    {
                        "subject_id": "unknown_id",
                        "relation": "near",
                        "object_id": "scene_obj_001",
                        "confidence": 0.75
                    }
                ],
                "confidence": 0.66,
                "reason": "mostly ignored"
            }"""
        ]
    )
    verifier = LocalVisionVerifier(client=client, max_crop_px=512, default_timeout_sec=8.0)
    image = Image.new("RGB", (640, 480), "white")

    census = verifier.build_scene_census(
        image=image,
        candidates=[LocalVLCandidate(candidate_id="scene_obj_001", label="green cube", bbox_xyxy=(40, 40, 120, 120))],
        confidence_threshold=0.50,
    )

    assert census.status == "ok"
    assert census.objects == ()
    assert census.relations == ()
