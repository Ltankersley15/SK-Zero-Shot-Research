"""Unit tests for YOLO-World open-vocabulary prompt generation."""

import numpy as np
from types import SimpleNamespace

from fr3_lvlm_agent.perception.proposal_yolo import YOLOWorldProposalGenerator
from fr3_lvlm_agent.perception.proposal_yolo import _infer_color_from_rgb_patch
from fr3_lvlm_agent.reasoning.command_reasoner import CommandReasoner


def test_build_class_prompts_uses_noun_phrase_and_terms() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    target = reasoner.parse("pick up the striped coffee mug")

    generator = YOLOWorldProposalGenerator.__new__(YOLOWorldProposalGenerator)
    prompts = generator._build_class_prompts(target)  # pylint: disable=protected-access

    assert prompts
    assert "striped coffee mug" in prompts
    assert "coffee mug" in prompts or "striped coffee" in prompts
    assert "object" in prompts


def test_build_class_prompts_deduplicates() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    target = reasoner.parse("pick up red cube")

    generator = YOLOWorldProposalGenerator.__new__(YOLOWorldProposalGenerator)
    prompts = generator._build_class_prompts(target)  # pylint: disable=protected-access

    assert len(prompts) == len(set(prompts))
    assert "cube" in prompts
    assert len(prompts) <= 16


def test_build_class_prompts_keeps_generic_anchor_terms() -> None:
    reasoner = CommandReasoner(default_terms=["object"], synonym_map_enabled=True)
    target = reasoner.parse("pick up the striped textured translucent faceted decorative coffee mug")

    generator = YOLOWorldProposalGenerator.__new__(YOLOWorldProposalGenerator)
    prompts = generator._build_class_prompts(target)  # pylint: disable=protected-access

    assert "object" in prompts
    assert "cube" in prompts
    assert "cylinder" in prompts


def test_infer_color_from_rgb_patch_detects_red() -> None:
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    patch[:, :, 0] = 220
    patch[:, :, 1] = 20
    patch[:, :, 2] = 20

    color, conf = _infer_color_from_rgb_patch(patch)

    assert color == "red"
    assert conf > 0.3


def test_infer_color_from_rgb_patch_detects_green() -> None:
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    patch[:, :, 0] = 25
    patch[:, :, 1] = 190
    patch[:, :, 2] = 30

    color, conf = _infer_color_from_rgb_patch(patch)

    assert color == "green"
    assert conf > 0.3


def test_infer_color_from_rgb_patch_detects_blue() -> None:
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    patch[:, :, 0] = 20
    patch[:, :, 1] = 50
    patch[:, :, 2] = 220

    color, conf = _infer_color_from_rgb_patch(patch)

    assert color == "blue"
    assert conf > 0.3


def test_infer_color_from_rgb_patch_uses_center_region() -> None:
    patch = np.zeros((30, 30, 3), dtype=np.uint8)
    patch[:, :, 0] = 220
    patch[:, :, 1] = 20
    patch[:, :, 2] = 20
    patch[10:20, 10:20, 0] = 20
    patch[10:20, 10:20, 1] = 50
    patch[10:20, 10:20, 2] = 220

    color, conf = _infer_color_from_rgb_patch(patch)

    assert color == "blue"
    assert conf > 0.2


def test_yolo_generator_uses_segmentation_mask_for_center_and_yaw() -> None:
    class _TensorWrap:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

        def __getitem__(self, item):
            return _TensorWrap(self._arr[item])

        def __float__(self):
            return float(self._arr.item())

        def __int__(self):
            return int(self._arr.item())

    class _Box:
        def __init__(self):
            self.xyxy = _TensorWrap([[20, 20, 80, 80]])
            self.conf = _TensorWrap([0.9])
            self.cls = _TensorWrap([0])

    class _Model:
        def set_classes(self, _prompts):
            return None

        def predict(self, _img, **_kwargs):
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[25:75, 35:65] = 1
            result = SimpleNamespace(
                boxes=[_Box()],
                names={0: "green cube"},
                masks=SimpleNamespace(data=[_TensorWrap(mask)]),
            )
            return [result]

    generator = YOLOWorldProposalGenerator.__new__(YOLOWorldProposalGenerator)
    generator.model = _Model()
    generator.conf_thres = 0.1
    generator.device = "cpu"
    target = CommandReasoner(default_terms=["object"], synonym_map_enabled=True).parse("pick up the green cube")

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[25:75, 35:65, 1] = 220
    proposals = generator(image, target)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.label == "green cube"
    assert proposal.proposal_shape_guess == "cube"
    assert proposal.center_uv == (50, 50)
    assert proposal.yaw_rad is not None
