from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
try:
    from hydra.errors import MissingConfigException
except Exception:
    class MissingConfigException(Exception):
        pass

from ..reasoning.command_reasoner import TargetSpec
from .proposal import Proposal

# Try importing SAM2; handle failure gracefully for systems still setting it up.
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False
    print("WARNING: 'sam2' module not found. SAM2 backend will not work.")


@dataclass
class TrackResult:
    bbox_xyxy: tuple[int, int, int, int]
    center_uv: tuple[int, int]
    yaw_rad: float | None
    mask_u8: np.ndarray
    mask_area_frac: float
    stability: float
    track_iou: float
    proposal: Proposal


class SAM2ImageBackend:
    """
    Wrapper for SAM 2 Image Predictor to support 'Tracking by Detection'.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        model_cfg: str = "configs/sam2/sam2_hiera_l.yaml",
        device: str = "cuda",
    ):
        if not HAS_SAM2:
            raise ImportError("SAM2 python module is not installed.")

        self.device = device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU for SAM2.")
            self.device = "cpu"

        # Resolve checkpoint path
        if checkpoint_path is None:
            # Default to a local 'checkpoints' folder in the package or workspace
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ckpt_dir = os.path.join(base_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            checkpoint_path = os.path.join(ckpt_dir, "sam2_hiera_large.pt")

        self._ensure_checkpoint(checkpoint_path)

        print(f"Loading SAM2 model from {checkpoint_path} on {self.device}...")
        self.model = self._build_model_with_fallback_cfg(model_cfg, checkpoint_path)
        self.predictor = SAM2ImagePredictor(self.model)
        print("SAM2 model loaded successfully.")

    def _build_model_with_fallback_cfg(self, model_cfg: str, checkpoint_path: str):
        """Build SAM2 model while tolerating legacy config names used in older code."""
        aliases = {
            "sam2_hiera_large.yaml": "configs/sam2/sam2_hiera_l.yaml",
            "sam2_hiera_small.yaml": "configs/sam2/sam2_hiera_s.yaml",
            "sam2_hiera_base_plus.yaml": "configs/sam2/sam2_hiera_b+.yaml",
            "sam2_hiera_tiny.yaml": "configs/sam2/sam2_hiera_t.yaml",
        }
        candidates: list[str] = []
        cfg = str(model_cfg or "").strip()
        if cfg in aliases:
            candidates.append(aliases[cfg])
        if cfg:
            candidates.append(cfg)
        if "configs/sam2/sam2_hiera_l.yaml" not in candidates:
            candidates.append("configs/sam2/sam2_hiera_l.yaml")

        last_err: Exception | None = None
        for cand in candidates:
            try:
                return build_sam2(cand, checkpoint_path, device=self.device)
            except MissingConfigException as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"Unable to resolve SAM2 config from candidates={candidates}")

    def _ensure_checkpoint(self, path: str):
        """Download the checkpoint if it doesn't exist."""
        if os.path.exists(path):
            return

        print(f"Checkpoint not found at {path}. Downloading sam2_hiera_large.pt...")
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, path)
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download SAM2 checkpoint: {e}")

    def get_mask(self, image: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
        """
        Predict mask for a given bounding box.
        
        Args:
            image: RGB numpy array (H, W, 3).
            bbox_xyxy: (x_min, y_min, x_max, y_max).
            
        Returns:
            Binary mask (H, W) as np.uint8 (0 or 255), or None if failure.
        """
        if image is None:
            return None

        # SAM2 expects image to be set first
        try:
            self.predictor.set_image(image)
            
            box = np.array(bbox_xyxy)
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )
            
            # masks is (1, H, W)
            if masks is None or masks.size == 0:
                return None
                
            best_mask = masks[0] # (H, W) bool
            return (best_mask.astype(np.uint8) * 255)
            
        except Exception as e:
            print(f"SAM2 Inference Error: {e}")
            return None


class SAM2ImageAdapter:
    """Single-frame SAM2 segmentation adapter."""

    runtime_backend = "sam2_image"

    def __init__(self, backend: SAM2ImageBackend) -> None:
        self._backend = backend

    def mask_from_bbox(self, image: object, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
        # Ensure writable contiguous memory for downstream torch conversion.
        arr = np.array(image, copy=True)
        if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
            return None
        mask = self._backend.get_mask(arr, bbox_xyxy)
        if mask is None:
            return None
        out = np.asarray(mask, dtype=np.uint8)
        if out.ndim != 2 or np.count_nonzero(out) <= 0:
            return None
        return out


class SAM2VideoAdapter:
    """
    Short-horizon SAM2 tracking-by-detection adapter.

    `SegmentTracker` handles temporal identity/reproposal. This adapter
    is responsible for per-frame SAM2 mask generation.
    """

    runtime_backend = "sam2_video"

    def __init__(self, backend: SAM2ImageBackend) -> None:
        self._backend = backend

    def mask_from_bbox(self, image: object, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
        # Ensure writable contiguous memory for downstream torch conversion.
        arr = np.array(image, copy=True)
        if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
            return None
        mask = self._backend.get_mask(arr, bbox_xyxy)
        if mask is None:
            return None
        out = np.asarray(mask, dtype=np.uint8)
        if out.ndim != 2 or np.count_nonzero(out) <= 0:
            return None
        return out


class SAM3VideoAdapter:
    """
    Placeholder for future SAM3 integration.

    Explicitly selectable so runtime behavior remains transparent.
    """

    runtime_backend = "sam3_video"

    def mask_from_bbox(self, image: object, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
        raise RuntimeError("sam3_video adapter is not implemented yet.")


class SegmentTracker:
    """
    SAM-style tracking interface.

    Current implementation keeps the interface stable and tracks masks across
    short frame windows. A future SAM3 adapter can swap internals without changing
    call sites.
    """

    def __init__(
        self,
        mask_from_bbox_fn: Callable[[object, tuple[int, int, int, int]], np.ndarray | None],
        mask_center_yaw_fn: Callable[[np.ndarray], tuple[tuple[int, int], float | None]],
        mask_iou_fn: Callable[[np.ndarray | None, np.ndarray | None], float],
        next_frame_fn: Callable[[], object | None],
        repropose_fn: Callable[[object, TargetSpec], list[Proposal]],
    ) -> None:
        self._mask_from_bbox_fn = mask_from_bbox_fn
        self._mask_center_yaw_fn = mask_center_yaw_fn
        self._mask_iou_fn = mask_iou_fn
        self._next_frame_fn = next_frame_fn
        self._repropose_fn = repropose_fn

    def track(
        self,
        image: object,
        proposal: Proposal,
        target: TargetSpec,
        max_frames: int,
        iou_min: float,
        stability_min: float,
    ) -> TrackResult | None:
        x0, y0, x1, y1 = proposal.bbox_xyxy
        mask = self._mask_from_bbox_fn(image, (x0, y0, x1, y1))
        if mask is None:
            return None

        center_uv, yaw = self._mask_center_yaw_fn(mask)
        prev_mask = mask
        final_mask = mask
        final_center = center_uv
        final_yaw = yaw
        final_bbox = (x0, y0, x1, y1)
        ious: list[float] = []
        stable_hits = 0

        for _ in range(1, max(1, int(max_frames))):
            frame = self._next_frame_fn()
            if frame is None:
                break

            proposals = self._repropose_fn(frame, target)
            if not proposals:
                break

            # Prefer proposal with best IoU overlap to keep temporal identity.
            best = proposals[0]
            best_iou = -1.0
            for p in proposals:
                px0, py0, px1, py1 = p.bbox_xyxy
                pmask = self._mask_from_bbox_fn(frame, (px0, py0, px1, py1))
                iou = self._mask_iou_fn(prev_mask, pmask)
                if iou > best_iou:
                    best_iou = iou
                    best = p
            bx0, by0, bx1, by1 = best.bbox_xyxy
            m = self._mask_from_bbox_fn(frame, (bx0, by0, bx1, by1))
            if m is None:
                break

            iou = float(self._mask_iou_fn(prev_mask, m))
            ious.append(iou)
            if iou >= float(iou_min):
                stable_hits += 1

            prev_mask = m
            final_mask = m
            final_bbox = (bx0, by0, bx1, by1)
            final_center, final_yaw = self._mask_center_yaw_fn(m)

        if ious:
            track_iou = float(np.mean(ious))
            stability = float(stable_hits) / float(len(ious))
        else:
            track_iou = 1.0
            stability = 1.0

        if track_iou < float(iou_min) or stability < float(stability_min):
            return None

        h, w = final_mask.shape[:2]
        area_frac = float(np.count_nonzero(final_mask)) / float(max(1, h * w))
        return TrackResult(
            bbox_xyxy=final_bbox,
            center_uv=(int(final_center[0]), int(final_center[1])),
            yaw_rad=final_yaw,
            mask_u8=np.asarray(final_mask, dtype=np.uint8),
            mask_area_frac=float(area_frac),
            stability=float(stability),
            track_iou=float(track_iou),
            proposal=proposal,
        )
