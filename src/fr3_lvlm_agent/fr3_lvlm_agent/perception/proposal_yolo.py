from __future__ import annotations

import colorsys
import os
import re
import importlib.util

import numpy as np

os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None  # type: ignore
from ..reasoning.command_reasoner import TargetSpec
from .proposal import Proposal
from .proposal_sam2_masks import classify_shape_from_mask


_COLOR_TERMS = {"red", "blue", "green", "yellow", "orange", "purple"}
_SHAPE_ALIASES = {
    "cube": "cube",
    "block": "cube",
    "box": "cube",
    "cylinder": "cylinder",
    "tube": "cylinder",
    "can": "cylinder",
    "bottle": "cylinder",
}


def _infer_color_from_rgb_patch(patch_rgb: np.ndarray) -> tuple[str, float]:
    if patch_rgb.ndim != 3 or patch_rgb.shape[2] < 3 or patch_rgb.size <= 0:
        return "unknown", 0.0
    work = np.asarray(patch_rgb[:, :, :3], dtype=np.float32)
    h_px, w_px = int(work.shape[0]), int(work.shape[1])
    if h_px >= 6 and w_px >= 6:
        y0 = int(round(0.3 * h_px))
        y1 = max(y0 + 1, int(round(0.7 * h_px)))
        x0 = int(round(0.3 * w_px))
        x1 = max(x0 + 1, int(round(0.7 * w_px)))
        work = work[y0:y1, x0:x1, :]

    pixels = work.reshape(-1, 3) / 255.0
    if pixels.size <= 0:
        return "unknown", 0.0
    p_max = np.max(pixels, axis=1)
    p_min = np.min(pixels, axis=1)
    delta = p_max - p_min
    sat = np.where(p_max > 1e-6, delta / np.maximum(p_max, 1e-6), 0.0)
    vivid_mask = (p_max > 0.15) & (sat > 0.08)
    if int(np.count_nonzero(vivid_mask)) >= 8:
        pixels = pixels[vivid_mask]

    mean_rgb = np.mean(pixels, axis=0)
    r = float(np.clip(mean_rgb[0], 0.0, 1.0))
    g = float(np.clip(mean_rgb[1], 0.0, 1.0))
    b = float(np.clip(mean_rgb[2], 0.0, 1.0))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.18 or v < 0.16:
        return "unknown", 0.0

    deg = float((h * 360.0) % 360.0)
    if deg < 20.0 or deg >= 340.0:
        color = "red"
    elif deg < 42.0:
        color = "orange"
    elif deg < 72.0:
        color = "yellow"
    elif deg < 165.0:
        color = "green"
    elif deg < 265.0:
        color = "blue"
    elif deg < 340.0:
        color = "purple"
    else:
        color = "unknown"

    conf = float(max(0.0, min(1.0, (0.65 * s) + (0.35 * v))))
    return color, conf


class YOLOWorldProposalGenerator:
    """YOLO-World based proposal generator for open-vocabulary detection."""

    def __init__(self, model_path: str | None = None, conf_thres: float = 0.15, device: str = "cpu"):
        self.model = None
        self.conf_thres = conf_thres
        self.device = str(device or "cpu").strip().lower()
        self._warmed_up = False
        
        if YOLO is None:
            print("Warning: Ultralytics not installed. YOLO proposal generator will fail.")
            return
        if importlib.util.find_spec("clip") is None:
            print("Warning: CLIP package not installed. Install with: pip install 'git+https://github.com/ultralytics/CLIP.git'")
            return

        if model_path is None:
             # Default to package share checkpoints
             try:
                 if get_package_share_directory is None:
                     raise RuntimeError("ament index unavailable")
                 share_dir = get_package_share_directory('fr3_lvlm_agent')
                 # UPGRADED: YOLOE-26s-seg -> YOLOE-26l-seg for a stronger open-vocabulary detector.
                 model_path = os.path.join(share_dir, 'checkpoints', 'yoloe-26l-seg.pt')
             except Exception as e:
                 print(f"Failed to resolve package share: {e}")
                 # Fallback to local relative path if running from source without install
                 # Keep the same default when the package share is unavailable.
                 model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'yoloe-26l-seg.pt')

        try:
            print(f"Loading YOLOE model from {model_path}")
            # Use YOLO class which handles loading .pt files
            self.model = YOLO(model_path)
            if self.model is not None:
                self.model.to(self.device)
        except Exception as e:
            print(f"Failed to load YOLOE model: {e}")

    def warmup(self) -> bool:
        if self.model is None:
            return False
        if self._warmed_up:
            return True
        try:
            self.model.set_classes(["object"])
            dummy = np.zeros((32, 32, 3), dtype=np.uint8)
            self.model.predict(dummy, conf=self.conf_thres, verbose=False, imgsz=32, max_det=1, device=self.device)
            self._warmed_up = True
            return True
        except Exception as e:
            print(f"YOLO warmup failed: {e}")
            return False

    def _build_class_prompts(self, target: TargetSpec) -> list[str]:
        primary: list[str] = []
        phrase = str(getattr(target, "noun_phrase", "") or "").strip().lower()
        if phrase:
            primary.append(phrase)
        for term in list(getattr(target, "canonical_terms", [])) + list(getattr(target, "target_terms", [])):
            t = str(term or "").strip().lower()
            if not t:
                continue
            t = re.sub(r"\s+", " ", t)
            primary.append(t)
        target_color = str(getattr(target, "target_color", "") or "").strip().lower()
        target_shape = str(getattr(target, "target_shape", "") or "").strip().lower()
        if target_shape:
            primary.append(target_shape)
        if target_color and target_shape:
            primary.append(f"{target_color} {target_shape}")
        anchors = ["object", "cube", "cylinder", "block", "box", "bottle", "can", "cup", "mug"]

        out: list[str] = []
        seen: set[str] = set()
        for p in primary:
            if len(p) < 2:
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
            if len(out) >= 12:
                break
        for p in anchors:
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out[:16] if out else ["object"]

    def __call__(self, image: object, target: TargetSpec) -> list[Proposal]:
        if self.model is None:
            return []

        # Convert PIL to numpy (RGB)
        if hasattr(image, "convert"):
            img_np = np.array(image.convert("RGB"))
        else:
            img_np = np.array(image)

        prompts = self._build_class_prompts(target)

        try:
            self.model.set_classes(prompts)
        except Exception as e:
            if "same device" in str(e).lower():
                try:
                    self.device = "cpu"
                    self.model.to(self.device)
                    self.model.set_classes(prompts)
                except Exception as fallback_error:
                    print(f"YOLO set_classes failed after cpu fallback: {fallback_error}")
                    return []
            else:
                print(f"YOLO set_classes failed: {e}")
                return []

        # Inference
        # verbose=False to reduce log spam
        try:
            results = self.model.predict(img_np, conf=self.conf_thres, verbose=False, device=self.device)
        except Exception as e:
            print(f"YOLO predict failed: {e}")
            return []
        
        proposals = []
        for result in results:
            masks_data = None
            if getattr(result, "masks", None) is not None:
                masks_data = getattr(result.masks, "data", None)
            for idx, box in enumerate(result.boxes):
                coords = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names[cls_id] if hasattr(result, "names") else prompts[0]
                label_tokens = set(re.findall(r"[a-z0-9]+", str(label).lower()))
                label_color_guess = "unknown"
                shape_guess = "unknown"
                for tok in label_tokens:
                    if tok in _COLOR_TERMS:
                        label_color_guess = tok
                    if tok in _SHAPE_ALIASES:
                        shape_guess = _SHAPE_ALIASES[tok]

                img_h, img_w = int(img_np.shape[0]), int(img_np.shape[1])
                x0, y0, x1, y1 = map(int, coords)
                x0 = int(np.clip(x0, 0, max(0, img_w - 1)))
                y0 = int(np.clip(y0, 0, max(0, img_h - 1)))
                x1 = int(np.clip(x1, x0 + 1, max(x0 + 1, img_w)))
                y1 = int(np.clip(y1, y0 + 1, max(y0 + 1, img_h)))
                if x1 <= x0 or y1 <= y0:
                    continue

                patch = img_np[y0:y1, x0:x1, :3]
                patch_color, patch_color_score = _infer_color_from_rgb_patch(patch)
                
                # Prefer YOLO's color detection if available and confident
                # Fall back to HSV-based color only if YOLO didn't detect color
                if label_color_guess != "unknown" and conf > 0.3:
                    color_guess = label_color_guess
                    color_score = conf
                elif patch_color != "unknown":
                    color_guess = patch_color
                    color_score = patch_color_score
                else:
                    color_guess = "unknown"
                    color_score = 0.0

                shape_score = 0.35 if shape_guess != "unknown" else 0.0
                cx = (x0 + x1) // 2
                cy = (y0 + y1) // 2
                yaw_rad = None

                if masks_data is not None and idx < len(masks_data):
                    try:
                        mask_arr = masks_data[idx].cpu().numpy()
                        mask_u8 = (np.asarray(mask_arr) > 0.5).astype(np.uint8) * 255
                        if mask_u8.ndim == 2 and int(np.count_nonzero(mask_u8)) > 16:
                            mask_shape, mask_shape_score, mask_features = classify_shape_from_mask(mask_u8)
                            if mask_shape != "unknown":
                                shape_guess = str(mask_shape)
                                shape_score = max(float(shape_score), float(mask_shape_score))
                            yaw_rad = mask_features.get("yaw_rad", None)
                            ys, xs = np.where(mask_u8 > 0)
                            if xs.size > 0 and ys.size > 0:
                                cx = int(round(float(np.mean(xs))))
                                cy = int(round(float(np.mean(ys))))
                    except Exception:
                        pass

                # Create Proposal
                prop = Proposal(
                    bbox_xyxy=(x0, y0, x1, y1),
                    center_uv=(cx, cy),
                    label=label,
                    confidence=conf,
                    source="yolo_world",
                    score=conf,
                    matched=True,
                    semantic_score=conf,
                    semantic_pass=True,
                    canonical_label=label,
                    proposal_color_guess=color_guess,
                    proposal_shape_guess=shape_guess,
                    proposal_color_score=float(color_score),
                    proposal_shape_score=float(shape_score),
                    yaw_rad=yaw_rad,
                )
                proposals.append(prop)
                
        return proposals
