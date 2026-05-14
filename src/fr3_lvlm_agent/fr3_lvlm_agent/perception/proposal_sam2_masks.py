from __future__ import annotations

import colorsys
import importlib
import math
from dataclasses import dataclass

import numpy as np

try:
    cv = importlib.import_module("cv" + "2")
except Exception:
    cv = None

try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except Exception:
    SAM2AutomaticMaskGenerator = None  # type: ignore

from ..reasoning.command_reasoner import TargetSpec, score_label_against_target
from .proposal import Proposal


def infer_canonical_color_from_hsv(
    h: float, s: float, v: float, *, saturation_min: float = 45.0, value_min: float = 35.0
) -> tuple[str, float]:
    """Map HSV mean to canonical tabletop color label."""
    if (not np.isfinite(h)) or (not np.isfinite(s)) or (not np.isfinite(v)):
        return "unknown", 0.0
    if s < saturation_min or v < value_min:
        return "unknown", 0.0

    hh = float(h)
    if hh <= 10.0 or hh >= 170.0:
        color = "red"
    elif hh < 24.0:
        color = "orange"
    elif hh < 40.0:
        color = "yellow"
    elif hh < 86.0:
        color = "green"
    elif hh < 136.0:
        color = "blue"
    elif hh < 170.0:
        color = "purple"
    else:
        color = "unknown"

    if color == "unknown":
        return "unknown", 0.0
    conf = min(1.0, max(0.0, 0.65 * (s / 255.0) + 0.35 * (v / 255.0)))
    return color, float(conf)


def _mask_perimeter(mask_bool: np.ndarray) -> float:
    if cv is not None:
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        contours, _ = cv.findContours(mask_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        return float(sum(cv.arcLength(c, True) for c in contours))
    m = mask_bool.astype(np.uint8)
    edge = np.abs(np.diff(m, axis=0)).sum() + np.abs(np.diff(m, axis=1)).sum()
    return float(edge)


def classify_shape_from_mask(mask_u8: np.ndarray) -> tuple[str, float, dict[str, float]]:
    """Classify primitive shape from binary mask geometry."""
    m = np.asarray(mask_u8, dtype=np.uint8) > 0
    ys, xs = np.where(m)
    if xs.size <= 0 or ys.size <= 0:
        return "unknown", 0.0, {"aspect_ratio": 0.0, "extent": 0.0, "solidity": 0.0, "circularity": 0.0}

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1.0, float(x1 - x0 + 1))
    bh = max(1.0, float(y1 - y0 + 1))
    area = float(xs.size)
    bbox_area = max(1.0, bw * bh)
    aspect_ratio = bw / bh
    extent = area / bbox_area
    perimeter = max(1e-6, _mask_perimeter(m))
    circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
    circularity = float(max(0.0, min(circularity, 1.5)))

    if cv is not None:
        mask_u8_cv = (m.astype(np.uint8) * 255)
        contours, _ = cv.findContours(mask_u8_cv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv.contourArea)
            hull = cv.convexHull(c)
            hull_area = float(max(1e-6, cv.contourArea(hull)))
            solidity = float(max(0.0, min(1.0, area / hull_area)))
            
            # Use Hough lines to find the dominant straight edge orientation
            edges = cv.Canny(mask_u8_cv, 50, 150)
            lines = cv.HoughLinesP(edges, 1, math.pi/180, threshold=15, minLineLength=15, maxLineGap=5)
            yaw_rad = None
            if lines is not None:
                angles = []
                weights = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = math.hypot(x2 - x1, y2 - y1)
                    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    # Normalize to [-45, 45) for cube edges
                    ang = (ang + 45) % 90 - 45
                    angles.append(ang)
                    weights.append(length)
                if angles:
                    rads = np.radians(angles) * 4  # multiply by 4 to map [-45, 45) to [-180, 180)
                    x_mean = np.sum(np.array(weights) * np.cos(rads))
                    y_mean = np.sum(np.array(weights) * np.sin(rads))
                    dom_rad = math.atan2(y_mean, x_mean) / 4.0
                    yaw_rad = float(dom_rad)
            features = {"yaw_rad": yaw_rad}
        else:
            solidity = float(max(0.0, min(1.0, extent)))
            features = {}
    else:
        solidity = float(max(0.0, min(1.0, extent)))
        features = {}

    cube_score = 0.0
    if 0.72 <= aspect_ratio <= 1.35:
        cube_score += 0.45
    if extent >= 0.62:
        cube_score += 0.30
    if solidity >= 0.84:
        cube_score += 0.20
    if circularity < 0.82:
        cube_score += 0.10
    cube_score = float(max(0.0, min(1.0, cube_score)))

    cylinder_score = 0.0
    if 0.60 <= aspect_ratio <= 1.70:
        cylinder_score += 0.20
    if circularity >= 0.72:
        cylinder_score += 0.50
    if solidity >= 0.86:
        cylinder_score += 0.20
    if extent <= 0.88:
        cylinder_score += 0.10
    cylinder_score = float(max(0.0, min(1.0, cylinder_score)))

    max_shape_score = max(cube_score, cylinder_score)
    ambiguity_margin = 0.30
    if max_shape_score < 0.45:
        shape = "unknown"
        score = max_shape_score
    elif abs(cube_score - cylinder_score) < ambiguity_margin:
        # Ambiguous silhouettes (common in top-down masks) should not force
        # an explicit cube-vs-cylinder mismatch in strict semantic mode.
        shape = "unknown"
        score = max_shape_score
    elif cube_score >= cylinder_score:
        shape = "cube"
        score = cube_score
    else:
        shape = "cylinder"
        score = cylinder_score

    features.update({
        "aspect_ratio": float(aspect_ratio),
        "extent": float(extent),
        "solidity": float(solidity),
        "circularity": float(circularity),
        "cube_score": float(cube_score),
        "cylinder_score": float(cylinder_score),
    })
    return shape, float(score), features


def adaptive_edge_margin(
    base_margin_px: int,
    semantic_score: float,
    mask_stability: float,
    *,
    enabled: bool,
    semantic_threshold: float,
    stability_threshold: float,
    override_margin_px: int,
) -> int:
    if not enabled:
        return int(base_margin_px)
    if semantic_score < semantic_threshold or mask_stability < stability_threshold:
        return int(base_margin_px)
    return int(max(0, min(int(base_margin_px), int(override_margin_px))))


@dataclass
class SAM2MaskProposalGenerator:
    model: object
    max_candidates: int = 40
    min_area_frac: float = 0.0002
    max_area_frac: float = 0.30

    def __post_init__(self) -> None:
        if SAM2AutomaticMaskGenerator is None:
            raise RuntimeError("SAM2AutomaticMaskGenerator is unavailable; install sam2 with automatic mask support.")
        self._generator = SAM2AutomaticMaskGenerator(self.model)

    def _mean_hsv(self, image_rgb: np.ndarray, mask_bool: np.ndarray) -> tuple[float, float, float]:
        if cv is not None:
            hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)
            vals = hsv[mask_bool]
            if vals.size <= 0:
                return float("nan"), float("nan"), float("nan")
            return float(np.mean(vals[:, 0])), float(np.mean(vals[:, 1])), float(np.mean(vals[:, 2]))
        vals = image_rgb[mask_bool]
        if vals.size <= 0:
            return float("nan"), float("nan"), float("nan")
        r = float(np.mean(vals[:, 0])) / 255.0
        g = float(np.mean(vals[:, 1])) / 255.0
        b = float(np.mean(vals[:, 2])) / 255.0
        hh, ss, vv = colorsys.rgb_to_hsv(r, g, b)
        return float(hh * 179.0), float(ss * 255.0), float(vv * 255.0)

    def propose(
        self,
        image: object,
        target: TargetSpec,
        *,
        min_score: float,
        synonym_map_enabled: bool,
        supported_colors: set[str],
        supported_shapes: set[str],
    ) -> list[Proposal]:
        rgb = np.array(image, copy=True)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return []

        if hasattr(self._generator, "set_image"):
            try:
                self._generator.set_image(rgb)
            except Exception:
                pass
        elif hasattr(self._generator, "predictor") and hasattr(self._generator.predictor, "set_image"):
            try:
                self._generator.predictor.set_image(rgb)
            except Exception:
                pass

        h, w = rgb.shape[:2]
        masks = self._generator.generate(rgb)
        if not masks:
            return []

        # Prefer masks with higher model confidence first.
        ranked_masks = sorted(
            masks,
            key=lambda m: (
                float(m.get("predicted_iou", 0.0)) + float(m.get("stability_score", 0.0)),
                float(m.get("area", 0.0)),
            ),
            reverse=True,
        )

        proposals: list[Proposal] = []
        for m in ranked_masks:
            if len(proposals) >= int(self.max_candidates):
                break
            seg = np.asarray(m.get("segmentation", None))
            if seg.ndim != 2:
                continue
            mask_bool = seg.astype(bool)
            area = float(np.count_nonzero(mask_bool))
            area_frac = area / float(max(1, w * h))
            if area_frac < float(self.min_area_frac) or area_frac > float(self.max_area_frac):
                continue

            bbox = m.get("bbox", [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
            x0 = max(0, min(int(round(float(bbox[0]))), w - 1))
            y0 = max(0, min(int(round(float(bbox[1]))), h - 1))
            bw = max(1, int(round(float(bbox[2]))))
            bh = max(1, int(round(float(bbox[3]))))
            x1 = max(x0 + 1, min(x0 + bw, w - 1))
            y1 = max(y0 + 1, min(y0 + bh, h - 1))

            ys, xs = np.where(mask_bool)
            if xs.size <= 0:
                continue
            cx = int(round(float(np.mean(xs))))
            cy = int(round(float(np.mean(ys))))

            hh, ss, vv = self._mean_hsv(rgb, mask_bool)
            color_guess, color_score = infer_canonical_color_from_hsv(hh, ss, vv)
            if color_guess not in supported_colors:
                color_guess = "unknown"
                color_score = 0.0

            shape_guess, shape_score, shape_features = classify_shape_from_mask(mask_bool.astype(np.uint8) * 255)
            if shape_guess not in supported_shapes:
                shape_guess = "unknown"
                shape_score = 0.0

            label_parts = []
            if color_guess != "unknown":
                label_parts.append(color_guess)
            if shape_guess != "unknown":
                label_parts.append(shape_guess)
            label = " ".join(label_parts).strip() or "object"

            semantic = score_label_against_target(
                label=label,
                target=target,
                min_score=float(min_score),
                synonym_map_enabled=bool(synonym_map_enabled),
            )
            conf = float(max(0.0, min(1.0, float(m.get("predicted_iou", 0.0)))))
            stability = float(max(0.0, min(1.0, float(m.get("stability_score", 0.0)))))
            rank_score = float(
                semantic.score + (0.40 * conf) + (0.20 * stability) + (0.20 * color_score) + (0.20 * shape_score)
            )
            proposals.append(
                Proposal(
                    bbox_xyxy=(int(x0), int(y0), int(x1), int(y1)),
                    center_uv=(int(cx), int(cy)),
                    label=str(label),
                    confidence=float(conf),
                    source="sam2_masks",
                    score=float(rank_score),
                    matched=bool(semantic.semantic_pass),
                    semantic_score=float(semantic.score),
                    semantic_pass=bool(semantic.semantic_pass),
                    canonical_label=str(semantic.canonical_label),
                    proposal_color_guess=str(color_guess),
                    proposal_shape_guess=str(shape_guess),
                    proposal_color_score=float(color_score),
                    proposal_shape_score=float(shape_score),
                    proposal_shape_features=dict(shape_features),
                    yaw_rad=shape_features.get("yaw_rad", None),
                )
            )

        proposals.sort(key=lambda p: float(p.score), reverse=True)
        return proposals
