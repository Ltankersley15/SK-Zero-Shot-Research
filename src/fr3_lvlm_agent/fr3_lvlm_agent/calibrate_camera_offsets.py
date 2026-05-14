#!/usr/bin/env python3
"""
Calibrate camera-frame translation offsets by comparing detected target positions.
against known ground-truth positions in Isaac Sim.

This node listens to /lvlm_agent/target_info (JSON String) published by
fr3_lvlm_agent and computes a recommended camera_x/y/z_offset.
"""

import json
import os
from typing import Dict, List

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _default_expected_targets() -> Dict[str, List[float]]:
    # Defaults from fr3_test.usda (workspace root).
    return {
        "red": [0.4, 0.2, 0.02],
        "green": [0.6, -0.3, 0.02],
        "blue": [0.5, 0.4, 0.02],
        "yellow": [0.35, -0.25, 0.02],
    }


def _load_expected_from_json(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[float]] = {}
    for k, v in (data or {}).items():
        if not isinstance(v, (list, tuple)) or len(v) < 3:
            continue
        out[str(k).strip().lower()] = [float(v[0]), float(v[1]), float(v[2])]
    return out


def _parse_usd_positions(path: str) -> Dict[str, List[float]]:
    # Minimal USD parser for xformOp:translate under known prims.
    targets = {}
    if not os.path.exists(path):
        return targets
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("def ") and "\"" in line:
                name = line.split("\"")[1]
                current = name
            if current and line.startswith("double3 xformOp:translate"):
                # Format: double3 xformOp:translate = (x, y, z)
                try:
                    rhs = line.split("=")[1].strip()
                    rhs = rhs.lstrip("(").rstrip(")")
                    parts = [p.strip() for p in rhs.split(",")]
                    if len(parts) >= 3:
                        targets[current] = [float(parts[0]), float(parts[1]), float(parts[2])]
                except Exception:
                    pass
    # Map known prims to color names.
    mapped = {}
    for prim, color in [
        ("red_block", "red"),
        ("green_block", "green"),
        ("blue_block", "blue"),
        ("cylinder", "yellow"),
    ]:
        if prim in targets:
            mapped[color] = targets[prim]
    return mapped


class CameraOffsetCalibrator(Node):
    def __init__(self):
        super().__init__("camera_offset_calibrator")

        self.declare_parameter("target_info_topic", "/lvlm_agent/target_info")
        self.declare_parameter("expected_targets_path", "")
        self.declare_parameter("usd_path", "")
        self.declare_parameter("report_every", 5)
        self.declare_parameter("min_samples", 3)
        self.declare_parameter("output_path", "")

        topic = str(self.get_parameter("target_info_topic").value)
        expected_path = str(self.get_parameter("expected_targets_path").value)
        usd_path = str(self.get_parameter("usd_path").value)
        self.report_every = max(1, int(self.get_parameter("report_every").value))
        self.min_samples = max(1, int(self.get_parameter("min_samples").value))
        self.output_path = str(self.get_parameter("output_path").value)

        self.expected = self._load_expected(expected_path, usd_path)
        if not self.expected:
            self.get_logger().error("No expected target positions loaded; calibration disabled.")

        self.samples: List[dict] = []
        self.samples_by_color: Dict[str, List[dict]] = {}

        self.sub = self.create_subscription(String, topic, self._on_info, 10)
        self.get_logger().info(f"Listening on {topic} for target_info.")
        self.get_logger().info(f"Expected targets: {self.expected}")

    def _load_expected(self, expected_path: str, usd_path: str) -> Dict[str, List[float]]:
        expected: Dict[str, List[float]] = {}
        if expected_path:
            if os.path.exists(expected_path):
                expected = _load_expected_from_json(expected_path)
            else:
                self.get_logger().warn(f"expected_targets_path not found: {expected_path}")
        if not expected and usd_path:
            expected = _parse_usd_positions(usd_path)
            if not expected:
                self.get_logger().warn(f"usd_path provided but no targets parsed: {usd_path}")
        if not expected:
            expected = _default_expected_targets()
        return expected

    def _on_info(self, msg: String) -> None:
        if not self.expected:
            return
        try:
            info = json.loads(msg.data)
        except Exception:
            return

        color = str(info.get("target_color", "")).strip().lower()
        if not color or color not in self.expected:
            return

        xyz_base = info.get("xyz_base") or info.get("xyz_base_raw")
        R_flat = info.get("R_base_from_cam")
        if xyz_base is None or R_flat is None:
            return

        try:
            xyz_base = np.array([float(xyz_base[0]), float(xyz_base[1]), float(xyz_base[2])], dtype=np.float64)
            R = np.array(R_flat, dtype=np.float64).reshape((3, 3))
        except Exception:
            return

        expected = np.array(self.expected[color], dtype=np.float64)
        err_base = expected - xyz_base
        delta_cam = R.T @ err_base

        sample = {
            "color": color,
            "err_base": err_base,
            "delta_cam": delta_cam,
            "R": R,
        }
        self.samples.append(sample)
        self.samples_by_color.setdefault(color, []).append(sample)

        n = len(self.samples)
        self.get_logger().info(
            f"[CALIB] {color} sample {n}: err_base=({err_base[0]:+.4f},{err_base[1]:+.4f},{err_base[2]:+.4f}) "
            f"delta_cam=({delta_cam[0]:+.4f},{delta_cam[1]:+.4f},{delta_cam[2]:+.4f})"
        )

        if n % self.report_every == 0 and n >= self.min_samples:
            self._report()

    def _report(self) -> None:
        if not self.samples:
            return
        deltas = np.array([s["delta_cam"] for s in self.samples], dtype=np.float64)
        mean = np.mean(deltas, axis=0)
        std = np.std(deltas, axis=0)

        # Compute RMS base error before/after applying mean offset.
        errs_before = []
        errs_after = []
        for s in self.samples:
            err_base = s["err_base"]
            R = s["R"]
            errs_before.append(err_base)
            errs_after.append(err_base - (R @ mean))
        errs_before = np.array(errs_before, dtype=np.float64)
        errs_after = np.array(errs_after, dtype=np.float64)
        rms_before = np.sqrt(np.mean(np.sum(errs_before ** 2, axis=1)))
        rms_after = np.sqrt(np.mean(np.sum(errs_after ** 2, axis=1)))

        self.get_logger().info(
            "[CALIB] Recommended camera offsets (apply to camera_x/y/z_offset): "
            f"x={mean[0]:+.4f} y={mean[1]:+.4f} z={mean[2]:+.4f} (std: {std[0]:.4f},{std[1]:.4f},{std[2]:.4f})"
        )
        self.get_logger().info(
            f"[CALIB] RMS base error: before={rms_before:.4f}m after={rms_after:.4f}m"
        )

        for color, samples in self.samples_by_color.items():
            deltas_c = np.array([s["delta_cam"] for s in samples], dtype=np.float64)
            mean_c = np.mean(deltas_c, axis=0)
            self.get_logger().info(
                f"[CALIB] {color} mean delta_cam: x={mean_c[0]:+.4f} y={mean_c[1]:+.4f} z={mean_c[2]:+.4f}"
            )

        if self.output_path:
            try:
                out = {
                    "camera_offset": [float(mean[0]), float(mean[1]), float(mean[2])],
                    "camera_offset_std": [float(std[0]), float(std[1]), float(std[2])],
                    "rms_before_m": float(rms_before),
                    "rms_after_m": float(rms_after),
                    "samples": len(self.samples),
                }
                out_dir = os.path.dirname(self.output_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(self.output_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)
                self.get_logger().info(f"[CALIB] Wrote summary to {self.output_path}")
            except Exception as e:
                self.get_logger().warn(f"[CALIB] Failed to write output: {e}")


def main() -> None:
    rclpy.init()
    node = CameraOffsetCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
