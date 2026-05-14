from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from fr3_zero_shot.core.types import ObjectHypothesis
from fr3_zero_shot.perception.container_detector import detect_dark_neutral_container_blobs

cv = importlib.import_module("cv" + "2")


@dataclass(frozen=True)
class LiveSceneCensusConfig:
    color_topic: str = '/fr3/d455/color/image_raw'
    depth_topic: str = '/fr3/d455/depth/image_raw'
    camera_info_topic: str = '/fr3/d455/color/camera_info'
    base_frame: str = 'fr3_link0'
    table_z_m: float = 0.02
    cube_height_m: float = 0.04
    min_blob_area_px: int = 500
    max_blob_area_px: int = 90000
    workspace_x_min_m: float = 0.25
    workspace_x_max_m: float = 0.90
    workspace_y_min_m: float = -0.45
    workspace_y_max_m: float = 0.45
    raw_z_min_m: float = 0.03
    raw_z_max_m: float = 0.35
    use_table_projection_xy: bool = True
    detect_neutral_containers: bool = True
    min_container_area_px: int = 800
    max_container_area_px: int = 150000


class LiveSceneCensusNode(Node):
    def __init__(self, config: LiveSceneCensusConfig | None = None) -> None:
        super().__init__('fr3_zero_shot_live_scene_census')
        self.config = LiveSceneCensusConfig() if config is None else config
        self.color_msg: Image | None = None
        self.depth_msg: Image | None = None
        self.camera_info: CameraInfo | None = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Image, self.config.color_topic, self._color_cb, 10)
        self.create_subscription(Image, self.config.depth_topic, self._depth_cb, 10)
        self.create_subscription(
            CameraInfo,
            self.config.camera_info_topic,
            self._camera_info_cb,
            10,
        )

    def capture(self, timeout_sec: float = 5.0) -> list[ObjectHypothesis]:
        deadline = self.get_clock().now().nanoseconds / 1e9 + float(timeout_sec)
        while rclpy.ok() and self.get_clock().now().nanoseconds / 1e9 < deadline:
            if self.color_msg and self.depth_msg and self.camera_info:
                break
            rclpy.spin_once(self, timeout_sec=0.05)
        if not (self.color_msg and self.depth_msg and self.camera_info):
            missing = []
            if self.color_msg is None:
                missing.append('color')
            if self.depth_msg is None:
                missing.append('depth')
            if self.camera_info is None:
                missing.append('camera_info')
            raise RuntimeError(f'missing live scene inputs: {", ".join(missing)}')
        self._wait_for_camera_tf(deadline)
        return self.detect_from_latest()

    def _wait_for_camera_tf(self, deadline_sec: float) -> None:
        frame = self.depth_msg.header.frame_id if self.depth_msg else ''
        while rclpy.ok() and self.get_clock().now().nanoseconds / 1e9 < deadline_sec:
            if self.tf_buffer.can_transform(
                self.config.base_frame,
                frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            ):
                return
            rclpy.spin_once(self, timeout_sec=0.05)

    def detect_from_latest(self) -> list[ObjectHypothesis]:
        if not (self.color_msg and self.depth_msg and self.camera_info):
            return []
        color = image_to_numpy(self.color_msg)
        depth = image_to_numpy(self.depth_msg)
        if color.ndim == 2:
            color = cv.cvtColor(color, cv.COLOR_GRAY2RGB)
        if self.color_msg.encoding.lower() in {'bgr8', 'bgra8'}:
            rgb = cv.cvtColor(color, cv.COLOR_BGR2RGB)
        elif color.shape[-1] == 4:
            rgb = color[:, :, :3]
        else:
            rgb = color[:, :, :3]
        hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)

        detections: list[ObjectHypothesis] = []
        for color_name, mask in color_masks(hsv).items():
            detections.extend(self._objects_for_mask(color_name, mask, depth))
        if self.config.detect_neutral_containers:
            detections.extend(self._containers_from_rgb(rgb, depth))
        return sorted(detections, key=lambda obj: obj.confidence, reverse=True)

    def _objects_for_mask(
        self,
        color_name: str,
        mask: np.ndarray,
        depth: np.ndarray,
    ) -> list[ObjectHypothesis]:
        num, labels, stats, centroids = cv.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        objects: list[ObjectHypothesis] = []
        for idx in range(1, num):
            area = int(stats[idx, cv.CC_STAT_AREA])
            if area < self.config.min_blob_area_px:
                continue
            if area > self.config.max_blob_area_px:
                continue
            x = int(stats[idx, cv.CC_STAT_LEFT])
            y = int(stats[idx, cv.CC_STAT_TOP])
            w = int(stats[idx, cv.CC_STAT_WIDTH])
            h = int(stats[idx, cv.CC_STAT_HEIGHT])
            u, v = centroids[idx]
            raw_xyz = self._pixel_to_base_xyz(float(u), float(v), depth, labels == idx)
            if raw_xyz is None:
                continue
            if not self._in_workspace(raw_xyz):
                continue
            table_xyz = self._project_color_pixel_to_table(float(u), float(v))
            xyz = table_xyz if (self.config.use_table_projection_xy and table_xyz is not None) else raw_xyz
            if not self._in_workspace(xyz, check_z=False):
                continue
            bbox = (x, y, x + w, y + h)
            footprint = self._footprint_from_bbox(w, h, raw_xyz[2])
            objects.append(
                ObjectHypothesis(
                    object_id=f'obj_{color_name}_cube',
                    label=f'{color_name} cube',
                    color=color_name,
                    shape='cube',
                    xyz=(
                        float(xyz[0]),
                        float(xyz[1]),
                        float(self.config.table_z_m + self.config.cube_height_m / 2.0),
                    ),
                    footprint_xy=footprint,
                    height_m=float(self.config.cube_height_m),
                    bbox_xyxy=bbox,
                    confidence=min(1.0, max(0.0, area / 2500.0)),
                    visible=True,
                    pickable=True,
                    container_like=False,
                    metadata={
                        'area_px': area,
                        'raw_depth_base_xyz': raw_xyz,
                        'table_projection_base_xyz': table_xyz,
                        'center_uv': (int(round(u)), int(round(v))),
                    },
                )
            )
        return objects

    def _containers_from_rgb(self, rgb: np.ndarray, depth: np.ndarray) -> list[ObjectHypothesis]:
        containers: list[ObjectHypothesis] = []
        for blob in detect_dark_neutral_container_blobs(
            rgb,
            min_area_px=self.config.min_container_area_px,
            max_area_px=self.config.max_container_area_px,
        ):
            x0, y0, x1, y1 = blob.bbox_xyxy
            u, v = blob.center_uv
            blob_mask = np.zeros(depth.shape, dtype=bool)
            blob_mask[y0:y1, x0:x1] = True
            raw_xyz = self._pixel_to_base_xyz(float(u), float(v), depth, blob_mask)
            if raw_xyz is None or not self._in_workspace(raw_xyz):
                continue
            table_xyz = self._project_color_pixel_to_table(float(u), float(v))
            xyz = table_xyz if table_xyz is not None else raw_xyz
            if not self._in_workspace(xyz, check_z=False):
                continue
            footprint = self._container_footprint_from_bbox(x1 - x0, y1 - y0, raw_xyz[2])
            containers.append(
                ObjectHypothesis(
                    object_id='obj_cup',
                    label='cup',
                    color=None,
                    shape='cup',
                    xyz=(
                        float(xyz[0]),
                        float(xyz[1]),
                        float(self.config.table_z_m + 0.04),
                    ),
                    footprint_xy=footprint,
                    height_m=0.08,
                    bbox_xyxy=blob.bbox_xyxy,
                    confidence=min(1.0, max(0.0, blob.area_px / 7000.0)),
                    visible=True,
                    pickable=False,
                    container_like=True,
                    metadata={
                        'area_px': blob.area_px,
                        'raw_depth_base_xyz': raw_xyz,
                        'table_projection_base_xyz': table_xyz,
                        'center_uv': (int(round(u)), int(round(v))),
                        'detector': 'dark_neutral_opening',
                    },
                )
            )
        return containers

    def _in_workspace(self, xyz: tuple[float, float, float], *, check_z: bool = True) -> bool:
        in_xy = bool(
            self.config.workspace_x_min_m <= float(xyz[0]) <= self.config.workspace_x_max_m
            and self.config.workspace_y_min_m <= float(xyz[1]) <= self.config.workspace_y_max_m
        )
        if not check_z:
            return in_xy
        return bool(in_xy and self.config.raw_z_min_m <= float(xyz[2]) <= self.config.raw_z_max_m)

    def _pixel_to_base_xyz(
        self,
        u: float,
        v: float,
        depth: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[float, float, float] | None:
        z = median_depth_at(depth, int(round(u)), int(round(v)), mask)
        if z is None:
            return None
        info = self.camera_info
        if info is None:
            return None
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None
        point = (
            (float(u) - cx) * z / fx,
            (float(v) - cy) * z / fy,
            z,
        )
        return self._transform_point(point, self.depth_msg.header.frame_id)

    def _project_color_pixel_to_table(self, u: float, v: float) -> tuple[float, float, float] | None:
        info = self.camera_info
        if info is None or self.color_msg is None:
            return None
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None

        color_frame = str(self.color_msg.header.frame_id or info.header.frame_id or '').strip()
        if not color_frame:
            return None
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.config.base_frame,
                color_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
        except TransformException as exc:
            self.get_logger().warn(f'color TF lookup failed: {exc}')
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        ray = (
            (float(u) - cx) / fx,
            (float(v) - cy) / fy,
            1.0,
        )
        rx, ry, rz = rotate_vector((float(q.x), float(q.y), float(q.z), float(q.w)), ray)
        if abs(rz) < 1e-6:
            return None
        scale = (float(self.config.table_z_m) - float(t.z)) / float(rz)
        if scale <= 0.0:
            return None
        return (
            float(t.x + scale * rx),
            float(t.y + scale * ry),
            float(self.config.table_z_m),
        )

    def _transform_point(
        self,
        point: tuple[float, float, float],
        source_frame: str,
    ) -> tuple[float, float, float] | None:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.config.base_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
        except TransformException as exc:
            self.get_logger().warn(f'TF lookup failed: {exc}')
            return None
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        rx, ry, rz = rotate_vector(
            (float(q.x), float(q.y), float(q.z), float(q.w)),
            point,
        )
        return (float(t.x + rx), float(t.y + ry), float(t.z + rz))

    def _footprint_from_bbox(
        self,
        width_px: int,
        height_px: int,
        z_base: float,
    ) -> tuple[float, float]:
        info = self.camera_info
        if info is None:
            return (0.04, 0.04)
        fx = max(1.0, float(info.k[0]))
        fy = max(1.0, float(info.k[4]))
        # Use raw camera depth if available in metadata. Clamp to cube scale.
        z = max(0.05, abs(float(z_base)))
        sx = min(0.08, max(0.025, float(width_px) * z / fx))
        sy = min(0.08, max(0.025, float(height_px) * z / fy))
        return (sx, sy)

    def _container_footprint_from_bbox(
        self,
        width_px: int,
        height_px: int,
        z_base: float,
    ) -> tuple[float, float]:
        info = self.camera_info
        if info is None:
            return (0.12, 0.12)
        fx = max(1.0, float(info.k[0]))
        fy = max(1.0, float(info.k[4]))
        z = max(0.05, abs(float(z_base)))
        sx = min(0.18, max(0.08, float(width_px) * z / fx))
        sy = min(0.18, max(0.08, float(height_px) * z / fy))
        return (sx, sy)

    def _color_cb(self, msg: Image) -> None:
        self.color_msg = msg

    def _depth_cb(self, msg: Image) -> None:
        self.depth_msg = msg

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self.camera_info = msg


def image_to_numpy(msg: Image) -> np.ndarray:
    dtype = np.float32 if msg.encoding == '32FC1' else np.uint8
    channels = 1
    enc = msg.encoding.lower()
    if enc in {'rgb8', 'bgr8'}:
        channels = 3
    elif enc in {'rgba8', 'bgra8'}:
        channels = 4
    array = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        return array.reshape((msg.height, msg.width))
    return array.reshape((msg.height, msg.width, channels))


def color_masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
    red_a = cv.inRange(hsv, (0, 80, 40), (10, 255, 255))
    red_b = cv.inRange(hsv, (170, 80, 40), (179, 255, 255))
    masks = {
        'red': cv.bitwise_or(red_a, red_b),
        'green': cv.inRange(hsv, (35, 70, 35), (90, 255, 255)),
        'blue': cv.inRange(hsv, (90, 60, 35), (135, 255, 255)),
        'yellow': cv.inRange(hsv, (18, 70, 45), (35, 255, 255)),
    }
    kernel = np.ones((5, 5), np.uint8)
    return {
        name: cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        for name, mask in masks.items()
    }


def median_depth_at(
    depth: np.ndarray,
    u: int,
    v: int,
    mask: np.ndarray,
    radius_px: int = 5,
) -> float | None:
    y0 = max(0, int(v) - radius_px)
    y1 = min(depth.shape[0], int(v) + radius_px + 1)
    x0 = max(0, int(u) - radius_px)
    x1 = min(depth.shape[1], int(u) + radius_px + 1)
    patch = depth[y0:y1, x0:x1]
    patch_mask = mask[y0:y1, x0:x1]
    values = patch[np.isfinite(patch) & (patch > 0.05) & patch_mask]
    if values.size == 0:
        values = patch[np.isfinite(patch) & (patch > 0.05)]
    if values.size == 0:
        return None
    return float(np.median(values))


def rotate_vector(
    quat_xyzw: tuple[float, float, float, float],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z, w = quat_xyzw
    vx, vy, vz = vec
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm > 1e-9:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )
