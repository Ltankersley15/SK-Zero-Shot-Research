from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

from fr3_zero_shot.core.types import ObjectHypothesis


class PlanningSceneAdapter:
    def __init__(
        self,
        node: Node,
        *,
        base_frame: str = 'fr3_link0',
        apply_service: str = '/fr3/apply_planning_scene',
        timeout_sec: float = 1.0,
    ) -> None:
        self._node = node
        self._base_frame = str(base_frame)
        self._client = node.create_client(ApplyPlanningScene, apply_service)
        self._timeout_sec = max(0.1, float(timeout_sec))

    def apply_cup_keepout(
        self,
        cup: ObjectHypothesis,
        *,
        table_z: float,
        radius_pad_m: float = 0.030,
        height_m: float = 0.180,
    ) -> bool:
        radius = max(float(cup.footprint_xy[0]), float(cup.footprint_xy[1])) * 0.5 + float(radius_pad_m)
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.CYLINDER
        primitive.dimensions = [float(height_m), float(radius)]

        pose = Pose()
        pose.orientation.w = 1.0
        pose.position.x = float(cup.xyz[0])
        pose.position.y = float(cup.xyz[1])
        pose.position.z = float(table_z + height_m * 0.5)

        obj = CollisionObject()
        obj.id = 'cup_keepout'
        obj.header.frame_id = self._base_frame
        obj.operation = CollisionObject.ADD
        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)
        return self._apply(scene)

    def clear_cup_keepout(self) -> bool:
        obj = CollisionObject()
        obj.id = 'cup_keepout'
        obj.header.frame_id = self._base_frame
        obj.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)
        return self._apply(scene)

    def _apply(self, scene: PlanningScene) -> bool:
        if not self._client.wait_for_service(timeout_sec=self._timeout_sec):
            return False
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._client.call_async(request)
        deadline = time.time() + self._timeout_sec
        while rclpy.ok() and not future.done() and time.time() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.05)
        if not future.done():
            return False
        result = future.result()
        return bool(result is not None and result.success)
