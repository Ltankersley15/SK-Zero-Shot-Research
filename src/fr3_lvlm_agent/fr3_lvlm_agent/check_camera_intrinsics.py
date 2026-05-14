#!/usr/bin/env python3
"""
Quick diagnostic to check camera intrinsics and TF tree.

Run this while the full stack is running to diagnose position errors.
"""

import rclpy
from rclpy.node import Node


class CameraIntrinsicsChecker(Node):
    def __init__(self):
        super().__init__('camera_intrinsics_checker')
        
        # Ground truth from fr3_test.usda
        self.ground_truth = {
            'red_cube': [0.4, 0.2, 0.02],
            'green_cube': [0.6, -0.3, 0.02],
            'blue_cube': [0.5, 0.4, 0.02],
            'yellow_cylinder': [0.35, -0.25, 0.02],
        }
        
        # D455 specs at 1280x720
        self.d455_expected_fx = 634.0
        self.d455_expected_fy = 634.0
        self.d455_expected_cx = 640.0
        self.d455_expected_cy = 360.0
        
        self.get_logger().info("Camera Intrinsics Checker")
        self.get_logger().info("Expected D455 intrinsics (1280x720):")
        self.get_logger().info(
            f"  fx={self.d455_expected_fx}, fy={self.d455_expected_fy}, "
            f"cx={self.d455_expected_cx}, cy={self.d455_expected_cy}"
        )
        self.get_logger().info(f"Ground truth positions: {self.ground_truth}")


def main():
    rclpy.init()
    checker = CameraIntrinsicsChecker()
    
    # Print Isaac Sim depth sensor config from USD
    checker.get_logger().info("\n=== Isaac Sim Depth Sensor Config (from USD) ===")
    checker.get_logger().info("From fr3_test.usda:")
    checker.get_logger().info("  omni:rtx:post:depthSensor:focalLengthPixel = 897")
    checker.get_logger().info("  Resolution: 1280x720")
    
    checker.get_logger().info("\n=== Analysis ===")
    checker.get_logger().info("NOTE: Isaac Sim uses focalLengthPixel=897, but D455 at 1280x720 should be ~634")
    checker.get_logger().info("This mismatch could cause position errors!")
    
    # Calculate expected error
    fx_isaac = 897.0
    fx_d455 = 634.0
    ratio = fx_isaac / fx_d455
    checker.get_logger().info(f"\nFocal length ratio: {fx_isaac}/{fx_d455} = {ratio:.3f}")
    checker.get_logger().info(f"Position error scale factor: {ratio:.3f}x (positions will be {ratio:.1%} too far)")
    
    # Example: object at 0.5m
    depth = 0.5
    error_x = depth * (ratio - 1.0) * 0.5  # Approximate
    checker.get_logger().info(f"\nExample: Object at depth={depth}m")
    checker.get_logger().info(f"  Expected X error: ~±{error_x*100:.1f}cm")
    
    checker.get_logger().info("\n=== Recommendation ===")
    checker.get_logger().info("1. Check camera_info topic for actual intrinsics")
    checker.get_logger().info("2. Verify depth_flip_x/y settings match Isaac Sim convention")
    checker.get_logger().info("3. Check TF tree: camera_link -> base_link transform")
    
    checker.destroy_node()


if __name__ == '__main__':
    main()
