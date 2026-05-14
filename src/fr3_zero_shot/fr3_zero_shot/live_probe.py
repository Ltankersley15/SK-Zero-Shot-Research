from __future__ import annotations

import argparse
from dataclasses import asdict
import json

import rclpy

from .perception.live_scene_census import LiveSceneCensusConfig, LiveSceneCensusNode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Capture live scene census.')
    parser.add_argument('--timeout-sec', type=float, default=5.0)
    parser.add_argument('--table-z', type=float, default=0.02)
    args = parser.parse_args(argv)

    rclpy.init()
    node = LiveSceneCensusNode(
        LiveSceneCensusConfig(table_z_m=float(args.table_z))
    )
    try:
        objects = node.capture(timeout_sec=float(args.timeout_sec))
        print(json.dumps([asdict(obj) for obj in objects], indent=2))
        return 0 if objects else 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())

