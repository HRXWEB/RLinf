#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Publish the fixed F1 insertion workspace for Foxglove visualization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

DEFAULT_TARGET_POSITION = (0.635837088, -0.110543916, 0.688607462)
DEFAULT_LOWER_OFFSET = (-0.20, -0.05, -0.02)
DEFAULT_UPPER_OFFSET = (0.20, 0.30, 0.10)


@dataclass(frozen=True)
class WorkspaceGeometry:
    """Axis-aligned workspace geometry in a single reference frame."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    center: tuple[float, float, float]
    scale: tuple[float, float, float]


def compute_workspace_geometry(
    target_position: Sequence[float],
    lower_offset: Sequence[float],
    upper_offset: Sequence[float],
) -> WorkspaceGeometry:
    """Calculate absolute workspace bounds around a target position."""

    if not all(
        len(values) == 3 for values in (target_position, lower_offset, upper_offset)
    ):
        raise ValueError(
            "target_position and offsets must contain exactly three values."
        )
    if any(lower >= upper for lower, upper in zip(lower_offset, upper_offset)):
        raise ValueError(
            "lower_offset must be smaller than upper_offset on every axis."
        )

    minimum = tuple(
        target + lower for target, lower in zip(target_position, lower_offset)
    )
    maximum = tuple(
        target + upper for target, upper in zip(target_position, upper_offset)
    )
    center = tuple((lower + upper) / 2.0 for lower, upper in zip(minimum, maximum))
    scale = tuple(upper - lower for lower, upper in zip(minimum, maximum))
    return WorkspaceGeometry(
        minimum=minimum,
        maximum=maximum,
        center=center,
        scale=scale,
    )


def _vector3(values: Sequence[float]):
    from geometry_msgs.msg import Vector3

    return Vector3(x=values[0], y=values[1], z=values[2])


def _build_markers(node, frame_id: str, geometry: WorkspaceGeometry, target_position):
    from visualization_msgs.msg import Marker, MarkerArray

    stamp = node.get_clock().now().to_msg()

    workspace = Marker()
    workspace.header.frame_id = frame_id
    workspace.header.stamp = stamp
    workspace.ns = "f1_right_arm_insertion_workspace"
    workspace.id = 0
    workspace.type = Marker.CUBE
    workspace.action = Marker.ADD
    workspace.pose.position.x, workspace.pose.position.y, workspace.pose.position.z = (
        geometry.center
    )
    workspace.pose.orientation.w = 1.0
    workspace.scale = _vector3(geometry.scale)
    workspace.color.r = 0.1
    workspace.color.g = 0.9
    workspace.color.b = 0.2
    workspace.color.a = 0.22

    target = Marker()
    target.header.frame_id = frame_id
    target.header.stamp = stamp
    target.ns = "f1_right_arm_insertion_workspace"
    target.id = 1
    target.type = Marker.SPHERE
    target.action = Marker.ADD
    target.pose.position.x, target.pose.position.y, target.pose.position.z = (
        target_position
    )
    target.pose.orientation.w = 1.0
    target.scale = _vector3((0.025, 0.025, 0.025))
    target.color.r = 1.0
    target.color.g = 0.1
    target.color.b = 0.1
    target.color.a = 1.0

    return MarkerArray(markers=[workspace, target])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--topic", default="/f1/right_arm/insertion_workspace_markers")
    parser.add_argument(
        "--target", nargs=3, type=float, default=DEFAULT_TARGET_POSITION
    )
    parser.add_argument(
        "--lower-offset", nargs=3, type=float, default=DEFAULT_LOWER_OFFSET
    )
    parser.add_argument(
        "--upper-offset", nargs=3, type=float, default=DEFAULT_UPPER_OFFSET
    )
    return parser.parse_args()


def main() -> None:
    """Run the ROS 2 workspace marker publisher."""

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from visualization_msgs.msg import MarkerArray

    args = _parse_args()
    geometry = compute_workspace_geometry(
        args.target,
        args.lower_offset,
        args.upper_offset,
    )
    rclpy.init()
    node = Node("f1_insertion_workspace_publisher")
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    publisher = node.create_publisher(MarkerArray, args.topic, qos)

    def publish() -> None:
        publisher.publish(_build_markers(node, args.frame_id, geometry, args.target))

    publish()
    node.create_timer(1.0, publish)
    node.get_logger().info(
        f"Publishing insertion workspace markers on {args.topic} in {args.frame_id}: "
        f"min={geometry.minimum}, max={geometry.maximum}"
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
