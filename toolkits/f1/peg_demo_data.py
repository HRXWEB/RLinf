# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parse and align F1 right-arm peg-insertion demonstrations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HEAD_IMAGE_TOPIC = "/camera/head/color/image_raw/compressed"
RIGHT_TCP_TOPIC = "/state/right_arm/tcp_pos"
RIGHT_GRIPPER_STATE_TOPIC = "/motion_ctl/gripper/right/state"
RIGHT_GRIPPER_COMMAND_TOPIC = "/motion_ctl/gripper/right"
TCP_NAMES = ("x", "y", "z", "rx", "ry", "rz")


class ConversionError(ValueError):
    """Raised when an episode cannot be converted safely."""


@dataclass(frozen=True)
class EpisodeStreams:
    """Timestamped streams required to convert one recorded episode."""

    image_timestamps_ns: np.ndarray
    images: tuple[np.ndarray | bytes, ...]
    tcp_timestamps_ns: np.ndarray
    tcp_poses_m_deg: np.ndarray
    gripper_timestamps_ns: np.ndarray
    gripper_values: np.ndarray
    gripper_command_timestamps_ns: np.ndarray
    gripper_command_values: np.ndarray


@dataclass(frozen=True)
class AlignedEpisode:
    """One episode represented as aligned 10 Hz transitions."""

    timestamps_ns: np.ndarray
    curr_images: np.ndarray
    next_images: np.ndarray
    curr_poses_m_deg: np.ndarray
    next_poses_m_deg: np.ndarray
    physical_actions: np.ndarray


def _as_sorted_arrays(
    rows: list[tuple[int, Any]],
    *,
    name: str,
) -> tuple[np.ndarray, tuple[Any, ...]]:
    if not rows:
        raise ConversionError(f"missing required {name} stream")
    rows.sort(key=lambda item: item[0])
    timestamps = np.asarray([item[0] for item in rows], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ConversionError(f"{name} timestamps must be strictly increasing")
    return timestamps, tuple(item[1] for item in rows)


def read_episode_mcap(path: str | Path) -> EpisodeStreams:
    """Read the four streams required from one ROS2 MCAP recording."""

    from mcap_ros2.reader import read_ros2_messages

    images: list[tuple[int, bytes]] = []
    tcp: list[tuple[int, np.ndarray]] = []
    gripper: list[tuple[int, float]] = []
    commands: list[tuple[int, float]] = []
    for record in read_ros2_messages(str(path)):
        topic = record.channel.topic
        timestamp_ns = int(record.log_time_ns)
        message = record.ros_msg
        if topic == HEAD_IMAGE_TOPIC:
            images.append((timestamp_ns, bytes(message.data)))
        elif topic == RIGHT_TCP_TOPIC:
            if tuple(message.name) != TCP_NAMES or len(message.position) != 6:
                raise ConversionError(
                    f"{RIGHT_TCP_TOPIC} must use names {TCP_NAMES} and six positions"
                )
            pose = np.asarray(message.position, dtype=np.float64)
            if not np.all(np.isfinite(pose)):
                raise ConversionError("right TCP pose contains non-finite values")
            pose[:3] *= 0.001
            tcp.append((timestamp_ns, pose))
        elif topic == RIGHT_GRIPPER_STATE_TOPIC:
            if len(message.position) != 1:
                raise ConversionError("right gripper state must contain one position")
            gripper.append((timestamp_ns, float(message.position[0])))
        elif topic == RIGHT_GRIPPER_COMMAND_TOPIC:
            commands.append((timestamp_ns, float(message.position)))

    image_times, image_values = _as_sorted_arrays(images, name="head image")
    tcp_times, tcp_values = _as_sorted_arrays(tcp, name="right TCP")
    gripper_times, gripper_values = _as_sorted_arrays(
        gripper, name="right gripper state"
    )
    if commands:
        command_times, command_values = _as_sorted_arrays(
            commands, name="right gripper command"
        )
    else:
        command_times = np.empty(0, dtype=np.int64)
        command_values = ()
    return EpisodeStreams(
        image_timestamps_ns=image_times,
        images=image_values,
        tcp_timestamps_ns=tcp_times,
        tcp_poses_m_deg=np.stack(tcp_values),
        gripper_timestamps_ns=gripper_times,
        gripper_values=np.asarray(gripper_values, dtype=np.float64),
        gripper_command_timestamps_ns=command_times,
        gripper_command_values=np.asarray(command_values, dtype=np.float64),
    )


def find_sustained_release(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    *,
    closed_threshold: float = 80.0,
    open_threshold: float = 20.0,
    min_hold_s: float = 0.1,
) -> int:
    """Return the first measured gripper-release timestamp."""

    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    positions = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or positions.shape != timestamps.shape:
        raise ConversionError("gripper timestamps and values must be one-dimensional")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ConversionError("gripper timestamps must be strictly increasing")
    hold_ns = int(round(float(min_hold_s) * 1_000_000_000))
    closed_seen = False
    for index, (timestamp, value) in enumerate(zip(timestamps, positions, strict=True)):
        closed_seen = closed_seen or value >= closed_threshold
        if not closed_seen or value > open_threshold:
            continue
        hold_end = int(timestamp) + hold_ns
        end_index = int(np.searchsorted(timestamps, hold_end, side="left"))
        if (
            end_index < len(timestamps)
            and np.max(positions[index : end_index + 1]) <= open_threshold
        ):
            return int(timestamp)
    raise ConversionError("episode has no sustained right-gripper release")


def _decode_rgb(image: np.ndarray | bytes) -> np.ndarray:
    if isinstance(image, np.ndarray):
        decoded = np.asarray(image, dtype=np.uint8)
    else:
        encoded = np.frombuffer(image, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ConversionError("failed to decode a head image")
        decoded = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ConversionError("decoded head image must have three channels")
    return np.array(decoded, dtype=np.uint8, copy=True)


def _nearest_indices(
    source_timestamps_ns: np.ndarray,
    target_timestamps_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(source_timestamps_ns, target_timestamps_ns, side="left")
    right = np.clip(right, 0, len(source_timestamps_ns) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(source_timestamps_ns[left] - target_timestamps_ns) <= np.abs(
        source_timestamps_ns[right] - target_timestamps_ns
    )
    indices = np.where(choose_left, left, right)
    deltas = np.abs(source_timestamps_ns[indices] - target_timestamps_ns)
    return indices, deltas


def interpolate_poses(
    timestamps_ns: np.ndarray,
    poses_m_deg: np.ndarray,
    target_timestamps_ns: np.ndarray,
) -> np.ndarray:
    source_s = (timestamps_ns - timestamps_ns[0]).astype(np.float64) / 1e9
    target_s = (target_timestamps_ns - timestamps_ns[0]).astype(np.float64) / 1e9
    values = np.asarray(poses_m_deg, dtype=np.float64).copy()
    if values.shape != (len(timestamps_ns), 6):
        raise ConversionError("right TCP stream must have shape (N, 6)")
    values[:, 3:] = np.rad2deg(np.unwrap(np.deg2rad(values[:, 3:]), axis=0))
    return np.column_stack(
        [np.interp(target_s, source_s, values[:, axis]) for axis in range(6)]
    )


def align_episode(
    streams: EpisodeStreams,
    *,
    release_ns: int,
    period_s: float = 0.1,
    max_image_delta_s: float = 0.05,
    include_images: bool = True,
) -> AlignedEpisode:
    """Align image and measured TCP streams into fixed-rate transitions."""

    period_ns = int(round(float(period_s) * 1e9))
    if period_ns <= 0:
        raise ConversionError("period_s must be positive")
    first_ns = max(
        int(streams.image_timestamps_ns[0]), int(streams.tcp_timestamps_ns[0])
    )
    last_ns = min(
        int(release_ns),
        int(streams.image_timestamps_ns[-1]) + 1,
        int(streams.tcp_timestamps_ns[-1]) + 1,
    )
    grid = np.arange(first_ns, last_ns, period_ns, dtype=np.int64)
    if len(grid) < 2:
        raise ConversionError("episode has fewer than two aligned observations")
    image_indices, image_deltas = _nearest_indices(streams.image_timestamps_ns, grid)
    tolerance_ns = int(round(float(max_image_delta_s) * 1e9))
    if np.any(image_deltas > tolerance_ns):
        worst_ms = float(np.max(image_deltas)) / 1e6
        raise ConversionError(
            f"image gap exceeds tolerance; worst aligned delta is {worst_ms:.3f} ms"
        )
    if include_images:
        images = np.stack(
            [_decode_rgb(streams.images[index]) for index in image_indices]
        )
        curr_images = images[:-1]
        next_images = images[1:]
    else:
        curr_images = np.empty((0, 0, 0, 3), dtype=np.uint8)
        next_images = np.empty((0, 0, 0, 3), dtype=np.uint8)
    poses = interpolate_poses(
        streams.tcp_timestamps_ns,
        streams.tcp_poses_m_deg,
        grid,
    )
    actions = np.diff(poses, axis=0)
    actions[:, 3:] = (actions[:, 3:] + 180.0) % 360.0 - 180.0
    return AlignedEpisode(
        timestamps_ns=grid,
        curr_images=curr_images,
        next_images=next_images,
        curr_poses_m_deg=poses[:-1],
        next_poses_m_deg=poses[1:],
        physical_actions=actions,
    )


def action_scale_statistics(
    episode_actions: list[np.ndarray],
    *,
    percentile: float = 99.0,
) -> dict[str, Any]:
    """Calculate grouped robust action scales and diagnostic percentiles."""

    if not episode_actions:
        raise ConversionError("at least one episode is required for action scales")
    actions = np.concatenate(
        [np.asarray(item, dtype=np.float64) for item in episode_actions], axis=0
    )
    if actions.ndim != 2 or actions.shape[1] != 6 or not np.all(np.isfinite(actions)):
        raise ConversionError("actions must be finite arrays with shape (N, 6)")
    absolute = np.abs(actions)
    position = absolute[:, :3].reshape(-1)
    orientation = absolute[:, 3:].reshape(-1)
    position_scale = float(np.percentile(position, percentile))
    orientation_scale = float(np.percentile(orientation, percentile))
    if position_scale <= 0.0 or orientation_scale <= 0.0:
        raise ConversionError("action scales must be positive")
    axis_names = ("x_m", "y_m", "z_m", "rx_deg", "ry_deg", "rz_deg")
    axis_stats: dict[str, dict[str, float]] = {}
    for axis, name in enumerate(axis_names):
        values = actions[:, axis]
        abs_values = absolute[:, axis]
        axis_stats[name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            **{
                f"abs_p{quantile:g}": float(np.percentile(abs_values, quantile))
                for quantile in (50.0, 90.0, 95.0, 99.0, 99.5)
            },
            "abs_max": float(np.max(abs_values)),
        }
    return {
        "percentile": float(percentile),
        "num_transitions": int(len(actions)),
        "position_scale_m": position_scale,
        "orientation_scale_deg": orientation_scale,
        "position_saturation_count": int(np.sum(position > position_scale)),
        "orientation_saturation_count": int(np.sum(orientation > orientation_scale)),
        "position_component_count": int(position.size),
        "orientation_component_count": int(orientation.size),
        "axis": axis_stats,
    }
