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

"""Tests for F1 MCAP demonstration alignment and scale calculation."""

# ruff: noqa: E402

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from toolkits.f1.peg_demo_data import (
    ConversionError,
    EpisodeStreams,
    action_scale_statistics,
    align_episode,
    find_sustained_release,
)


def test_sustained_release_ignores_a_transient_open_sample() -> None:
    timestamps = np.arange(8, dtype=np.int64) * 20_000_000
    values = np.array([100, 100, 0, 100, 100, 0, 0, 0], dtype=np.float64)

    release_ns = find_sustained_release(
        timestamps,
        values,
        min_hold_s=0.04,
    )

    assert release_ns == timestamps[5]


def test_sustained_release_rejects_a_recording_without_release() -> None:
    timestamps = np.arange(10, dtype=np.int64) * 20_000_000

    with pytest.raises(ConversionError, match="sustained right-gripper release"):
        find_sustained_release(timestamps, np.full(10, 100.0))


def _streams(*, far_image: bool = False) -> EpisodeStreams:
    image_times = np.array([0, 100, 200, 300], dtype=np.int64) * 1_000_000
    if far_image:
        image_times[2] = 251_000_000
    images = tuple(np.full((4, 4, 3), i, dtype=np.uint8) for i in range(4))
    tcp_times = np.array([0, 50, 100, 150, 200, 250, 300], dtype=np.int64) * 1_000_000
    poses = np.zeros((7, 6), dtype=np.float64)
    poses[:, 0] = np.arange(7) * 0.001
    poses[:, 3] = [179.0, 179.5, -180.0, -179.5, -179.0, -178.5, -178.0]
    gripper_times = np.arange(18, dtype=np.int64) * 20_000_000
    gripper = np.array([100.0] * 15 + [0.0] * 3)
    return EpisodeStreams(
        image_timestamps_ns=image_times,
        images=images,
        tcp_timestamps_ns=tcp_times,
        tcp_poses_m_deg=poses,
        gripper_timestamps_ns=gripper_times,
        gripper_values=gripper,
        gripper_command_timestamps_ns=np.empty(0, dtype=np.int64),
        gripper_command_values=np.empty(0, dtype=np.float64),
    )


def test_align_episode_builds_ten_hz_transitions_and_wraps_euler_actions() -> None:
    aligned = align_episode(
        _streams(),
        release_ns=300_000_000,
        period_s=0.1,
        max_image_delta_s=0.05,
    )

    np.testing.assert_array_equal(aligned.timestamps_ns, [0, 100_000_000, 200_000_000])
    assert aligned.curr_images.shape == (2, 4, 4, 3)
    assert aligned.next_images.shape == (2, 4, 4, 3)
    np.testing.assert_allclose(aligned.curr_poses_m_deg[:, 0], [0.0, 0.002])
    np.testing.assert_allclose(aligned.next_poses_m_deg[:, 0], [0.002, 0.004])
    np.testing.assert_allclose(aligned.physical_actions[:, 0], [0.002, 0.002])
    np.testing.assert_allclose(aligned.physical_actions[:, 3], [1.0, 1.0])
    assert aligned.max_image_delta_ns == 0
    assert aligned.repeated_image_count == 0


def test_align_episode_rejects_an_image_outside_tolerance() -> None:
    with pytest.raises(ConversionError, match="image gap"):
        align_episode(
            _streams(far_image=True),
            release_ns=300_000_000,
            period_s=0.1,
            max_image_delta_s=0.05,
        )


def test_align_episode_default_tolerates_a_logged_camera_stall() -> None:
    aligned = align_episode(
        _streams(far_image=True),
        release_ns=300_000_000,
    )

    assert aligned.max_image_delta_ns == 51_000_000


def test_align_episode_can_skip_image_decoding_for_analysis() -> None:
    aligned = align_episode(
        _streams(),
        release_ns=300_000_000,
        include_images=False,
    )

    assert aligned.curr_images.shape == (0, 0, 0, 3)
    assert aligned.next_images.shape == (0, 0, 0, 3)
    assert aligned.physical_actions.shape == (2, 6)


def test_action_scale_statistics_pool_components_and_report_saturation() -> None:
    actions = [
        np.array(
            [
                [0.001, -0.002, 0.003, 0.1, -0.2, 0.3],
                [0.004, -0.005, 0.006, 0.4, -0.5, 0.6],
            ],
            dtype=np.float64,
        )
    ]

    stats = action_scale_statistics(actions, percentile=90.0)

    assert stats["position_scale_m"] == pytest.approx(
        np.percentile(np.arange(1, 7) * 0.001, 90)
    )
    assert stats["orientation_scale_deg"] == pytest.approx(
        np.percentile(np.arange(1, 7) * 0.1, 90)
    )
    assert stats["num_transitions"] == 2
    assert stats["position_saturation_count"] == 1
    assert stats["orientation_saturation_count"] == 1
