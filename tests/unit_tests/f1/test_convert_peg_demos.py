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

"""Tests for converting aligned F1 peg demos to RLinf trajectories."""

# ruff: noqa: E402

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.envs.realworld.f1.tasks.peg_reward import PegRewardConfig
from toolkits.f1.convert_peg_demos import (
    _validate_published_buffer,
    build_trajectory,
)
from toolkits.f1.peg_demo_data import AlignedEpisode


def _reward_config() -> PegRewardConfig:
    return PegRewardConfig(
        target_tcp_pose_m_deg=(0.002, 0.0, 0.0, 0.0, 0.0, 0.0),
        position_reward_scale_m=0.02,
        orientation_reward_scale_deg=10.0,
        position_weight=0.7,
        orientation_weight=0.3,
        step_penalty=0.01,
        position_tolerance_m=0.003,
        orientation_tolerance_deg=3.0,
        success_bonus=5.0,
    )


def _aligned_episode() -> AlignedEpisode:
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.002, 0.0, 0.0, 0.2, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    images = np.zeros((3, 720, 1280, 3), dtype=np.uint8)
    images[:, :, 280:1000, 1] = 255
    return AlignedEpisode(
        timestamps_ns=np.array([0, 100_000_000, 200_000_000], dtype=np.int64),
        curr_images=images[:-1],
        next_images=images[1:],
        curr_poses_m_deg=poses[:-1],
        next_poses_m_deg=poses[1:],
        physical_actions=np.diff(poses, axis=0),
    )


def test_build_trajectory_normalizes_actions_images_and_terminal_success() -> None:
    trajectory = build_trajectory(
        _aligned_episode(),
        position_scale_m=0.002,
        orientation_scale_deg=0.2,
        reward_config=_reward_config(),
        policy_image_shape=(128, 128),
    )

    assert trajectory.actions.shape == (2, 1, 6)
    np.testing.assert_allclose(trajectory.actions[:, 0, 0], 0.5)
    np.testing.assert_allclose(trajectory.actions[:, 0, 3], 0.5)
    assert torch.all(trajectory.intervene_flags)
    assert trajectory.curr_obs["main_images"].shape == (2, 1, 128, 128, 3)
    assert trajectory.curr_obs["main_images"].dtype == torch.uint8
    assert torch.all(trajectory.curr_obs["main_images"][..., 1] == 255)
    assert trajectory.curr_obs["states"].shape == (2, 1, 6)
    assert trajectory.forward_inputs["action"].data_ptr() != 0
    assert trajectory.rewards.shape == (2, 1, 1)
    assert trajectory.dones.shape == (3, 1, 1)
    assert trajectory.terminations.shape == (3, 1, 1)
    assert trajectory.truncations.shape == (3, 1, 1)
    assert trajectory.dones[:, 0, 0].tolist() == [False, False, True]
    assert trajectory.terminations[:, 0, 0].tolist() == [False, False, True]
    assert trajectory.truncations[:, 0, 0].tolist() == [False, False, False]
    assert trajectory.rewards[-1, 0, 0] > 4.9


def test_build_trajectory_rejects_nonpositive_scales() -> None:
    with pytest.raises(ValueError, match="action scales must be positive"):
        build_trajectory(
            _aligned_episode(),
            position_scale_m=0.0,
            orientation_scale_deg=0.2,
            reward_config=_reward_config(),
        )


def test_trajectory_round_trips_through_replay_buffer(tmp_path: Path) -> None:
    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=False,
        auto_save=True,
        auto_save_path=str(tmp_path),
        trajectory_format="pt",
    )
    try:
        buffer.add_trajectories(
            [
                build_trajectory(
                    _aligned_episode(),
                    position_scale_m=0.002,
                    orientation_scale_deg=0.2,
                    reward_config=_reward_config(),
                )
            ]
        )
    finally:
        buffer.close()

    validation = _validate_published_buffer(tmp_path, expected_size=1)

    assert validation["num_trajectories"] == 1
    assert validation["total_samples"] == 2
    assert validation["sampled_transitions"] == 2
