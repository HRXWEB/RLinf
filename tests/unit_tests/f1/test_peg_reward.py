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

"""Tests for the shared F1 peg-insertion reward calculation."""

import numpy as np
import pytest

from rlinf.envs.realworld.f1.tasks.peg_reward import (
    PegRewardConfig,
    peg_pose_metrics,
    peg_transition_reward,
    wrapped_orientation_delta_deg,
)


def _config() -> PegRewardConfig:
    return PegRewardConfig(
        target_tcp_pose_m_deg=(0.1, 0.0, 0.0, 179.0, 0.0, 0.0),
        position_reward_scale_m=0.02,
        orientation_reward_scale_deg=10.0,
        position_weight=0.7,
        orientation_weight=0.3,
        step_penalty=0.01,
        position_tolerance_m=0.003,
        orientation_tolerance_deg=3.0,
        success_bonus=5.0,
    )


def test_wrapped_orientation_delta_uses_shortest_path() -> None:
    delta = wrapped_orientation_delta_deg(
        np.array([-179.0, 5.0, -5.0]),
        np.array([179.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(delta, [2.0, 5.0, -5.0])


def test_peg_pose_metrics_combines_position_and_orientation_scores() -> None:
    config = _config()
    pose = np.array([0.12, 0.0, 0.0, -179.0, 0.0, 0.0])

    metrics = peg_pose_metrics(pose, config)

    expected_position = -(0.02 / 0.02)
    expected_orientation = -(2.0 / 10.0)
    assert metrics["position_error_m"] == pytest.approx(0.02)
    assert metrics["orientation_error_deg"] == pytest.approx(2.0)
    assert metrics["task_potential"] == pytest.approx(
        0.7 * expected_position + 0.3 * expected_orientation
    )
    assert metrics["within_success_region"] is False


def test_peg_transition_reward_has_signal_far_from_target() -> None:
    config = _config()
    previous = np.array([0.30, 0.0, 0.0, 179.0, 0.0, 0.0])
    current = np.array([0.29, 0.0, 0.0, 179.0, 0.0, 0.0])

    reward = peg_transition_reward(previous, current, config)

    assert reward == pytest.approx(0.7 * 0.01 / 0.02 - config.step_penalty)


def test_peg_transition_reward_adds_terminal_bonus_only_when_requested() -> None:
    config = _config()
    previous = np.array([0.12, 0.0, 0.0, 179.0, 0.0, 0.0])
    current = np.array(config.target_tcp_pose_m_deg)

    dense = peg_transition_reward(previous, current, config)
    terminal = peg_transition_reward(
        previous,
        current,
        config,
        terminal_success=True,
    )

    expected_dense = (
        peg_pose_metrics(current, config)["task_potential"]
        - peg_pose_metrics(previous, config)["task_potential"]
        - config.step_penalty
    )
    assert dense == pytest.approx(expected_dense)
    assert terminal == pytest.approx(expected_dense + config.success_bonus)
