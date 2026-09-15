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

"""Pure reward helpers shared by online and offline F1 peg insertion."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PegRewardConfig:
    """Parameters required by the peg-insertion shaping reward."""

    target_tcp_pose_m_deg: tuple[float, ...]
    position_reward_scale_m: float
    orientation_reward_scale_deg: float
    position_weight: float
    orientation_weight: float
    step_penalty: float
    position_tolerance_m: float
    orientation_tolerance_deg: float
    success_bonus: float


def wrapped_orientation_delta_deg(
    current: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Return component-wise shortest-path Euler deltas in degrees."""

    return (np.asarray(current) - np.asarray(target) + 180.0) % 360.0 - 180.0


def peg_pose_metrics(
    pose: np.ndarray,
    config: PegRewardConfig,
) -> dict[str, float | bool]:
    """Calculate shaping potential and success-region diagnostics."""

    pose_array = np.asarray(pose, dtype=np.float64)
    target = np.asarray(config.target_tcp_pose_m_deg, dtype=np.float64)
    if pose_array.shape != (6,) or target.shape != (6,):
        raise ValueError("pose and target_tcp_pose_m_deg must have shape (6,)")
    if not np.all(np.isfinite(pose_array)) or not np.all(np.isfinite(target)):
        raise ValueError("pose and target_tcp_pose_m_deg must be finite")

    position_error = float(np.linalg.norm(pose_array[:3] - target[:3]))
    orientation_error = float(
        np.linalg.norm(wrapped_orientation_delta_deg(pose_array[3:], target[3:]))
    )
    position_score = -(position_error / config.position_reward_scale_m)
    orientation_score = -(orientation_error / config.orientation_reward_scale_deg)
    potential = (
        config.position_weight * position_score
        + config.orientation_weight * orientation_score
    )
    return {
        "position_error_m": position_error,
        "orientation_error_deg": orientation_error,
        "position_score": position_score,
        "orientation_score": orientation_score,
        "task_potential": potential,
        "within_success_region": (
            position_error <= config.position_tolerance_m
            and orientation_error <= config.orientation_tolerance_deg
        ),
    }


def peg_transition_reward(
    previous_pose: np.ndarray,
    next_pose: np.ndarray,
    config: PegRewardConfig,
    *,
    terminal_success: bool = False,
) -> float:
    """Calculate one potential-difference reward for a TCP transition."""

    previous_potential = float(
        peg_pose_metrics(previous_pose, config)["task_potential"]
    )
    next_potential = float(peg_pose_metrics(next_pose, config)["task_potential"])
    reward = next_potential - previous_potential - config.step_penalty
    if terminal_success:
        reward += config.success_bonus
    return float(reward)
