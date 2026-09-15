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

"""Right-arm fixed-target peg insertion for the F1 real-world robot."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from time import monotonic
from typing import Any

import gymnasium as gym
import numpy as np
from f1_robot_controller import DualArmResetCommand

from ..f1_robot_env import F1RobotConfig, F1RobotEnv
from .peg_reward import peg_pose_metrics, peg_transition_reward

_RIGHT_TCP_ACTION = slice(7, 13)


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return normalized


def _nonnegative_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return normalized


def _pose(name: str, value: object) -> tuple[float, ...]:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite six-dimensional pose") from error
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite six-dimensional pose")
    return tuple(float(component) for component in pose)


def _joint_pose(name: str, value: object) -> tuple[float, ...]:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain seven finite joint angles") from error
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must contain seven finite joint angles")
    return tuple(float(component) for component in pose)


@dataclass
class RightArmPegInsertionConfig(F1RobotConfig):
    """Configuration for a calibrated, fixed-location right-arm insertion."""

    target_tcp_pose_m_deg: tuple[float, ...] = field(default_factory=tuple)
    reset_left_joint_pose_deg: tuple[float, ...] = field(default_factory=tuple)
    reset_right_joint_pose_deg: tuple[float, ...] = field(default_factory=tuple)
    joint_reset_tolerance_deg: float = 2.0
    tcp_reference_frame: str = ""
    position_reward_scale_m: float = 0.02
    orientation_reward_scale_deg: float = 10.0
    position_weight: float = 0.7
    orientation_weight: float = 0.3
    step_penalty: float = 0.01
    position_tolerance_m: float = 0.003
    orientation_tolerance_deg: float = 3.0
    success_hold_steps: int = 3
    success_bonus: float = 5.0

    def __post_init__(self) -> None:
        """Validate target calibration before opening the robot controller."""

        self.target_tcp_pose_m_deg = _pose(
            "target_tcp_pose_m_deg", self.target_tcp_pose_m_deg
        )
        self.reset_left_joint_pose_deg = _joint_pose(
            "reset_left_joint_pose_deg", self.reset_left_joint_pose_deg
        )
        self.reset_right_joint_pose_deg = _joint_pose(
            "reset_right_joint_pose_deg", self.reset_right_joint_pose_deg
        )
        self.joint_reset_tolerance_deg = _positive_float(
            "joint_reset_tolerance_deg", self.joint_reset_tolerance_deg
        )
        if not isinstance(self.tcp_reference_frame, str) or not (
            self.tcp_reference_frame.strip()
        ):
            raise ValueError("tcp_reference_frame must be a non-empty string")
        self.tcp_reference_frame = self.tcp_reference_frame.strip()

        for name in (
            "position_reward_scale_m",
            "orientation_reward_scale_deg",
            "position_weight",
            "orientation_weight",
            "position_tolerance_m",
            "orientation_tolerance_deg",
            "success_bonus",
        ):
            setattr(self, name, _positive_float(name, getattr(self, name)))
        self.step_penalty = _nonnegative_float("step_penalty", self.step_penalty)
        if isinstance(self.success_hold_steps, bool) or not isinstance(
            self.success_hold_steps, Integral
        ):
            raise TypeError("success_hold_steps must be an integer")
        if self.success_hold_steps <= 0:
            raise ValueError("success_hold_steps must be positive")
        self.success_hold_steps = int(self.success_hold_steps)
        super().__post_init__()


class RightArmPegInsertionEnv(F1RobotEnv):
    """Insert an already-grasped object using only the F1 right arm."""

    def __init__(self, config: RightArmPegInsertionConfig) -> None:
        super().__init__(config)
        self.config = config
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    OrderedDict((("proprioception", self._float_vector_space(6)),))
                ),
                "frames": gym.spaces.Dict(
                    {
                        "head_color": self._rgb_frame_space(
                            (*config.policy_image_shape, 3)
                        )
                    }
                ),
            }
        )
        self._previous_pose = np.zeros(6, dtype=np.float64)
        self._success_steps = 0
        self._success_bonus_awarded = False
        self._last_task_metrics: dict[str, float | bool] = {}

    @property
    def task_description(self) -> str:
        """Return the policy-facing task instruction."""

        return "Use the right arm to insert the held object into the fixed target."

    def _policy_observation(self, observation: Any) -> dict[str, Any]:
        _, _, right_tcp, _ = self._tcp_state(observation)
        return {
            "state": {"proprioception": right_tcp.astype(np.float32)},
            "frames": {
                "head_color": self._resize_policy_image(observation.head_color_rgb)
            },
        }

    def _reset_task(self, *, options: dict[str, Any] | None) -> dict[str, Any]:
        del options
        controller = self._active_controller
        measured = self._read_observation()
        command_id = self._allocate_command_id()
        target_left_rad = np.deg2rad(self.config.reset_left_joint_pose_deg)
        target_right_rad = np.deg2rad(self.config.reset_right_joint_pose_deg)
        tolerance_rad = np.deg2rad(self.config.joint_reset_tolerance_deg)
        command = DualArmResetCommand(
            command_id=command_id,
            left_joint_target_rad=target_left_rad,
            left_gripper_target=measured.left_gripper_position,
            right_joint_target_rad=target_right_rad,
            right_gripper_target=measured.right_gripper_position,
            duration_s=self.config.reset_duration_s,
            tolerance_rad=tolerance_rad,
            timeout_s=self.config.reset_timeout_s,
            created_at_monotonic_s=monotonic(),
        )
        controller.submit_reset_command(command)
        deadline_s = monotonic() + self.config.reset_timeout_s
        self._wait_for_finished_dispatch(
            command_id=command_id,
            context="joint reset",
            deadline_s=deadline_s,
        )
        while True:
            measured = self._read_observation()
            left_error = np.max(
                np.abs(measured.left_joint_position_rad - target_left_rad)
            )
            right_error = np.max(
                np.abs(measured.right_joint_position_rad - target_right_rad)
            )
            if max(left_error, right_error) <= tolerance_rad:
                break
            remaining_s = deadline_s - monotonic()
            if remaining_s <= 0.0:
                raise TimeoutError(
                    "dual-arm joint reset did not converge within "
                    f"{self.config.reset_timeout_s} seconds"
                )
            self._period_wait.wait(min(self.config.control_period_s, remaining_s))
        self._success_steps = 0
        self._success_bonus_awarded = False
        return {
            "reset_mode": "dual_arm_joint_pose",
            "reset_command_id": command_id,
            "tcp_reference_frame": self.config.tcp_reference_frame,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        metrics = self._task_metrics(observation)
        self._last_task_metrics = metrics
        self._previous_pose = np.asarray(
            observation["state"]["proprioception"], dtype=np.float64
        ).copy()
        return observation, info

    def _task_metrics(self, observation: dict[str, Any]) -> dict[str, float | bool]:
        pose = np.asarray(observation["state"]["proprioception"], dtype=np.float64)
        return peg_pose_metrics(pose, self.config)

    def _calc_step_reward(self, observation: dict[str, Any]) -> float:
        metrics = self._task_metrics(observation)
        pose = np.asarray(observation["state"]["proprioception"], dtype=np.float64)
        reaches_success_hold = (
            bool(metrics["within_success_region"])
            and self._success_steps + 1 >= self.config.success_hold_steps
        )
        terminal_success = reaches_success_hold and not self._success_bonus_awarded
        reward = peg_transition_reward(
            self._previous_pose,
            pose,
            self.config,
            terminal_success=terminal_success,
        )
        self._previous_pose = pose.copy()
        if terminal_success:
            self._success_bonus_awarded = True
        self._last_task_metrics = metrics
        return reward

    def _is_success(self, observation: dict[str, Any]) -> bool:
        metrics = self._task_metrics(observation)
        if bool(metrics["within_success_region"]):
            self._success_steps += 1
        else:
            self._success_steps = 0
        self._last_task_metrics = metrics
        return self._success_steps >= self.config.success_hold_steps

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        info.update(self._last_task_metrics)
        info["is_success"] = terminated
        info["tcp_reference_frame"] = self.config.tcp_reference_frame
        return observation, reward, terminated, truncated, info


class RightArmActionWrapper(gym.ActionWrapper):
    """Expose a 6D right-TCP action while holding all inactive actuators."""

    def __init__(self, env: RightArmPegInsertionEnv) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        try:
            right_action = np.asarray(action, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("action must be a numeric 6D right-TCP array") from error
        if right_action.shape != (6,) or not np.all(np.isfinite(right_action)):
            raise ValueError("action must be a finite shape-(6,) right-TCP array")
        if not self.action_space.contains(right_action):
            raise ValueError("action must stay within normalized [-1, 1] bounds")
        dual_arm_action = np.zeros(14, dtype=np.float32)
        dual_arm_action[_RIGHT_TCP_ACTION] = right_action
        return dual_arm_action


def create_right_arm_peg_insertion_env(
    override_cfg: Mapping[str, Any] | None = None,
    worker_info: Any = None,
    hardware_info: Any = None,
    env_idx: int = 0,
    env_cfg: Any = None,
) -> RightArmActionWrapper:
    """Create the single-arm task from RealWorldEnv-compatible arguments."""

    del worker_info, hardware_info, env_idx, env_cfg
    if override_cfg is None:
        config_values: dict[str, Any] = {}
    elif isinstance(override_cfg, Mapping):
        config_values = dict(override_cfg)
    else:
        raise TypeError("override_cfg must be a mapping")
    environment = RightArmPegInsertionEnv(RightArmPegInsertionConfig(**config_values))
    return RightArmActionWrapper(environment)
