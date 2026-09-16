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

"""Right-arm fixed-target XYZ reaching task for the F1 real-world robot."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from ..f1_robot_env import resize_rgb
from .right_arm_peg_insertion_env import (
    RightArmPegInsertionConfig,
    RightArmPegInsertionEnv,
)

_RIGHT_POSITION_ACTION = slice(7, 10)


def _vector3(name: str, value: object) -> tuple[float, float, float]:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain three finite values") from error
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return tuple(float(component) for component in vector)


def _rotation3(name: str, value: object) -> tuple[tuple[float, ...], ...]:
    try:
        rotation = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite 3x3 rotation") from error
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be a finite 3x3 rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError(f"{name} must be an orthonormal right-handed rotation")
    return tuple(tuple(float(component) for component in row) for row in rotation)


@dataclass
class RightArmFixedReachConfig(RightArmPegInsertionConfig):
    """Configuration for fixed-orientation right-arm XYZ reaching."""

    fixed_orientation_deg: tuple[float, ...] = field(default_factory=tuple)
    vendor_to_base_rotation: tuple[tuple[float, ...], ...] = field(
        default_factory=tuple
    )
    workspace_lower_offset_m: tuple[float, ...] = field(default_factory=tuple)
    workspace_upper_offset_m: tuple[float, ...] = field(default_factory=tuple)
    workspace_command_margin_m: float = 0.0
    xy_tolerance_m: float = 0.005
    z_tolerance_m: float = 0.003
    post_action_image_source: str = "right_wrist_color"

    def __post_init__(self) -> None:
        """Validate fixed pose and base-frame workspace calibration."""

        self.fixed_orientation_deg = _vector3(
            "fixed_orientation_deg", self.fixed_orientation_deg
        )
        self.vendor_to_base_rotation = _rotation3(
            "vendor_to_base_rotation", self.vendor_to_base_rotation
        )
        self.workspace_lower_offset_m = _vector3(
            "workspace_lower_offset_m", self.workspace_lower_offset_m
        )
        self.workspace_upper_offset_m = _vector3(
            "workspace_upper_offset_m", self.workspace_upper_offset_m
        )
        if any(
            lower >= upper
            for lower, upper in zip(
                self.workspace_lower_offset_m, self.workspace_upper_offset_m
            )
        ):
            raise ValueError(
                "workspace_lower_offset_m must be smaller than workspace_upper_offset_m"
            )
        self.workspace_command_margin_m = float(self.workspace_command_margin_m)
        if (
            not np.isfinite(self.workspace_command_margin_m)
            or self.workspace_command_margin_m < 0.0
        ):
            raise ValueError(
                "workspace_command_margin_m must be finite and nonnegative"
            )
        workspace_width = np.subtract(
            self.workspace_upper_offset_m,
            self.workspace_lower_offset_m,
        )
        if np.any(2.0 * self.workspace_command_margin_m >= workspace_width):
            raise ValueError(
                "workspace_command_margin_m must leave a nonempty command workspace"
            )
        super().__post_init__()
        self.xy_tolerance_m = float(self.xy_tolerance_m)
        self.z_tolerance_m = float(self.z_tolerance_m)
        if not np.isfinite(self.xy_tolerance_m) or self.xy_tolerance_m <= 0.0:
            raise ValueError("xy_tolerance_m must be positive and finite")
        if not np.isfinite(self.z_tolerance_m) or self.z_tolerance_m <= 0.0:
            raise ValueError("z_tolerance_m must be positive and finite")


class RightArmFixedReachEnv(RightArmPegInsertionEnv):
    """Reach one fixed TCP position with XYZ actions and a fixed orientation."""

    def __init__(self, config: RightArmFixedReachConfig) -> None:
        super().__init__(config)
        self.config = config
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    OrderedDict((("proprioception", self._float_vector_space(3)),))
                ),
                "frames": gym.spaces.Dict(
                    {
                        "right_wrist_color": self._rgb_frame_space(
                            (*config.policy_image_shape, 3)
                        )
                    }
                ),
            }
        )
        self._previous_position = np.zeros(3, dtype=np.float64)

    @property
    def task_description(self) -> str:
        """Return the policy-facing task instruction."""

        return "Move the right wrist to the fixed XYZ target."

    @property
    def _rotation(self) -> np.ndarray:
        return np.asarray(self.config.vendor_to_base_rotation, dtype=np.float64)

    def _policy_observation(self, observation: Any) -> dict[str, Any]:
        _, _, right_tcp, _ = self._tcp_state(observation)
        return {
            "state": {"proprioception": right_tcp[:3].astype(np.float32)},
            "frames": {
                "right_wrist_color": resize_rgb(
                    observation.right_wrist_color_rgb,
                    self.config.policy_image_shape,
                )
            },
        }

    def _bounded_tcp_target(
        self,
        *,
        arm_name: str,
        current: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        target = current + action.astype(np.float64)
        if arm_name != "right_arm":
            return target
        fixed_orientation = np.asarray(
            self.config.fixed_orientation_deg, dtype=np.float64
        )
        orientation_limit = float(
            self._arm_tcp_envelope("right_arm")["max_delta"]["orientation_deg"]
        )
        orientation_delta = np.abs(fixed_orientation - current[3:])
        if np.any(orientation_delta > orientation_limit):
            raise ValueError(
                "fixed orientation delta exceeds the configured right-arm "
                f"limit of {orientation_limit} degrees"
            )
        configured_target = np.asarray(
            self.config.target_tcp_pose_m_deg, dtype=np.float64
        )
        relative_base = self._rotation @ (target[:3] - configured_target[:3])
        command_margin = self.config.workspace_command_margin_m
        clipped_base = np.clip(
            relative_base,
            np.asarray(self.config.workspace_lower_offset_m, dtype=np.float64)
            + command_margin,
            np.asarray(self.config.workspace_upper_offset_m, dtype=np.float64)
            - command_margin,
        )
        target[:3] = configured_target[:3] + self._rotation.T @ clipped_base
        target[3:] = fixed_orientation
        return target

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        self._previous_position = np.asarray(
            observation["state"]["proprioception"], dtype=np.float64
        ).copy()
        return observation, info

    def _position_error_base(self, position: np.ndarray) -> np.ndarray:
        target = np.asarray(self.config.target_tcp_pose_m_deg[:3], dtype=np.float64)
        return self._rotation @ (position - target)

    def _task_metrics(self, observation: dict[str, Any]) -> dict[str, float | bool]:
        position = np.asarray(observation["state"]["proprioception"], dtype=np.float64)
        error = self._position_error_base(position)
        return {
            "position_error_m": float(np.linalg.norm(error)),
            "x_error_m": float(abs(error[0])),
            "y_error_m": float(abs(error[1])),
            "z_error_m": float(abs(error[2])),
            "within_success_region": bool(
                abs(error[0]) <= self.config.xy_tolerance_m
                and abs(error[1]) <= self.config.xy_tolerance_m
                and abs(error[2]) <= self.config.z_tolerance_m
            ),
        }

    def _calc_step_reward(self, observation: dict[str, Any]) -> float:
        position = np.asarray(observation["state"]["proprioception"], dtype=np.float64)
        previous_distance = np.linalg.norm(
            self._position_error_base(self._previous_position)
        )
        current_distance = np.linalg.norm(self._position_error_base(position))
        reaches_success_hold = (
            bool(self._task_metrics(observation)["within_success_region"])
            and self._success_steps + 1 >= self.config.success_hold_steps
        )
        terminal_success = reaches_success_hold and not self._success_bonus_awarded
        reward = (
            previous_distance - current_distance
        ) / self.config.position_reward_scale_m - self.config.step_penalty
        if terminal_success:
            reward += self.config.success_bonus
            self._success_bonus_awarded = True
        self._previous_position = position.copy()
        self._last_task_metrics = self._task_metrics(observation)
        return float(reward)


class RightArmPositionActionWrapper(gym.ActionWrapper):
    """Expose only normalized right-arm XYZ deltas."""

    def __init__(self, env: RightArmFixedReachEnv) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        try:
            right_position = np.asarray(action, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "action must be a numeric 3D right-position array"
            ) from error
        if right_position.shape != (3,) or not np.all(np.isfinite(right_position)):
            raise ValueError("action must be a finite shape-(3,) right-position array")
        if not self.action_space.contains(right_position):
            raise ValueError("action must stay within normalized [-1, 1] bounds")
        dual_arm_action = np.zeros(14, dtype=np.float32)
        dual_arm_action[_RIGHT_POSITION_ACTION] = right_position
        return dual_arm_action


def create_right_arm_fixed_reach_env(
    override_cfg: Mapping[str, Any] | None = None,
    worker_info: Any = None,
    hardware_info: Any = None,
    env_idx: int = 0,
    env_cfg: Any = None,
) -> RightArmPositionActionWrapper:
    """Create the curriculum base task from RealWorldEnv-compatible arguments."""

    del worker_info, hardware_info, env_idx, env_cfg
    if override_cfg is None:
        config_values: dict[str, Any] = {}
    elif isinstance(override_cfg, Mapping):
        config_values = dict(override_cfg)
    else:
        raise TypeError("override_cfg must be a mapping")
    environment = RightArmFixedReachEnv(RightArmFixedReachConfig(**config_values))
    return RightArmPositionActionWrapper(environment)
