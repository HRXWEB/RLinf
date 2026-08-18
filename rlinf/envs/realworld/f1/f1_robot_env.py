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

"""Platform-level Gymnasium environment for the F1 dual-arm robot."""

from dataclasses import dataclass
from threading import Event
from time import monotonic
from typing import Any

import gymnasium as gym
import numpy as np
from f1_robot_controller import (
    CommandStatus,
    ControllerConfig,
    ControllerError,
    DualArmCommand,
    DualArmResetCommand,
    F1RobotController,
    ObservationUnavailableError,
    RobotObservation,
    create_controller,
)


@dataclass
class F1RobotConfig:
    """Configuration shared by F1 real-world tasks."""

    is_dummy: bool = True
    control_period_s: float = 0.01
    max_num_steps: int = 4
    max_observation_age_s: float = 0.25
    max_observation_skew_s: float = 0.05
    max_joint_delta_rad: tuple[float, ...] = (0.01,) * 14
    max_gripper_delta: float = 0.05
    reset_joint_target_rad: tuple[float, ...] = (0.0,) * 14
    reset_gripper_target: tuple[float, float] = (0.5, 0.5)
    reset_duration_s: float = 0.01
    reset_timeout_s: float = 1.0
    reset_tolerance_rad: float = 0.01


class F1RobotEnv(gym.Env):
    """Own the F1 Controller lifecycle and common action/observation schema."""

    metadata = {"render_modes": []}

    def __init__(self, config: F1RobotConfig) -> None:
        """Open one Controller and publish the phase-one Gym spaces.

        Args:
            config: F1 Controller, timing, action-scale, and reset configuration.
        """

        super().__init__()
        self.config = config
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "left_joint_position": self._float_vector_space(7),
                        "left_gripper": self._gripper_space(),
                        "right_joint_position": self._float_vector_space(7),
                        "right_gripper": self._gripper_space(),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        "head_color": self._rgb_frame_space(),
                        "left_wrist_color": self._rgb_frame_space(),
                        "right_wrist_color": self._rgb_frame_space(),
                    }
                ),
            }
        )
        self._closed = False
        self._controller: F1RobotController | None = None
        self._next_command_id = 0
        self._num_steps = 0
        self._period_wait = Event()
        controller = create_controller(
            is_dummy=config.is_dummy,
            config=ControllerConfig(
                control_period_s=config.control_period_s,
                max_observation_age_s=config.max_observation_age_s,
                max_observation_skew_s=config.max_observation_skew_s,
            ),
        )
        self._controller = controller
        try:
            controller.open()
            controller.wait_ready(timeout_s=config.reset_timeout_s)
        except BaseException:
            controller.close()
            self._closed = True
            raise

    @staticmethod
    def _float_vector_space(size: int) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(size,),
            dtype=np.float32,
        )

    @staticmethod
    def _gripper_space() -> gym.spaces.Box:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    @staticmethod
    def _rgb_frame_space() -> gym.spaces.Box:
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(128, 128, 3),
            dtype=np.uint8,
        )

    @property
    def _active_controller(self) -> F1RobotController:
        if self._closed or self._controller is None:
            raise RuntimeError("F1RobotEnv is closed")
        return self._controller

    def _allocate_command_id(self) -> int:
        command_id = self._next_command_id
        self._next_command_id += 1
        return command_id

    def _read_observation(
        self,
        *,
        newer_than: float | None = None,
    ) -> RobotObservation:
        return self._active_controller.read_observation(
            max_age_s=self.config.max_observation_age_s,
            max_skew_s=self.config.max_observation_skew_s,
            newer_than=newer_than,
        )

    @staticmethod
    def _policy_observation(observation: RobotObservation) -> dict[str, Any]:
        return {
            "state": {
                "left_joint_position": np.array(
                    observation.left_joint_position_rad,
                    dtype=np.float32,
                    copy=True,
                ),
                "left_gripper": np.array(
                    [observation.left_gripper_position],
                    dtype=np.float32,
                ),
                "right_joint_position": np.array(
                    observation.right_joint_position_rad,
                    dtype=np.float32,
                    copy=True,
                ),
                "right_gripper": np.array(
                    [observation.right_gripper_position],
                    dtype=np.float32,
                ),
            },
            "frames": {
                "head_color": np.array(
                    observation.head_color_rgb,
                    dtype=np.uint8,
                    copy=True,
                ),
                "left_wrist_color": np.array(
                    observation.left_wrist_color_rgb,
                    dtype=np.uint8,
                    copy=True,
                ),
                "right_wrist_color": np.array(
                    observation.right_wrist_color_rgb,
                    dtype=np.uint8,
                    copy=True,
                ),
            },
        }

    @staticmethod
    def _validated_action(action: np.ndarray) -> np.ndarray:
        try:
            policy_action = np.array(action, dtype=np.float32, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("action must be a numeric 16D array") from error
        if policy_action.shape != (16,):
            raise ValueError("action must have shape (16,)")
        if not np.all(np.isfinite(policy_action)):
            raise ValueError("action must contain only finite values")
        if np.any(policy_action < -1.0) or np.any(policy_action > 1.0):
            raise ValueError("action must be normalized to [-1, 1]")
        return policy_action

    def _absolute_command(
        self,
        policy_action: np.ndarray,
        observation: RobotObservation,
    ) -> DualArmCommand:
        joint_delta = np.asarray(self.config.max_joint_delta_rad, dtype=np.float64)
        left_joint_target = (
            observation.left_joint_position_rad
            + policy_action[:7].astype(np.float64) * joint_delta[:7]
        )
        right_joint_target = (
            observation.right_joint_position_rad
            + policy_action[8:15].astype(np.float64) * joint_delta[7:]
        )
        left_gripper_target = np.clip(
            observation.left_gripper_position
            + float(policy_action[7]) * self.config.max_gripper_delta,
            0.0,
            1.0,
        )
        right_gripper_target = np.clip(
            observation.right_gripper_position
            + float(policy_action[15]) * self.config.max_gripper_delta,
            0.0,
            1.0,
        )
        return DualArmCommand(
            command_id=self._allocate_command_id(),
            left_joint_target_rad=left_joint_target,
            left_gripper_target=float(left_gripper_target),
            right_joint_target_rad=right_joint_target,
            right_gripper_target=float(right_gripper_target),
            duration_s=self.config.control_period_s,
            created_at_monotonic_s=monotonic(),
        )

    def _calc_step_reward(self, observation: dict[str, Any]) -> float:
        del observation
        return 0.0

    def _is_success(self, observation: dict[str, Any]) -> bool:
        del observation
        return False

    def _reset_command(self) -> DualArmResetCommand:
        joint_target = np.asarray(
            self.config.reset_joint_target_rad,
            dtype=np.float64,
        )
        return DualArmResetCommand(
            command_id=self._allocate_command_id(),
            left_joint_target_rad=joint_target[:7],
            left_gripper_target=self.config.reset_gripper_target[0],
            right_joint_target_rad=joint_target[7:],
            right_gripper_target=self.config.reset_gripper_target[1],
            duration_s=self.config.reset_duration_s,
            tolerance_rad=self.config.reset_tolerance_rad,
            timeout_s=self.config.reset_timeout_s,
            created_at_monotonic_s=monotonic(),
        )

    @staticmethod
    def _reset_has_converged(
        observation: RobotObservation,
        command: DualArmResetCommand,
    ) -> bool:
        tolerance = command.tolerance_rad
        return bool(
            np.allclose(
                observation.left_joint_position_rad,
                command.left_joint_target_rad,
                rtol=0.0,
                atol=tolerance,
            )
            and np.allclose(
                observation.right_joint_position_rad,
                command.right_joint_target_rad,
                rtol=0.0,
                atol=tolerance,
            )
            and abs(observation.left_gripper_position - command.left_gripper_target)
            <= tolerance
            and abs(observation.right_gripper_position - command.right_gripper_target)
            <= tolerance
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset robot pose and return only after measured-state convergence."""

        super().reset(seed=seed)
        del options
        command = self._reset_command()
        deadline_s = monotonic() + self.config.reset_timeout_s
        receipt = self._active_controller.submit_reset_command(command)
        failed_statuses = {
            CommandStatus.REJECTED,
            CommandStatus.EXPIRED,
            CommandStatus.CANCELLED,
            CommandStatus.FAULT,
        }
        while True:
            status = self._active_controller.get_command_status(command.command_id)
            if status in failed_statuses:
                raise ControllerError(
                    f"reset command {command.command_id} failed with {status.value}"
                )
            try:
                measured = self._read_observation(
                    newer_than=receipt.accepted_at_monotonic_s
                )
            except ObservationUnavailableError:
                measured = None
            if measured is not None and self._reset_has_converged(
                measured,
                command,
            ):
                self._num_steps = 0
                return self._policy_observation(measured), {
                    "reset_command_id": command.command_id,
                    "command_status": status,
                }
            remaining_s = deadline_s - monotonic()
            if remaining_s <= 0.0:
                raise TimeoutError(
                    f"reset command {command.command_id} did not converge within "
                    f"{self.config.reset_timeout_s} seconds"
                )
            self._period_wait.wait(min(self.config.control_period_s, remaining_s))

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one normalized delta as one absolute Controller command."""

        policy_action = self._validated_action(action)
        measured_before = self._read_observation()
        command = self._absolute_command(policy_action, measured_before)
        receipt = self._active_controller.submit_command(command)
        period_deadline = monotonic() + self.config.control_period_s
        remaining_s = period_deadline - monotonic()
        if remaining_s > 0.0:
            self._period_wait.wait(remaining_s)
        measured_after = self._read_observation(
            newer_than=receipt.accepted_at_monotonic_s
        )
        observation = self._policy_observation(measured_after)
        self._num_steps += 1
        reward = self._calc_step_reward(observation)
        terminated = self._is_success(observation)
        truncated = self._num_steps >= self.config.max_num_steps
        info = {
            "policy_action": policy_action.copy(),
            "command_id": command.command_id,
            "command_status": self._active_controller.get_command_status(
                command.command_id
            ),
        }
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the owned Controller exactly once."""

        if self._closed:
            return
        if self._controller is not None:
            self._controller.close()
        self._closed = True
        super().close()
