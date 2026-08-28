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

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from threading import Event
from time import monotonic
from typing import Any

import gymnasium as gym
import numpy as np
from f1_robot_controller import (
    CommandStatus,
    ControllerError,
    ControllerNotReadyError,
    DualArmTcpCommand,
    F1RobotController,
    ObservationUnavailableError,
    RobotObservation,
    create_controller,
    load_controller_config,
    validate_motion_envelope,
)
from PIL import Image

F1_STATE_ORDER = (
    "left_joint_position",
    "left_gripper",
    "right_joint_position",
    "right_gripper",
)
_FAILED_COMMAND_STATUSES = frozenset(
    {
        CommandStatus.REJECTED,
        CommandStatus.EXPIRED,
        CommandStatus.CANCELLED,
        CommandStatus.FAULT,
    }
)
_IN_FLIGHT_COMMAND_STATUSES = frozenset(
    {
        CommandStatus.ACCEPTED,
        CommandStatus.DISPATCHED,
        CommandStatus.ACTIVE,
    }
)
_LEFT_ACTION = slice(0, 6)
_LEFT_GRIPPER_ACTION = 6
_RIGHT_ACTION = slice(7, 13)
_RIGHT_GRIPPER_ACTION = 13


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_float(name: str, value: object) -> float:
    normalized = _finite_float(name, value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative_float(name: str, value: object) -> float:
    normalized = _finite_float(name, value)
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


@dataclass
class F1RobotConfig:
    """Configuration shared by F1 real-world tasks."""

    controller: Mapping[str, Any]
    action_scale: Mapping[str, Any]
    motion_envelope: Mapping[str, Any]
    max_num_steps: int = 10
    reset_duration_s: float = 0.01
    reset_timeout_s: float = 30.0
    tcp_position_tolerance_m: float = 0.005
    tcp_orientation_tolerance_deg: float = 1.0
    gripper_tolerance_percent_closed: float = 1.0
    policy_image_shape: tuple[int, int] = (128, 128)
    control_period_s: float = field(default=0.01, init=False)
    max_observation_age_s: float = field(default=0.25, init=False)
    max_observation_skew_s: float = field(default=0.05, init=False)
    _action_scale_vector: tuple[float, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize immutable sequences and reject unsafe configuration."""

        if isinstance(self.max_num_steps, bool) or not isinstance(
            self.max_num_steps, Integral
        ):
            raise TypeError("max_num_steps must be an integer")
        if self.max_num_steps <= 0:
            raise ValueError("max_num_steps must be positive")
        self.max_num_steps = int(self.max_num_steps)

        positive_fields = ("reset_duration_s", "reset_timeout_s")
        for name in positive_fields:
            setattr(self, name, _positive_float(name, getattr(self, name)))
        self.tcp_position_tolerance_m = _positive_float(
            "tcp_position_tolerance_m",
            self.tcp_position_tolerance_m,
        )
        self.tcp_orientation_tolerance_deg = _positive_float(
            "tcp_orientation_tolerance_deg",
            self.tcp_orientation_tolerance_deg,
        )
        self.gripper_tolerance_percent_closed = _positive_float(
            "gripper_tolerance_percent_closed",
            self.gripper_tolerance_percent_closed,
        )
        if (
            not isinstance(self.policy_image_shape, (list, tuple))
            or len(self.policy_image_shape) != 2
        ):
            raise ValueError("policy_image_shape must be [height, width]")
        self.policy_image_shape = tuple(int(dim) for dim in self.policy_image_shape)
        if any(dim <= 0 for dim in self.policy_image_shape):
            raise ValueError("policy_image_shape dimensions must be positive")
        if not isinstance(self.controller, Mapping):
            raise TypeError("controller must be an inline mapping")
        if not isinstance(self.motion_envelope, Mapping):
            raise TypeError("motion_envelope must be an inline mapping")
        if not isinstance(self.action_scale, Mapping):
            raise TypeError("action_scale must be an inline mapping")

        motion_envelope = deepcopy(dict(self.motion_envelope))
        validate_motion_envelope(motion_envelope)

        controller = deepcopy(dict(self.controller))
        controller["motion_envelope"] = motion_envelope
        loaded_controller = load_controller_config(controller)

        action_scale = {
            "tcp_position_m": _positive_float(
                "action_scale.tcp_position_m",
                self.action_scale.get("tcp_position_m"),
            ),
            "tcp_orientation_deg": _positive_float(
                "action_scale.tcp_orientation_deg",
                self.action_scale.get("tcp_orientation_deg"),
            ),
            "gripper_percent_closed": _positive_float(
                "action_scale.gripper_percent_closed",
                self.action_scale.get("gripper_percent_closed"),
            ),
        }
        extra_action_scale = set(self.action_scale) - set(action_scale)
        if extra_action_scale:
            raise ValueError(
                "action_scale contains unrecognized field "
                f"{sorted(extra_action_scale)[0]}"
            )

        self.controller = controller
        self.motion_envelope = motion_envelope
        self.action_scale = action_scale
        self.control_period_s = loaded_controller.control_period_s
        self.max_observation_age_s = loaded_controller.max_observation_age_s
        self.max_observation_skew_s = loaded_controller.max_observation_skew_s
        self._action_scale_vector = (
            action_scale["tcp_position_m"],
            action_scale["tcp_position_m"],
            action_scale["tcp_position_m"],
            action_scale["tcp_orientation_deg"],
            action_scale["tcp_orientation_deg"],
            action_scale["tcp_orientation_deg"],
            action_scale["gripper_percent_closed"],
            action_scale["tcp_position_m"],
            action_scale["tcp_position_m"],
            action_scale["tcp_position_m"],
            action_scale["tcp_orientation_deg"],
            action_scale["tcp_orientation_deg"],
            action_scale["tcp_orientation_deg"],
            action_scale["gripper_percent_closed"],
        )

    @property
    def action_scale_vector(self) -> np.ndarray:
        """Return the 14D normalized-action scale in physical units."""

        return np.array(self._action_scale_vector, dtype=np.float64)


class F1RobotEnv(gym.Env, ABC):
    """Own the F1 Controller lifecycle and common action/observation schema."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: F1RobotConfig,
        *,
        controller_factory: Callable[[Mapping[str, Any]], F1RobotController]
        | None = None,
    ) -> None:
        """Open one Controller and publish the phase-one Gym spaces.

        Args:
            config: F1 Controller, timing, action-scale, and reset configuration.
        """

        super().__init__()
        self.config = config
        self._motion_envelope = config.motion_envelope
        action_low = np.full(14, -1.0, dtype=np.float32)
        action_high = np.full(14, 1.0, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=action_low,
            high=action_high,
            shape=(14,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    OrderedDict((("proprioception", self._float_vector_space(16)),))
                ),
                "frames": gym.spaces.Dict(
                    {
                        "head_color": self._rgb_frame_space(
                            (*config.policy_image_shape, 3)
                        ),
                        "left_wrist_color": self._rgb_frame_space(
                            (*config.policy_image_shape, 3)
                        ),
                        "right_wrist_color": self._rgb_frame_space(
                            (*config.policy_image_shape, 3)
                        ),
                    }
                ),
            }
        )
        self._closed = False
        self._controller: F1RobotController | None = None
        self._next_command_id = 0
        self._num_steps = 0
        self._period_wait = Event()
        self._session_origin: np.ndarray | None = None
        self._session_id = id(self)
        factory = (
            create_controller if controller_factory is None else controller_factory
        )
        controller = factory(config.controller)
        self._controller = controller
        try:
            controller.open()
            controller.wait_ready(timeout_s=config.reset_timeout_s)
            self._session_origin = self._state_vector(self._read_observation())
        except BaseException:
            controller.close()
            self._closed = True
            raise

    @property
    @abstractmethod
    def task_description(self) -> str:
        """Return the policy-facing instruction for the concrete task."""

    @staticmethod
    def _float_vector_space(size: int) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(size,),
            dtype=np.float32,
        )

    @staticmethod
    def _rgb_frame_space(shape: tuple[int, int, int]) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
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
    def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} is required")
        return value

    def _policy_observation(self, observation: RobotObservation) -> dict[str, Any]:
        return {
            "state": {
                "proprioception": self._state_vector(observation).astype(np.float32)
            },
            "frames": {
                "head_color": self._resize_policy_image(observation.head_color_rgb),
                "left_wrist_color": self._resize_policy_image(
                    observation.left_wrist_color_rgb
                ),
                "right_wrist_color": self._resize_policy_image(
                    observation.right_wrist_color_rgb
                ),
            },
        }

    def _resize_policy_image(self, image: np.ndarray) -> np.ndarray:
        target_height, target_width = self.config.policy_image_shape
        image = np.asarray(image, dtype=np.uint8)
        if image.shape == (target_height, target_width, 3):
            return np.array(image, copy=True)
        resized = Image.fromarray(image).resize(
            (target_width, target_height),
            resample=Image.Resampling.BILINEAR,
        )
        return np.array(resized, dtype=np.uint8, copy=True)

    @staticmethod
    def _state_vector(observation: RobotObservation) -> np.ndarray:
        """Return the canonical 16D joint/gripper state vector."""

        state_components = {
            "left_joint_position": np.asarray(
                observation.left_joint_position_rad, dtype=np.float64
            ),
            "left_gripper": np.asarray(
                [observation.left_gripper_position], dtype=np.float64
            ),
            "right_joint_position": np.asarray(
                observation.right_joint_position_rad, dtype=np.float64
            ),
            "right_gripper": np.asarray(
                [observation.right_gripper_position], dtype=np.float64
            ),
        }
        return np.concatenate(
            [state_components[name] for name in F1_STATE_ORDER], axis=0
        )

    @staticmethod
    def _validated_action(action: np.ndarray) -> np.ndarray:
        try:
            policy_action = np.array(action, dtype=np.float32, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("action must be a numeric 14D TCP array") from error
        if policy_action.shape != (14,):
            raise ValueError("action must have shape (14,)")
        if not np.all(np.isfinite(policy_action)):
            raise ValueError("action must contain only finite values")
        return policy_action

    @staticmethod
    def _validated_tcp_vector(value: object, name: str) -> np.ndarray:
        value = np.array(value, dtype=np.float64, copy=True)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ObservationUnavailableError(
                f"controller observation field {name} must be finite shape (6,)"
            )
        return value

    @staticmethod
    def _validated_gripper_percent(value: object, name: str) -> float:
        value = _finite_float(name, value)
        if not 0.0 <= value <= 100.0:
            raise ObservationUnavailableError(f"{name} must be in [0, 100]")
        return value

    def _tcp_state(
        self,
        observation: RobotObservation,
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        try:
            return (
                self._validated_tcp_vector(
                    observation.left_tcp_pose_m_deg,
                    "left_tcp_pose_m_deg",
                ),
                self._validated_gripper_percent(
                    observation.left_gripper_position,
                    "left_gripper_position_percent_closed",
                ),
                self._validated_tcp_vector(
                    observation.right_tcp_pose_m_deg,
                    "right_tcp_pose_m_deg",
                ),
                self._validated_gripper_percent(
                    observation.right_gripper_position,
                    "right_gripper_position_percent_closed",
                ),
            )
        except AttributeError as error:
            raise ObservationUnavailableError(
                "controller observation is missing public TCP state fields"
            ) from error

    def _arm_tcp_envelope(self, arm_name: str) -> Mapping[str, Any]:
        if self._motion_envelope is None:
            raise RuntimeError(f"{arm_name} motion envelope is unavailable")
        return self._motion_envelope[arm_name]["tcp"]

    @staticmethod
    def _tcp_delta_bounds(tcp_envelope: Mapping[str, Any]) -> np.ndarray:
        max_delta = tcp_envelope["max_delta"]
        return np.array(
            [
                max_delta["position_m"],
                max_delta["position_m"],
                max_delta["position_m"],
                max_delta["orientation_deg"],
                max_delta["orientation_deg"],
                max_delta["orientation_deg"],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _arm_delta_bounds(arm_envelope: Mapping[str, Any]) -> np.ndarray:
        return np.array(
            [
                *F1RobotEnv._tcp_delta_bounds(arm_envelope["tcp"]),
                100.0,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _action_bounds(
        motion_envelope: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        high = np.concatenate(
            [
                F1RobotEnv._arm_delta_bounds(motion_envelope["left_arm"]),
                F1RobotEnv._arm_delta_bounds(motion_envelope["right_arm"]),
            ]
        ).astype(np.float32)
        return -high, high

    @staticmethod
    def _tcp_limit_bounds(
        motion_envelope: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        left_low, left_high = F1RobotEnv._tcp_workspace_low_high(
            motion_envelope["left_arm"]["tcp"]
        )
        right_low, right_high = F1RobotEnv._tcp_workspace_low_high(
            motion_envelope["right_arm"]["tcp"]
        )
        return np.concatenate([left_low, right_low]), np.concatenate(
            [left_high, right_high]
        )

    @staticmethod
    def _tcp_workspace_low_high(
        tcp_envelope: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        bounds = tcp_envelope["workspace_bounds"]
        low = np.array(
            [
                bounds["position_m"]["x"]["min"],
                bounds["position_m"]["y"]["min"],
                bounds["position_m"]["z"]["min"],
                bounds["orientation_deg"]["rx"]["min"],
                bounds["orientation_deg"]["ry"]["min"],
                bounds["orientation_deg"]["rz"]["min"],
            ],
            dtype=np.float64,
        )
        high = np.array(
            [
                bounds["position_m"]["x"]["max"],
                bounds["position_m"]["y"]["max"],
                bounds["position_m"]["z"]["max"],
                bounds["orientation_deg"]["rx"]["max"],
                bounds["orientation_deg"]["ry"]["max"],
                bounds["orientation_deg"]["rz"]["max"],
            ],
            dtype=np.float64,
        )
        return low, high

    def _bounded_tcp_target(
        self,
        *,
        arm_name: str,
        current: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        del arm_name
        target = current + action.astype(np.float64)
        return target

    def _bounded_gripper_target(
        self,
        *,
        arm_name: str,
        current_percent_closed: float,
        action: float,
    ) -> float:
        del arm_name
        target = current_percent_closed + float(action)
        return float(target)

    def _physical_policy_delta(self, policy_action: np.ndarray) -> np.ndarray:
        return policy_action.astype(np.float64) * self.config.action_scale_vector

    def _absolute_command(
        self,
        policy_action: np.ndarray,
        observation: RobotObservation,
    ) -> DualArmTcpCommand:
        left_tcp, left_gripper, right_tcp, right_gripper = self._tcp_state(observation)
        physical_delta = self._physical_policy_delta(policy_action)
        return DualArmTcpCommand(
            command_id=self._allocate_command_id(),
            left_tcp_target_m_deg=self._bounded_tcp_target(
                arm_name="left_arm",
                current=left_tcp,
                action=physical_delta[_LEFT_ACTION],
            ),
            left_gripper_target=self._bounded_gripper_target(
                arm_name="left_arm",
                current_percent_closed=left_gripper,
                action=float(physical_delta[_LEFT_GRIPPER_ACTION]),
            ),
            right_tcp_target_m_deg=self._bounded_tcp_target(
                arm_name="right_arm",
                current=right_tcp,
                action=physical_delta[_RIGHT_ACTION],
            ),
            right_gripper_target=self._bounded_gripper_target(
                arm_name="right_arm",
                current_percent_closed=right_gripper,
                action=float(physical_delta[_RIGHT_GRIPPER_ACTION]),
            ),
            duration_s=self.config.control_period_s,
            created_at_monotonic_s=monotonic(),
        )

    @abstractmethod
    def _calc_step_reward(self, observation: dict[str, Any]) -> float:
        """Compute the concrete task reward for one post-action observation."""

    @abstractmethod
    def _is_success(self, observation: dict[str, Any]) -> bool:
        """Return whether the concrete task succeeded after the action."""

    def _require_healthy_controller(self, context: str) -> None:
        health = self._active_controller.health()
        if not health.ready or health.faulted:
            detail = health.reason or "controller is not ready"
            raise ControllerError(f"{context}: {detail}")

    @staticmethod
    def _require_successful_status(
        status: CommandStatus,
        *,
        command_id: int,
        context: str,
    ) -> None:
        if status in _FAILED_COMMAND_STATUSES:
            raise ControllerError(
                f"{context} command {command_id} failed with {status.value}"
            )

    def _wait_for_finished_dispatch(
        self,
        *,
        command_id: int,
        context: str,
        deadline_s: float,
    ) -> tuple[CommandStatus, float]:
        started_at_s: float | None = None
        while True:
            status = self._active_controller.get_command_status(command_id)
            if status is CommandStatus.FINISHED_DISPATCH:
                if started_at_s is None:
                    return status, 0.0
                return status, monotonic() - started_at_s
            if status in _FAILED_COMMAND_STATUSES:
                detail = ""
                if status is CommandStatus.FAULT:
                    try:
                        health = self._active_controller.health()
                    except Exception:
                        pass
                    else:
                        if health.reason:
                            detail = f": {health.reason}"
                raise ControllerError(
                    f"{context} command {command_id} failed with {status.value}{detail}"
                )
            if status not in _IN_FLIGHT_COMMAND_STATUSES:
                raise ControllerError(
                    f"{context} command {command_id} returned unsupported "
                    f"status {status.value}"
                )
            if started_at_s is None:
                started_at_s = monotonic()
            remaining_s = deadline_s - monotonic()
            if remaining_s <= 0.0:
                raise TimeoutError(
                    f"{context} command {command_id} did not reach "
                    f"{CommandStatus.FINISHED_DISPATCH.value} within "
                    f"{self.config.control_period_s} seconds"
                )
            self._period_wait.wait(min(self.config.control_period_s, remaining_s))

    def _stop_after_failure(self, context: str, error: BaseException) -> None:
        try:
            self._active_controller.stop_command_dispatch(f"{context} failed: {error}")
        except BaseException:
            # Preserve the failure that required the safety stop.
            pass

    @staticmethod
    def _stop_previous_motion_if_active(
        controller: F1RobotController,
    ) -> None:
        try:
            health = controller.health()
        except ControllerNotReadyError:
            return
        if health.ready and not health.faulted:
            controller.stop_command_dispatch("starting robot reset")

    def recapture_reset_snapshot(self) -> None:
        """Explicitly replace the session origin from measured robot state."""

        observation = self._read_observation()
        self._session_origin = self._state_vector(observation)

    @abstractmethod
    def _reset_task(
        self,
        *,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Perform task-specific reset work and return task reset metadata."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the task reset hook and start an episode from its final state."""

        super().reset(seed=seed)
        controller = self._active_controller
        try:
            controller.open()
            controller.wait_ready(timeout_s=self.config.reset_timeout_s)
            self._require_healthy_controller("reset readiness check failed")
            reset_info = self._reset_task(options=options)
            if not isinstance(reset_info, dict):
                raise TypeError("_reset_task must return a dict")
            measured = self._read_observation()
            self._session_origin = self._state_vector(measured)
            self._num_steps = 0
            info = dict(reset_info)
            info["session_origin_state"] = self._session_origin.copy()
            return self._policy_observation(measured), info
        except BaseException as error:
            self._stop_after_failure("reset", error)
            raise

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one physical TCP delta as one absolute Controller command."""

        try:
            measured_before = self._read_observation()
            policy_action = self._validated_action(action)
            if not self.action_space.contains(policy_action):
                raise ValueError("action must stay within normalized [-1, 1] bounds")
            command = self._absolute_command(policy_action, measured_before)
        except BaseException as error:
            self._stop_after_failure("policy setup", error)
            raise
        try:
            receipt = self._active_controller.submit_command(command)
            period_deadline = monotonic() + self.config.control_period_s
            dispatch_deadline = monotonic() + max(
                self.config.control_period_s * 3.0,
                self.config.control_period_s,
            )
            status, dispatch_latency_s = self._wait_for_finished_dispatch(
                command_id=command.command_id,
                context="policy",
                deadline_s=dispatch_deadline,
            )
            remaining_s = period_deadline - monotonic()
            if remaining_s > 0.0:
                self._period_wait.wait(remaining_s)
            measured_after = self._read_observation(
                newer_than=receipt.accepted_at_monotonic_s
            )
            self._require_healthy_controller("policy health check failed")
        except BaseException as error:
            self._stop_after_failure("policy motion", error)
            raise
        observation = self._policy_observation(measured_after)
        self._num_steps += 1
        reward = self._calc_step_reward(observation)
        terminated = self._is_success(observation)
        truncated = self._num_steps >= self.config.max_num_steps
        info = {
            "policy_action": policy_action.copy(),
            "physical_delta_commanded": self._physical_policy_delta(policy_action),
            "command_id": command.command_id,
            "command_status": status,
            "dispatch_latency_s": dispatch_latency_s,
            "absolute_left_tcp_target_m_deg": np.array(
                command.left_tcp_target_m_deg,
                dtype=np.float64,
                copy=True,
            ),
            "absolute_left_gripper_target": command.left_gripper_target,
            "absolute_right_tcp_target_m_deg": np.array(
                command.right_tcp_target_m_deg,
                dtype=np.float64,
                copy=True,
            ),
            "absolute_right_gripper_target": command.right_gripper_target,
            "absolute_target_commanded": {
                "left_arm": {
                    "tcp_pose": {
                        "position_m": np.asarray(command.left_tcp_target_m_deg[:3])
                        .astype(float)
                        .tolist(),
                        "orientation_deg": np.asarray(command.left_tcp_target_m_deg[3:])
                        .astype(float)
                        .tolist(),
                    },
                    "gripper_percent_closed": float(command.left_gripper_target),
                },
                "right_arm": {
                    "tcp_pose": {
                        "position_m": np.asarray(command.right_tcp_target_m_deg[:3])
                        .astype(float)
                        .tolist(),
                        "orientation_deg": np.asarray(
                            command.right_tcp_target_m_deg[3:]
                        )
                        .astype(float)
                        .tolist(),
                    },
                    "gripper_percent_closed": float(command.right_gripper_target),
                },
            },
            "command_created_at_monotonic_s": command.created_at_monotonic_s,
            "command_accepted_at_monotonic_s": receipt.accepted_at_monotonic_s,
            "command_expires_at_monotonic_s": receipt.expires_at_monotonic_s,
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
