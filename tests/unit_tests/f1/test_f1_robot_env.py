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

import importlib.util
import inspect
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from types import ModuleType, SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from f1_robot_controller import (
    CommandReceipt,
    CommandStatus,
    ControllerError,
    ControllerHealth,
    ControllerLifecycleError,
    ControllerNotReadyError,
    DualArmTcpCommand,
    DualArmTcpResetCommand,
    F1RobotController,
    ObservationUnavailableError,
    SensorTimestamps,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
F1_PACKAGE_DIR = ROOT / "rlinf" / "envs" / "realworld" / "f1"


def _load_f1_package() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_f1_env_under_test",
        F1_PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(F1_PACKAGE_DIR)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the F1 Env package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


F1_PACKAGE = _load_f1_package()
F1RobotConfig = F1_PACKAGE.F1RobotConfig
F1RobotEnvBase = F1_PACKAGE.F1RobotEnv
F1_ENV_MODULE = sys.modules["_f1_env_under_test.f1_robot_env"]


class F1RobotEnv(F1RobotEnvBase):
    """Concrete task double for exercising the F1 platform template."""

    task_reset_options: dict[str, Any] | None = None

    @property
    def task_description(self) -> str:
        return "F1 platform test task"

    def _reset_task(self, *, options: dict[str, Any] | None) -> dict[str, Any]:
        self.task_reset_options = options
        events = getattr(self._active_controller, "events", None)
        if events is not None:
            events.append(("reset_task", options))
        return {"reset_mode": "current_state_origin"}

    def _calc_step_reward(self, observation: dict[str, Any]) -> float:
        del observation
        return 0.0

    def _is_success(self, observation: dict[str, Any]) -> bool:
        del observation
        return False


EXPECTED_F1_STATE_ORDER = (
    "left_joint_position",
    "left_gripper",
    "right_joint_position",
    "right_gripper",
)

SENSOR_NAMES = (
    "left_joint_position",
    "left_gripper_position",
    "right_joint_position",
    "right_gripper_position",
    "head_color",
    "left_wrist_color",
    "right_wrist_color",
)


def _approved_motion_envelope() -> dict[str, object]:
    arm = {
        "tcp": {
            "max_delta": {"position_m": 0.005, "orientation_deg": 1.0},
            "workspace_bounds": {
                "position_m": {
                    "x": {"min": -1.0, "max": 1.0},
                    "y": {"min": -1.0, "max": 1.0},
                    "z": {"min": 0.0, "max": 1.0},
                },
                "orientation_deg": {
                    "rx": {"min": -180.0, "max": 180.0},
                    "ry": {"min": -180.0, "max": 180.0},
                    "rz": {"min": -180.0, "max": 180.0},
                },
            },
        }
    }
    return {
        "left_arm": arm,
        "right_arm": arm,
        "gripper_percent_closed": [0.0, 100.0],
    }


def _fake_controller_config(**overrides: object) -> dict[str, object]:
    return {
        "backend": "fake",
        "control_period_s": overrides.pop("control_period_s", 0.001),
        "max_observation_age_s": overrides.pop("max_observation_age_s", 0.25),
        "max_observation_skew_s": overrides.pop("max_observation_skew_s", 0.05),
        "fake": {
            "seed": 7,
            "image_height": overrides.pop("image_height", 128),
            "image_width": overrides.pop("image_width", 128),
            "command_latency_s": 0.0,
            "command_history_limit": 64,
        },
    }


def _action_scale() -> dict[str, float]:
    return {
        "tcp_position_m": 1.0,
        "tcp_orientation_deg": 1.0,
        "gripper_percent_closed": 1.0,
    }


def _f1_config(**overrides: object) -> F1RobotConfig:
    controller_overrides = {}
    for key in (
        "control_period_s",
        "max_observation_age_s",
        "max_observation_skew_s",
        "image_height",
        "image_width",
    ):
        if key in overrides:
            controller_overrides[key] = overrides.pop(key)
    overrides.pop("is_dummy", None)
    overrides.pop("architecture_smoke", None)
    overrides.pop("phase2_handoff", None)
    overrides.pop("command_capability", None)
    return F1RobotConfig(
        controller=_fake_controller_config(**controller_overrides),
        action_scale=_action_scale(),
        motion_envelope=_approved_motion_envelope(),
        **overrides,
    )


def _robot_observation(
    *,
    left_joints: np.ndarray | None = None,
    left_gripper: float | None = None,
    right_joints: np.ndarray | None = None,
    right_gripper: float | None = None,
    left_tcp: np.ndarray | None = None,
    left_gripper_percent_closed: float = 50.0,
    right_tcp: np.ndarray | None = None,
    right_gripper_percent_closed: float = 50.0,
) -> SimpleNamespace:
    if left_joints is None:
        left_joints = np.linspace(0.1, 0.7, 7)
    if right_joints is None:
        right_joints = np.linspace(-0.1, -0.7, 7)
    if left_tcp is None:
        left_tcp = np.array([0.3, 0.2, 0.4, 0.0, 0.0, 0.0], dtype=np.float64)
    if right_tcp is None:
        right_tcp = np.array([0.3, -0.2, 0.4, 0.0, 0.0, 0.0], dtype=np.float64)
    if left_gripper is None:
        left_gripper = left_gripper_percent_closed
    if right_gripper is None:
        right_gripper = right_gripper_percent_closed
    return SimpleNamespace(
        left_joint_position_rad=np.array(left_joints, dtype=np.float64, copy=True),
        left_gripper_position=left_gripper,
        right_joint_position_rad=np.array(right_joints, dtype=np.float64, copy=True),
        right_gripper_position=right_gripper,
        left_tcp_pose_m_deg=np.array(left_tcp, dtype=np.float64, copy=True),
        left_gripper_percent_closed=left_gripper_percent_closed,
        right_tcp_pose_m_deg=np.array(right_tcp, dtype=np.float64, copy=True),
        right_gripper_percent_closed=right_gripper_percent_closed,
        head_color_rgb=np.full((128, 128, 3), 11, dtype=np.uint8),
        left_wrist_color_rgb=np.full((128, 128, 3), 22, dtype=np.uint8),
        right_wrist_color_rgb=np.full((128, 128, 3), 33, dtype=np.uint8),
        timestamps=SensorTimestamps(
            source_timestamp_s=dict.fromkeys(SENSOR_NAMES, 1.0),
            received_at_monotonic_s=dict.fromkeys(SENSOR_NAMES, 1.0),
        ),
    )


class RecordingController(F1RobotController):
    """Complete in-process Controller double recording Env boundary behavior."""

    def __init__(
        self,
        observation: Any | None = None,
        *,
        converge_reset: bool = True,
        policy_status: CommandStatus = CommandStatus.FINISHED_DISPATCH,
        reset_status: CommandStatus = CommandStatus.FINISHED_DISPATCH,
        health: ControllerHealth | None = None,
        ready_error: Exception | None = None,
    ) -> None:
        self.observation = observation or _robot_observation()
        self.converge_reset = converge_reset
        self.policy_status = policy_status
        self.reset_status = reset_status
        self.controller_health = health or ControllerHealth(
            ready=True,
            faulted=False,
            reason=None,
            checked_at_monotonic_s=1.0,
        )
        self.health_error_once: Exception | None = None
        self.ready_error = ready_error
        self.read_error: Exception | None = None
        self.stop_error: Exception | None = None
        self.policy_commands: list[DualArmTcpCommand] = []
        self.reset_commands: list[DualArmTcpResetCommand] = []
        self.statuses: dict[int, CommandStatus] = {}
        self.status_sequences: dict[int, list[CommandStatus]] = {}
        self.read_requests: list[dict[str, float | None]] = []
        self.events: list[tuple[str, object]] = []
        self.stop_reasons: list[str] = []
        self.close_calls = 0
        self.open_calls = 0
        self.wait_ready_calls: list[float] = []
        self.last_receipt: CommandReceipt | None = None
        self.submitted_at_s: float | None = None
        self.post_submit_read_at_s: float | None = None

    def open(self) -> None:
        self.open_calls += 1
        self.events.append(("open", None))

    def wait_ready(self, timeout_s: float) -> None:
        self.wait_ready_calls.append(timeout_s)
        self.events.append(("wait_ready", timeout_s))
        if self.ready_error is not None:
            raise self.ready_error

    def read_observation(
        self,
        *,
        max_age_s: float,
        max_skew_s: float,
        newer_than: float | None = None,
    ) -> Any:
        self.events.append(("read_observation", newer_than))
        self.read_requests.append(
            {
                "max_age_s": max_age_s,
                "max_skew_s": max_skew_s,
                "newer_than": newer_than,
            }
        )
        if self.policy_commands:
            self.post_submit_read_at_s = monotonic()
        if self.read_error is not None:
            raise self.read_error
        return self.observation

    def submit_command(self, command: DualArmTcpCommand) -> CommandReceipt:
        self.events.append(("submit_command", command.command_id))
        self.policy_commands.append(command)
        self.submitted_at_s = monotonic()
        head_color_rgb = np.array(self.observation.head_color_rgb, copy=True)
        left_wrist_color_rgb = np.array(
            self.observation.left_wrist_color_rgb, copy=True
        )
        right_wrist_color_rgb = np.array(
            self.observation.right_wrist_color_rgb,
            copy=True,
        )
        self.observation = _robot_observation(
            left_joints=self.observation.left_joint_position_rad,
            left_gripper=command.left_gripper_target,
            right_joints=self.observation.right_joint_position_rad,
            right_gripper=command.right_gripper_target,
            left_tcp=command.left_tcp_target_m_deg,
            left_gripper_percent_closed=command.left_gripper_target,
            right_tcp=command.right_tcp_target_m_deg,
            right_gripper_percent_closed=command.right_gripper_target,
        )
        self.observation.head_color_rgb = head_color_rgb
        self.observation.left_wrist_color_rgb = left_wrist_color_rgb
        self.observation.right_wrist_color_rgb = right_wrist_color_rgb
        receipt = CommandReceipt(
            command_id=command.command_id,
            accepted_at_monotonic_s=100.0 + command.command_id,
            expires_at_monotonic_s=101.0 + command.command_id,
        )
        self.observation.timestamps = SensorTimestamps(
            source_timestamp_s=dict.fromkeys(SENSOR_NAMES, 1.0),
            received_at_monotonic_s=dict.fromkeys(
                SENSOR_NAMES, receipt.accepted_at_monotonic_s + 1.0
            ),
        )
        self.last_receipt = receipt
        self.statuses[command.command_id] = self.policy_status
        return receipt

    def submit_reset_command(self, command: DualArmTcpResetCommand) -> CommandReceipt:
        self.events.append(("submit_reset_command", command.command_id))
        self.reset_commands.append(command)
        if self.converge_reset:
            self.observation = _robot_observation(
                left_joints=self.observation.left_joint_position_rad,
                left_gripper=command.left_gripper_target,
                right_joints=self.observation.right_joint_position_rad,
                right_gripper=command.right_gripper_target,
                left_tcp=command.left_tcp_target_m_deg,
                left_gripper_percent_closed=command.left_gripper_target,
                right_tcp=command.right_tcp_target_m_deg,
                right_gripper_percent_closed=command.right_gripper_target,
            )
        receipt = CommandReceipt(
            command_id=command.command_id,
            accepted_at_monotonic_s=100.0 + command.command_id,
            expires_at_monotonic_s=101.0 + command.command_id,
        )
        self.last_receipt = receipt
        self.statuses[command.command_id] = self.reset_status
        return receipt

    def get_command_status(self, command_id: int) -> CommandStatus:
        self.events.append(("get_command_status", command_id))
        sequence = self.status_sequences.get(command_id)
        if sequence:
            status = sequence.pop(0)
            self.statuses[command_id] = status
            return status
        return self.statuses[command_id]

    def health(self) -> ControllerHealth:
        self.events.append(("health", None))
        if self.health_error_once is not None:
            error = self.health_error_once
            self.health_error_once = None
            raise error
        return self.controller_health

    def stop_command_dispatch(self, reason: str) -> None:
        self.events.append(("stop_command_dispatch", reason))
        self.stop_reasons.append(reason)
        if self.stop_error is not None:
            raise self.stop_error

    def close(self) -> None:
        self.close_calls += 1


def _install_recording_controller(
    monkeypatch: pytest.MonkeyPatch,
    controller: RecordingController,
) -> dict[str, Any]:
    factory_call: dict[str, Any] = {}

    def factory(config: Mapping[str, Any]) -> RecordingController:
        factory_call.update(config=config)
        return controller

    monkeypatch.setattr(F1_ENV_MODULE, "create_controller", factory)
    return factory_call


def test_f1_robot_config_has_the_frozen_phase_one_defaults() -> None:
    config = _f1_config()

    assert asdict(config) == {
        "controller": {
            **_fake_controller_config(),
            "motion_envelope": _approved_motion_envelope(),
        },
        "action_scale": _action_scale(),
        "motion_envelope": _approved_motion_envelope(),
        "max_num_steps": 10,
        "reset_duration_s": 0.01,
        "reset_timeout_s": 30.0,
        "tcp_position_tolerance_m": 0.005,
        "tcp_orientation_tolerance_deg": 1.0,
        "gripper_tolerance_percent_closed": 1.0,
        "policy_image_shape": (128, 128),
        "post_action_observation_timeout_s": 0.25,
        "post_action_image_source": "head_color",
        "control_period_s": 0.001,
        "max_observation_age_s": 0.25,
        "max_observation_skew_s": 0.05,
        "_action_scale_vector": (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ),
    }


def test_f1_robot_env_requires_concrete_task_interfaces() -> None:
    assert inspect.isabstract(F1RobotEnvBase)
    assert {
        "task_description",
        "_reset_task",
        "_calc_step_reward",
        "_is_success",
    } <= F1RobotEnvBase.__abstractmethods__


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_dummy": 1},
        {"control_period_s": 0.0},
        {"max_observation_age_s": np.inf},
        {"max_observation_skew_s": -0.01},
        {"max_num_steps": True},
        {"max_num_steps": 0},
        {"max_num_steps": 1.5},
        {"reset_duration_s": 0.0},
        {"reset_timeout_s": np.inf},
        {"tcp_position_tolerance_m": 0.0},
        {"tcp_orientation_tolerance_deg": 0.0},
        {"gripper_tolerance_percent_closed": 0.0},
        {"policy_image_shape": (0, 128)},
        {"policy_image_shape": "128x128"},
        {"motion_envelope": object()},
    ],
)
def test_f1_robot_config_rejects_invalid_safety_values(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        F1RobotConfig(**overrides)


def test_f1_robot_config_validates_and_copies_the_motion_envelope() -> None:
    envelope = _approved_motion_envelope()

    config = F1RobotConfig(
        controller=_fake_controller_config(),
        action_scale=_action_scale(),
        motion_envelope=envelope,
    )
    envelope["gripper_percent_closed"] = [10.0, 90.0]

    assert config.motion_envelope == _approved_motion_envelope()
    assert config.controller["motion_envelope"] == _approved_motion_envelope()


def test_f1_exposes_one_canonical_policy_state_vector() -> None:
    assert F1_PACKAGE.F1_STATE_ORDER == EXPECTED_F1_STATE_ORDER

    env = F1RobotEnv(_f1_config())
    try:
        state_space = env.observation_space["state"]
        assert tuple(state_space.spaces) == ("proprioception",)
        assert state_space["proprioception"].shape == (16,)
    finally:
        env.close()


def test_f1_normalizes_policy_state_and_heterogeneous_images_before_public_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _robot_observation(
        left_joints=np.arange(1, 8, dtype=np.float64),
        left_gripper=8.0,
        right_joints=np.arange(9, 16, dtype=np.float64),
        right_gripper=16.0,
    )
    raw.head_color_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    raw.left_wrist_color_rgb = np.full((480, 848, 3), 1, dtype=np.uint8)
    raw.right_wrist_color_rgb = np.full((480, 848, 3), 2, dtype=np.uint8)
    controller = RecordingController(raw)
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config())
    try:
        observation, _ = env.reset()
        np.testing.assert_array_equal(
            observation["state"]["proprioception"],
            np.arange(1, 17, dtype=np.float32),
        )
        assert all(
            frame.shape == (128, 128, 3) for frame in observation["frames"].values()
        )
    finally:
        env.close()


def test_center_crop_and_resize_rgb_removes_equal_horizontal_margins() -> None:
    assert hasattr(F1_ENV_MODULE, "center_crop_and_resize_rgb")
    center_crop_and_resize_rgb = F1_ENV_MODULE.center_crop_and_resize_rgb
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :280] = [255, 0, 0]
    image[:, 1000:] = [0, 0, 255]
    image[:, 280:1000] = [0, 255, 0]

    resized = center_crop_and_resize_rgb(image, (128, 128))

    assert resized.shape == (128, 128, 3)
    assert resized.dtype == np.uint8
    assert np.all(resized == np.array([0, 255, 0], dtype=np.uint8))


def test_center_crop_and_resize_rgb_returns_an_independent_square_frame() -> None:
    assert hasattr(F1_ENV_MODULE, "center_crop_and_resize_rgb")
    center_crop_and_resize_rgb = F1_ENV_MODULE.center_crop_and_resize_rgb
    image = np.full((128, 128, 3), 17, dtype=np.uint8)

    resized = center_crop_and_resize_rgb(image, (128, 128))
    resized.fill(99)

    assert resized.shape == (128, 128, 3)
    assert np.all(image == 17)


def test_resize_rgb_preserves_the_full_rectangular_field_of_view() -> None:
    """Catch the right-wrist path silently center-cropping its source image."""

    assert hasattr(F1_ENV_MODULE, "resize_rgb")
    resize_rgb = F1_ENV_MODULE.resize_rgb
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[:, :2] = [255, 0, 0]
    image[:, -2:] = [0, 0, 255]

    resized = resize_rgb(image, (2, 4))

    assert resized.shape == (2, 4, 3)
    assert resized.dtype == np.uint8
    assert resized[0, 0, 0] > resized[0, 0, 2]
    assert resized[0, -1, 2] > resized[0, -1, 0]


def test_f1_robot_env_exposes_14d_tcp_action_16d_state_and_three_rgb_frames() -> None:
    env = F1RobotEnv(_f1_config())
    try:
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (14,)
        assert env.action_space.dtype == np.dtype(np.float32)
        np.testing.assert_allclose(env.action_space.low, [-1.0] * 14)
        np.testing.assert_allclose(env.action_space.high, [1.0] * 14)

        state_space = env.observation_space["state"]
        assert tuple(state_space.spaces) == ("proprioception",)
        assert sum(space.shape[0] for space in state_space.spaces.values()) == 16
        assert state_space["proprioception"].shape == (16,)

        frame_space = env.observation_space["frames"]
        assert set(frame_space.spaces) == {
            "head_color",
            "left_wrist_color",
            "right_wrist_color",
        }
        assert all(
            space.shape == (128, 128, 3) for space in frame_space.spaces.values()
        )
        assert all(
            space.dtype == np.dtype(np.uint8) for space in frame_space.spaces.values()
        )
    finally:
        env.close()


def test_f1_state_vector_follows_the_declared_state_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        F1_ENV_MODULE,
        "F1_STATE_ORDER",
        (
            "right_gripper",
            "right_joint_position",
            "left_gripper",
            "left_joint_position",
        ),
    )
    observation = _robot_observation(
        left_joints=np.arange(1, 8, dtype=np.float64),
        left_gripper=8.0,
        right_joints=np.arange(9, 16, dtype=np.float64),
        right_gripper=16.0,
    )

    np.testing.assert_array_equal(
        F1RobotEnv._state_vector(observation),
        np.array([16.0, *range(9, 16), 8.0, *range(1, 8)], dtype=np.float64),
    )


def _ros2_controller_config() -> dict[str, object]:
    return {
        "backend": "ros2",
        "control_period_s": 0.001,
        "max_observation_age_s": 0.25,
        "max_observation_skew_s": 0.05,
        "ros2": {
            "node_name": "f1_robot_controller",
            "sensor_topics": {
                "head_color": "/camera/head/color/image_raw/compressed",
                "left_wrist_color": "/camera/left/color/image_raw/compressed",
                "right_wrist_color": "/camera/right/color/image_raw/compressed",
                "joint_state": "/hal/joint_states",
                "left_gripper_position": "/motion_ctl/gripper/left/state",
                "right_gripper_position": "/motion_ctl/gripper/right/state",
            },
            "image_transports": {
                "head_color": "compressed",
                "left_wrist_color": "compressed",
                "right_wrist_color": "compressed",
            },
            "left_joint_names": tuple(f"arm_l_j{idx}" for idx in range(1, 8)),
            "right_joint_names": tuple(f"arm_r_j{idx}" for idx in range(1, 8)),
            "joint_position_scale_to_rad": 0.017453292519943295,
            "left_gripper_raw_open": 0.0,
            "left_gripper_raw_closed": 100.0,
            "right_gripper_raw_open": 0.0,
            "right_gripper_raw_closed": 100.0,
            "command_topics": {
                "left_tcp": "/motion_ctl/left_arm/tcp_pos_ctl",
                "right_tcp": "/motion_ctl/right_arm/tcp_pos_ctl",
                "left_gripper": "/motion_ctl/gripper/left",
                "right_gripper": "/motion_ctl/gripper/right",
            },
            "command_state_topics": {
                "left_tcp": "/state/left_arm/tcp_pos",
                "right_tcp": "/state/right_arm/tcp_pos",
                "left_gripper": "/motion_ctl/gripper/left/state",
                "right_gripper": "/motion_ctl/gripper/right/state",
            },
        },
    }


def _ros2_f1_config(**overrides: object) -> F1RobotConfig:
    return F1RobotConfig(
        controller=_ros2_controller_config(),
        action_scale=_action_scale(),
        motion_envelope=_approved_motion_envelope(),
        **overrides,
    )


def test_real_backend_builds_controller_020_ros2_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    factory_call = _install_recording_controller(monkeypatch, controller)

    env = F1RobotEnv(_ros2_f1_config())
    try:
        controller_config = factory_call["config"]
        assert controller_config["backend"] == "ros2"
        assert controller_config["motion_envelope"] == _approved_motion_envelope()
        assert controller_config["ros2"] == _ros2_controller_config()["ros2"]
        assert controller.open_calls == 1
    finally:
        env.close()


@pytest.mark.parametrize(
    "missing_field",
    [
        "node_name",
        "sensor_topics",
        "image_transports",
        "left_joint_names",
        "right_joint_names",
        "joint_position_scale_to_rad",
        "left_gripper_raw_open",
        "left_gripper_raw_closed",
        "right_gripper_raw_open",
        "right_gripper_raw_closed",
        "command_topics",
        "command_state_topics",
    ],
)
def test_real_backend_requires_strict_ros2_controller_fields_before_factory(
    monkeypatch: pytest.MonkeyPatch,
    missing_field: str,
) -> None:
    def fail_if_factory_runs(config: Mapping[str, object]) -> RecordingController:
        del config
        raise AssertionError("factory must not run with invalid ros2 config")

    monkeypatch.setattr(F1_ENV_MODULE, "create_controller", fail_if_factory_runs)
    controller_config = _ros2_controller_config()
    ros2 = dict(controller_config["ros2"])
    ros2.pop(missing_field)
    controller_config["ros2"] = ros2

    with pytest.raises(ValueError, match=missing_field):
        F1RobotEnv(
            F1RobotConfig(
                controller=controller_config,
                action_scale=_action_scale(),
                motion_envelope=_approved_motion_envelope(),
            )
        )


def test_real_backend_step_uses_controller_020_mapping_and_14d_delta_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    factory_call = _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_ros2_f1_config())
    action = np.array([0.001, 0, 0, 0, 0, 0, 0] + [0] * 7, dtype=np.float64)
    expected_policy_action = action.astype(np.float32)
    try:
        observation, reward, terminated, truncated, info = env.step(action)

        assert factory_call["config"]["ros2"]["command_topics"] == {
            "left_tcp": "/motion_ctl/left_arm/tcp_pos_ctl",
            "right_tcp": "/motion_ctl/right_arm/tcp_pos_ctl",
            "left_gripper": "/motion_ctl/gripper/left",
            "right_gripper": "/motion_ctl/gripper/right",
        }
        assert len(controller.policy_commands) == 1
        command = controller.policy_commands[0]
        assert isinstance(command, DualArmTcpCommand)
        np.testing.assert_allclose(
            command.left_tcp_target_m_deg,
            np.array([0.301, 0.2, 0.4, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            command.right_tcp_target_m_deg,
            np.array([0.3, -0.2, 0.4, 0.0, 0.0, 0.0]),
        )
        assert command.left_gripper_target == pytest.approx(50.0)
        assert command.right_gripper_target == pytest.approx(50.0)
        assert controller.read_requests[-1]["newer_than"] is None
        np.testing.assert_array_equal(info["policy_action"], expected_policy_action)
        np.testing.assert_array_equal(
            info["absolute_left_tcp_target_m_deg"],
            command.left_tcp_target_m_deg,
        )
        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_step_submits_one_physical_absolute_command_and_preserves_policy_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    factory_call = _install_recording_controller(monkeypatch, controller)
    config = _f1_config(control_period_s=0.02)
    env = F1RobotEnv(config)
    action = np.array(
        [0.01, -0.01, 0.005, 0.0, 0.5, -0.5, 0.75]
        + [-0.01, 0.01, -0.005, 0.0, -0.5, 0.5, -0.75],
        dtype=np.float64,
    )
    expected_policy_action = action.astype(np.float32)
    try:
        observation, reward, terminated, truncated, info = env.step(action)
        action.fill(0.0)

        controller_config = factory_call["config"]
        assert controller_config["control_period_s"] == 0.02
        assert controller_config["max_observation_age_s"] == 0.25
        assert controller_config["max_observation_skew_s"] == 0.05
        assert len(controller.policy_commands) == 1
        assert controller.reset_commands == []
        command = controller.policy_commands[0]
        assert isinstance(command, DualArmTcpCommand)
        np.testing.assert_allclose(
            command.left_tcp_target_m_deg,
            np.array([0.31, 0.19, 0.405, 0.0, 0.5, -0.5]),
        )
        np.testing.assert_allclose(
            command.right_tcp_target_m_deg,
            np.array([0.29, -0.19, 0.395, 0.0, -0.5, 0.5]),
        )
        assert command.left_gripper_target == pytest.approx(50.75)
        assert command.right_gripper_target == pytest.approx(49.25)
        assert command.duration_s == 0.02
        np.testing.assert_array_equal(info["policy_action"], expected_policy_action)
        assert info["policy_action"].dtype == np.float32
        assert info["command_id"] == command.command_id
        assert info["command_status"] is CommandStatus.FINISHED_DISPATCH
        np.testing.assert_array_equal(
            info["absolute_left_tcp_target_m_deg"],
            command.left_tcp_target_m_deg,
        )
        np.testing.assert_array_equal(
            info["absolute_right_tcp_target_m_deg"],
            command.right_tcp_target_m_deg,
        )
        assert info["absolute_left_gripper_target"] == command.left_gripper_target
        assert info["absolute_right_gripper_target"] == command.right_gripper_target
        assert info["command_created_at_monotonic_s"] == command.created_at_monotonic_s
        assert (
            info["command_accepted_at_monotonic_s"]
            == controller.last_receipt.accepted_at_monotonic_s
        )
        assert (
            info["command_expires_at_monotonic_s"]
            == controller.last_receipt.expires_at_monotonic_s
        )
        info["absolute_left_tcp_target_m_deg"].fill(99.0)
        assert not np.any(command.left_tcp_target_m_deg == 99.0)
        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert env.observation_space.contains(observation)
        assert controller.last_receipt is not None
        assert controller.read_requests[-1]["newer_than"] is None
        assert controller.submitted_at_s is not None
        assert controller.post_submit_read_at_s is not None
        assert controller.post_submit_read_at_s - controller.submitted_at_s < 0.015
    finally:
        env.close()


def test_boundary_policy_action_is_submitted_unchanged_for_controller_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RLinf must not clip unsafe targets before Controller fail-closed checks."""

    observation = _robot_observation(
        left_tcp=np.array([0.999, 0.2, 0.4, 0.0, 0.0, 0.0], dtype=np.float64),
        left_gripper_percent_closed=99.5,
    )
    controller = RecordingController(
        observation=observation,
        policy_status=CommandStatus.REJECTED,
    )
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(control_period_s=0.001))
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 7)
    try:
        with pytest.raises(ControllerError, match="rejected"):
            env.step(action)

        assert len(controller.policy_commands) == 1
        command = controller.policy_commands[0]
        np.testing.assert_allclose(
            command.left_tcp_target_m_deg,
            np.array([1.999, 0.2, 0.4, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        assert command.left_gripper_target == pytest.approx(100.5)
        assert env._num_steps == 0
        assert controller.stop_reasons
    finally:
        env.close()


def test_step_waits_for_finished_dispatch_before_returning_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController(policy_status=CommandStatus.ACCEPTED)
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(control_period_s=0.001))
    try:
        controller.status_sequences[0] = [
            CommandStatus.ACCEPTED,
            CommandStatus.DISPATCHED,
            CommandStatus.FINISHED_DISPATCH,
        ]

        _, _, _, _, info = env.step(np.zeros(14, dtype=np.float32))

        assert [
            event for event in controller.events if event == ("get_command_status", 0)
        ] == [
            ("get_command_status", 0),
            ("get_command_status", 0),
            ("get_command_status", 0),
        ]
        assert info["command_status"] is CommandStatus.FINISHED_DISPATCH
        assert info["dispatch_latency_s"] >= 0.0
        assert controller.stop_reasons == []
    finally:
        env.close()


def test_step_times_out_if_dispatch_never_finishes_and_returns_no_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController(policy_status=CommandStatus.ACCEPTED)
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(control_period_s=0.001))
    try:
        with pytest.raises(TimeoutError, match="policy command 0"):
            env.step(np.zeros(14, dtype=np.float32))

        assert len(controller.policy_commands) == 1
        assert controller.stop_reasons
    finally:
        env.close()


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(13, dtype=np.float32),
        np.array(["bad"] * 14, dtype=object),
        np.full(14, np.nan, dtype=np.float32),
        np.array([1.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 2, dtype=np.float32),
    ],
    ids=["wrong-shape", "non-numeric", "non-finite", "outside-envelope-delta"],
)
def test_step_rejects_invalid_policy_actions_before_submission(
    monkeypatch: pytest.MonkeyPatch,
    action: np.ndarray,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config())
    try:
        with pytest.raises(ValueError, match="action"):
            env.step(action)
        assert controller.policy_commands == []
    finally:
        env.close()


def test_step_initial_observation_failure_stops_without_masking_or_submitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    stale_error = ObservationUnavailableError("initial observation is stale")
    controller.stop_error = ControllerLifecycleError("stop cleanup failed")
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config())
    controller.read_error = stale_error
    try:
        with pytest.raises(ObservationUnavailableError) as captured:
            env.step(np.zeros(14, dtype=np.float32))

        assert captured.value is stale_error
        assert len(controller.stop_reasons) == 1
        assert controller.policy_commands == []
        assert controller.reset_commands == []
        assert env._num_steps == 0
    finally:
        env.close()


def test_step_returns_copies_of_controller_observation_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(control_period_s=0.001))
    try:
        observation, *_ = env.step(np.zeros(14, dtype=np.float32))
        controller_observation = controller.observation

        observation["state"]["proprioception"].fill(99.0)
        observation["frames"]["head_color"].fill(255)

        assert not np.any(controller_observation.left_joint_position_rad == 99.0)
        assert not np.any(controller_observation.head_color_rgb == 255)
    finally:
        env.close()


def test_step_checks_rate_limit_before_command_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    clock_values = iter([10.0, 20.0, 20.001, 20.002, 20.003])
    clock_calls: list[float] = []

    def fake_monotonic() -> float:
        value = next(clock_values)
        clock_calls.append(value)
        return value

    original_submit = controller.submit_command
    clock_call_count_at_submit: list[int] = []

    def recording_submit(command: DualArmTcpCommand) -> CommandReceipt:
        clock_call_count_at_submit.append(len(clock_calls))
        return original_submit(command)

    monkeypatch.setattr(F1_ENV_MODULE, "monotonic", fake_monotonic)
    monkeypatch.setattr(controller, "submit_command", recording_submit)
    env = F1RobotEnv(_f1_config(control_period_s=0.02))
    try:
        env.step(np.zeros(14, dtype=np.float32))

        assert clock_call_count_at_submit == [2]
    finally:
        env.close()


@pytest.mark.parametrize(
    "status",
    [
        CommandStatus.REJECTED,
        CommandStatus.EXPIRED,
        CommandStatus.CANCELLED,
        CommandStatus.FAULT,
    ],
)
def test_step_stops_motion_and_raises_for_failed_command_status(
    monkeypatch: pytest.MonkeyPatch,
    status: CommandStatus,
) -> None:
    controller = RecordingController(policy_status=status)
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(control_period_s=0.001))
    try:
        with pytest.raises(ControllerError, match=status.value):
            env.step(np.zeros(14, dtype=np.float32))

        assert len(controller.stop_reasons) == 1
        assert env._num_steps == 0
    finally:
        env.close()


def test_step_surfaces_controller_fault_reason_for_failed_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController(
        policy_status=CommandStatus.FAULT,
        health=ControllerHealth(
            ready=False,
            faulted=True,
            reason="command 0 sink failed: right_gripper publish rejected",
            checked_at_monotonic_s=1.0,
        ),
    )
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(control_period_s=0.001))
    try:
        with pytest.raises(
            ControllerError,
            match="right_gripper publish rejected",
        ):
            env.step(np.zeros(14, dtype=np.float32))

        assert len(controller.stop_reasons) == 1
        assert env._num_steps == 0
    finally:
        env.close()


@pytest.mark.parametrize(
    "health",
    [
        ControllerHealth(
            ready=False,
            faulted=False,
            reason="not ready",
            checked_at_monotonic_s=1.0,
        ),
        ControllerHealth(
            ready=True,
            faulted=True,
            reason="executor fault",
            checked_at_monotonic_s=1.0,
        ),
    ],
)
def test_step_stops_motion_and_raises_for_unhealthy_controller(
    monkeypatch: pytest.MonkeyPatch,
    health: ControllerHealth,
) -> None:
    controller = RecordingController(health=health)
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(control_period_s=0.001))
    try:
        with pytest.raises(ControllerError, match=health.reason):
            env.step(np.zeros(14, dtype=np.float32))

        assert len(controller.stop_reasons) == 1
        assert env._num_steps == 0
    finally:
        env.close()


def test_step_uses_unique_command_ids_and_truncates_at_the_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(control_period_s=0.001, max_num_steps=4))
    try:
        truncations = [env.step(np.zeros(14, dtype=np.float32))[3] for _ in range(4)]

        assert truncations == [False, False, False, True]
        assert [command.command_id for command in controller.policy_commands] == [
            0,
            1,
            2,
            3,
        ]
    finally:
        env.close()


def test_reset_uses_current_measured_state_as_session_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_observation = _robot_observation(
        left_tcp=np.array([0.32, 0.21, 0.43, 1.0, 2.0, 3.0]),
        left_gripper_percent_closed=12.0,
        right_tcp=np.array([0.34, -0.22, 0.44, -1.0, -2.0, -3.0]),
        right_gripper_percent_closed=88.0,
    )
    controller = RecordingController(observation=initial_observation)
    _install_recording_controller(monkeypatch, controller)
    config = _f1_config(
        control_period_s=0.001,
        reset_duration_s=0.004,
        reset_timeout_s=0.05,
        tcp_position_tolerance_m=0.002,
        tcp_orientation_tolerance_deg=0.5,
        gripper_tolerance_percent_closed=0.25,
    )
    env = F1RobotEnv(config)
    controller.observation = _robot_observation(
        left_tcp=np.array([0.5, 0.3, 0.5, 10.0, 20.0, 30.0]),
        left_gripper_percent_closed=50.0,
        right_tcp=np.array([0.5, -0.3, 0.5, -10.0, -20.0, -30.0]),
        right_gripper_percent_closed=50.0,
    )
    try:
        controller.events.clear()
        observation, info = env.reset(seed=7, options={"operator": "ready"})

        assert controller.policy_commands == []
        assert controller.reset_commands == []
        assert controller.last_receipt is None
        assert controller.read_requests[-1]["newer_than"] is None
        assert env.observation_space.contains(observation)
        assert info["reset_mode"] == "current_state_origin"
        assert env.task_reset_options == {"operator": "ready"}
        assert info["session_origin_state"].shape == (16,)
        np.testing.assert_allclose(
            info["session_origin_state"][:8],
            observation["state"]["proprioception"][:8],
        )
        assert [event[0] for event in controller.events] == [
            "open",
            "wait_ready",
            "health",
            "reset_task",
            "read_observation",
        ]
    finally:
        env.close()


def test_reset_is_not_a_transition_resets_horizon_and_keeps_ids_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(
        _f1_config(
            control_period_s=0.001,
            max_num_steps=2,
            reset_timeout_s=0.05,
        )
    )
    try:
        assert env.step(np.zeros(14, dtype=np.float32))[3] is False
        assert env.step(np.zeros(14, dtype=np.float32))[3] is True

        reset_result = env.reset()
        first_step_after_reset = env.step(np.zeros(14, dtype=np.float32))

        assert len(reset_result) == 2
        assert first_step_after_reset[3] is False
        assert controller.reset_commands == []
        assert [command.command_id for command in controller.policy_commands] == [
            0,
            1,
            2,
        ]
    finally:
        env.close()


def test_reset_does_not_wait_for_motion_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController(converge_reset=False)
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(
        _f1_config(
            control_period_s=0.001,
            reset_timeout_s=0.01,
        )
    )
    controller.observation = _robot_observation(
        left_tcp=np.array([0.5, 0.3, 0.5, 10.0, 20.0, 30.0]),
        right_tcp=np.array([0.5, -0.3, 0.5, -10.0, -20.0, -30.0]),
    )
    try:
        observation, info = env.reset()
        assert env.observation_space.contains(observation)
        assert info["reset_mode"] == "current_state_origin"
        assert controller.reset_commands == []
        assert controller.policy_commands == []
        assert controller.stop_reasons == []
    finally:
        env.close()


def test_reset_ignores_stale_reset_command_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController(reset_status=CommandStatus.REJECTED)
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(reset_timeout_s=0.05))
    try:
        observation, info = env.reset()
        assert env.observation_space.contains(observation)
        assert info["reset_mode"] == "current_state_origin"
        assert controller.stop_reasons == []
        assert controller.reset_commands == []
    finally:
        env.close()


def test_reset_reopens_stopped_controller_without_repeating_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(reset_timeout_s=0.05))
    controller.health_error_once = ControllerNotReadyError("controller is stopped")
    controller.events.clear()
    try:
        with pytest.raises(ControllerNotReadyError, match="controller is stopped"):
            env.reset()

        assert controller.stop_reasons == [
            "reset failed: controller is stopped",
        ]
        assert [event[0] for event in controller.events] == [
            "open",
            "wait_ready",
            "health",
            "stop_command_dispatch",
        ]
    finally:
        env.close()


def test_reset_does_not_treat_other_health_errors_as_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(reset_timeout_s=0.05))
    health_error = ControllerLifecycleError("health probe failed")
    controller.health_error_once = health_error
    controller.events.clear()
    try:
        with pytest.raises(ControllerLifecycleError) as captured:
            env.reset()

        assert captured.value is health_error
        assert controller.stop_reasons == ["reset failed: health probe failed"]
        assert [event[0] for event in controller.events] == [
            "open",
            "wait_ready",
            "health",
            "stop_command_dispatch",
        ]
        assert controller.reset_commands == []
    finally:
        env.close()


def test_reset_does_not_stop_previous_motion_before_reading_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(reset_timeout_s=0.05))
    stop_error = ControllerLifecycleError("stop cleanup failed")
    controller.stop_error = stop_error
    controller.events.clear()
    try:
        observation, info = env.reset()
        assert env.observation_space.contains(observation)
        assert info["reset_mode"] == "current_state_origin"
        assert controller.stop_reasons == []
        assert [event[0] for event in controller.events] == [
            "open",
            "wait_ready",
            "health",
            "reset_task",
            "read_observation",
        ]
        assert controller.reset_commands == []
    finally:
        env.close()


def test_reset_unexpected_exception_stops_reset_motion_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(reset_timeout_s=0.05))
    controller.read_error = RuntimeError("read failed")
    try:
        with pytest.raises(RuntimeError, match="read failed"):
            env.reset()

        assert len(controller.stop_reasons) == 1
        assert controller.reset_commands == []
    finally:
        env.close()


def test_close_is_idempotent_and_constructor_failure_closes_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config())

    env.close()
    env.close()

    assert controller.close_calls == 1

    failed_controller = RecordingController(
        ready_error=ControllerNotReadyError("not ready")
    )
    _install_recording_controller(monkeypatch, failed_controller)
    with pytest.raises(ControllerNotReadyError, match="not ready"):
        F1RobotEnv(_f1_config())
    assert failed_controller.close_calls == 1


def test_installed_fake_reset_and_step_use_fresh_receipts_without_ros_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ros_modules_before = {
        name
        for name in sys.modules
        if name == "rclpy"
        or name.startswith("rclpy.")
        or name == "cv_bridge"
        or name.startswith("cv_bridge.")
    }
    env = F1RobotEnv(
        _f1_config(
            control_period_s=0.001,
            reset_timeout_s=0.05,
        )
    )
    controller = env._active_controller
    original_read = controller.read_observation
    newer_than_values: list[float | None] = []

    def recording_read_observation(
        *,
        max_age_s: float,
        max_skew_s: float,
        newer_than: float | None = None,
    ) -> Any:
        newer_than_values.append(newer_than)
        return original_read(
            max_age_s=max_age_s,
            max_skew_s=max_skew_s,
            newer_than=newer_than,
        )

    monkeypatch.setattr(controller, "read_observation", recording_read_observation)
    try:
        reset_observation, reset_info = env.reset()
        action = np.array(
            [0.0025, 0.0025, 0.0025, 0.25, 0.25, 0.25, 0.0] * 2,
            dtype=np.float32,
        )
        step_observation, reward, terminated, truncated, step_info = env.step(action)

        assert env.observation_space.contains(reset_observation)
        assert env.observation_space.contains(step_observation)
        assert reset_info["reset_mode"] == "current_state_origin"
        assert reset_info["session_origin_state"].shape == (16,)
        assert step_info["command_status"] is CommandStatus.FINISHED_DISPATCH
        assert step_info["command_id"] == 0
        assert newer_than_values[0] is None
        assert newer_than_values[-1] is None
        assert (
            step_info["command_created_at_monotonic_s"]
            <= step_info["command_accepted_at_monotonic_s"]
            < step_info["command_expires_at_monotonic_s"]
        )
        assert "absolute_left_tcp_target_m_deg" in step_info
        assert "absolute_right_tcp_target_m_deg" in step_info
        assert reward == 0.0
        assert not terminated
        assert not truncated
    finally:
        env.close()

    ros_modules_after = {
        name
        for name in sys.modules
        if name == "rclpy"
        or name.startswith("rclpy.")
        or name == "cv_bridge"
        or name.startswith("cv_bridge.")
    }
    assert ros_modules_after == ros_modules_before


def test_step_waits_for_a_fresh_post_action_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(
        _f1_config(control_period_s=0.001, post_action_observation_timeout_s=0.05)
    )
    original_read = controller.read_observation
    fresh_attempts = 0
    read_calls = 0

    def temporarily_stale_read(
        *,
        max_age_s: float,
        max_skew_s: float,
        newer_than: float | None = None,
    ) -> Any:
        nonlocal fresh_attempts, read_calls
        read_calls += 1
        if read_calls >= 2:
            fresh_attempts += 1
            if fresh_attempts == 1:
                raise ObservationUnavailableError("synthetic camera delivery gap")
        observation = original_read(
            max_age_s=max_age_s,
            max_skew_s=max_skew_s,
            newer_than=newer_than,
        )
        if fresh_attempts >= 2:
            observation.timestamps = SensorTimestamps(
                source_timestamp_s=observation.timestamps.source_timestamp_s,
                received_at_monotonic_s=dict.fromkeys(SENSOR_NAMES, 1.0e20),
            )
        return observation

    monkeypatch.setattr(controller, "read_observation", temporarily_stale_read)
    try:
        env.step(np.zeros(14, dtype=np.float32))
        assert fresh_attempts == 2
    finally:
        env.close()


def test_post_action_wait_uses_the_configured_policy_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch right-wrist policies still synchronizing against the head camera."""

    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(
        _f1_config(
            post_action_observation_timeout_s=0.01,
            post_action_image_source="right_wrist_color",
        )
    )
    observation = SimpleNamespace(
        timestamps=SimpleNamespace(
            received_at_monotonic_s={
                "head_color": 0.0,
                "right_wrist_color": 6.0,
            }
        )
    )
    monkeypatch.setattr(env, "_read_observation", lambda: observation)
    try:
        assert env._wait_for_post_action_observation(newer_than=5.0) is observation
    finally:
        env.close()


def test_step_observation_window_starts_after_command_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(control_period_s=0.1))
    thresholds: list[float] = []
    original_wait = env._wait_for_post_action_observation

    def recording_wait(*, newer_than: float) -> Any:
        thresholds.append(newer_than)
        return original_wait(newer_than=newer_than)

    monkeypatch.setattr(env, "_wait_for_post_action_observation", recording_wait)
    try:
        env.step(np.zeros(14, dtype=np.float32))

        assert controller.last_receipt is not None
        assert thresholds == [controller.last_receipt.accepted_at_monotonic_s + 0.1]
    finally:
        env.close()


def test_consecutive_policy_commands_are_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(_f1_config(control_period_s=0.02))
    try:
        env.step(np.zeros(14, dtype=np.float32))
        first_submitted_at = controller.submitted_at_s
        env.step(np.zeros(14, dtype=np.float32))

        assert first_submitted_at is not None
        assert controller.submitted_at_s is not None
        assert controller.submitted_at_s - first_submitted_at >= 0.015
    finally:
        env.close()


def test_installed_fake_recovers_from_step_read_failure_through_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = F1RobotEnv(
        _f1_config(
            control_period_s=0.001,
            reset_timeout_s=0.05,
        )
    )
    controller = env._active_controller
    original_read = controller.read_observation
    stale_error = ObservationUnavailableError("synthetic stale observation")
    fail_next_read = True

    def one_shot_failed_read(
        *,
        max_age_s: float,
        max_skew_s: float,
        newer_than: float | None = None,
    ) -> Any:
        nonlocal fail_next_read
        if fail_next_read:
            fail_next_read = False
            raise stale_error
        return original_read(
            max_age_s=max_age_s,
            max_skew_s=max_skew_s,
            newer_than=newer_than,
        )

    monkeypatch.setattr(controller, "read_observation", one_shot_failed_read)
    try:
        with pytest.raises(ObservationUnavailableError) as captured:
            env.step(np.zeros(14, dtype=np.float32))
        assert captured.value is stale_error
        with pytest.raises(ControllerNotReadyError):
            controller.health()

        reset_observation, reset_info = env.reset()
        step_result = env.step(np.zeros(14, dtype=np.float32))

        assert env.observation_space.contains(reset_observation)
        assert reset_info["reset_mode"] == "current_state_origin"
        assert reset_info["session_origin_state"].shape == (16,)
        assert step_result[4]["command_id"] == 0
        assert step_result[4]["command_status"] is CommandStatus.FINISHED_DISPATCH
        health = controller.health()
        assert health.ready
        assert not health.faulted
    finally:
        env.close()
