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
import sys
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from types import ModuleType
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from f1_robot_controller import (
    BackendUnavailableError,
    CommandReceipt,
    CommandStatus,
    ControllerHealth,
    ControllerNotReadyError,
    DualArmCommand,
    DualArmResetCommand,
    F1RobotController,
    RobotObservation,
    SensorTimestamps,
)

ROOT = Path(__file__).resolve().parents[3]
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
F1RobotEnv = F1_PACKAGE.F1RobotEnv
F1_ENV_MODULE = sys.modules["_f1_env_under_test.f1_robot_env"]

SENSOR_NAMES = (
    "left_joint_position",
    "left_gripper_position",
    "right_joint_position",
    "right_gripper_position",
    "head_color",
    "left_wrist_color",
    "right_wrist_color",
)


def _robot_observation(
    *,
    left_joints: np.ndarray | None = None,
    left_gripper: float = 0.5,
    right_joints: np.ndarray | None = None,
    right_gripper: float = 0.5,
) -> RobotObservation:
    if left_joints is None:
        left_joints = np.linspace(0.1, 0.7, 7)
    if right_joints is None:
        right_joints = np.linspace(-0.1, -0.7, 7)
    return RobotObservation(
        left_joint_position_rad=np.array(left_joints, dtype=np.float64, copy=True),
        left_gripper_position=left_gripper,
        right_joint_position_rad=np.array(right_joints, dtype=np.float64, copy=True),
        right_gripper_position=right_gripper,
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
        observation: RobotObservation | None = None,
        *,
        converge_reset: bool = True,
        ready_error: Exception | None = None,
    ) -> None:
        self.observation = observation or _robot_observation()
        self.converge_reset = converge_reset
        self.ready_error = ready_error
        self.policy_commands: list[DualArmCommand] = []
        self.reset_commands: list[DualArmResetCommand] = []
        self.statuses: dict[int, CommandStatus] = {}
        self.read_requests: list[dict[str, float | None]] = []
        self.close_calls = 0
        self.open_calls = 0
        self.wait_ready_calls: list[float] = []
        self.last_receipt: CommandReceipt | None = None
        self.submitted_at_s: float | None = None
        self.post_submit_read_at_s: float | None = None

    def open(self) -> None:
        self.open_calls += 1

    def wait_ready(self, timeout_s: float) -> None:
        self.wait_ready_calls.append(timeout_s)
        if self.ready_error is not None:
            raise self.ready_error

    def read_observation(
        self,
        *,
        max_age_s: float,
        max_skew_s: float,
        newer_than: float | None = None,
    ) -> RobotObservation:
        self.read_requests.append(
            {
                "max_age_s": max_age_s,
                "max_skew_s": max_skew_s,
                "newer_than": newer_than,
            }
        )
        if newer_than is not None:
            self.post_submit_read_at_s = monotonic()
        return self.observation

    def submit_command(self, command: DualArmCommand) -> CommandReceipt:
        self.policy_commands.append(command)
        self.submitted_at_s = monotonic()
        self.observation = _robot_observation(
            left_joints=command.left_joint_target_rad,
            left_gripper=command.left_gripper_target,
            right_joints=command.right_joint_target_rad,
            right_gripper=command.right_gripper_target,
        )
        receipt = CommandReceipt(
            command_id=command.command_id,
            accepted_at_monotonic_s=100.0 + command.command_id,
            expires_at_monotonic_s=101.0 + command.command_id,
        )
        self.last_receipt = receipt
        self.statuses[command.command_id] = CommandStatus.FINISHED_DISPATCH
        return receipt

    def submit_reset_command(self, command: DualArmResetCommand) -> CommandReceipt:
        self.reset_commands.append(command)
        if self.converge_reset:
            self.observation = _robot_observation(
                left_joints=command.left_joint_target_rad,
                left_gripper=command.left_gripper_target,
                right_joints=command.right_joint_target_rad,
                right_gripper=command.right_gripper_target,
            )
        receipt = CommandReceipt(
            command_id=command.command_id,
            accepted_at_monotonic_s=100.0 + command.command_id,
            expires_at_monotonic_s=101.0 + command.command_id,
        )
        self.last_receipt = receipt
        self.statuses[command.command_id] = CommandStatus.FINISHED_DISPATCH
        return receipt

    def get_command_status(self, command_id: int) -> CommandStatus:
        return self.statuses[command_id]

    def health(self) -> ControllerHealth:
        return ControllerHealth(
            ready=True,
            faulted=False,
            reason=None,
            checked_at_monotonic_s=1.0,
        )

    def stop_experiment_motion(self, reason: str) -> None:
        del reason

    def close(self) -> None:
        self.close_calls += 1


def _install_recording_controller(
    monkeypatch: pytest.MonkeyPatch,
    controller: RecordingController,
) -> dict[str, Any]:
    factory_call: dict[str, Any] = {}

    def factory(*, is_dummy: bool, config: object) -> RecordingController:
        factory_call.update(is_dummy=is_dummy, config=config)
        return controller

    monkeypatch.setattr(F1_ENV_MODULE, "create_controller", factory)
    return factory_call


def test_f1_robot_config_has_the_frozen_phase_one_defaults() -> None:
    config = F1RobotConfig()

    assert asdict(config) == {
        "is_dummy": True,
        "control_period_s": 0.01,
        "max_num_steps": 4,
        "max_observation_age_s": 0.25,
        "max_observation_skew_s": 0.05,
        "max_joint_delta_rad": (0.01,) * 14,
        "max_gripper_delta": 0.05,
        "reset_joint_target_rad": (0.0,) * 14,
        "reset_gripper_target": (0.5, 0.5),
        "reset_duration_s": 0.01,
        "reset_timeout_s": 1.0,
        "reset_tolerance_rad": 0.01,
    }


def test_f1_robot_env_exposes_16d_action_state_and_three_rgb_frames() -> None:
    env = F1RobotEnv(F1RobotConfig())
    try:
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (16,)
        assert env.action_space.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(env.action_space.low, np.full(16, -1.0))
        np.testing.assert_array_equal(env.action_space.high, np.full(16, 1.0))

        state_space = env.observation_space["state"]
        assert list(state_space.spaces) == [
            "left_gripper",
            "left_joint_position",
            "right_gripper",
            "right_joint_position",
        ]
        assert sum(space.shape[0] for space in state_space.spaces.values()) == 16
        assert state_space["left_joint_position"].shape == (7,)
        assert state_space["left_gripper"].shape == (1,)
        assert state_space["right_joint_position"].shape == (7,)
        assert state_space["right_gripper"].shape == (1,)

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


def test_real_backend_unavailability_propagates_the_controller_error() -> None:
    with pytest.raises(BackendUnavailableError):
        F1RobotEnv(F1RobotConfig(is_dummy=False))


def test_step_submits_one_scaled_absolute_command_and_preserves_policy_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    factory_call = _install_recording_controller(monkeypatch, controller)
    config = F1RobotConfig(control_period_s=0.02)
    env = F1RobotEnv(config)
    action = np.array(
        [1.0, -1.0, 0.5, 0.0, 0.25, -0.25, 0.75, 1.0]
        + [-1.0, 1.0, -0.5, 0.0, -0.25, 0.25, -0.75, -1.0],
        dtype=np.float64,
    )
    expected_policy_action = action.astype(np.float32)
    try:
        observation, reward, terminated, truncated, info = env.step(action)
        action.fill(0.0)

        assert factory_call["is_dummy"] is True
        controller_config = factory_call["config"]
        assert controller_config.control_period_s == 0.02
        assert controller_config.max_observation_age_s == 0.25
        assert controller_config.max_observation_skew_s == 0.05
        assert len(controller.policy_commands) == 1
        assert controller.reset_commands == []
        command = controller.policy_commands[0]
        np.testing.assert_allclose(
            command.left_joint_target_rad,
            np.array([0.11, 0.19, 0.305, 0.4, 0.5025, 0.5975, 0.7075]),
        )
        np.testing.assert_allclose(
            command.right_joint_target_rad,
            np.array([-0.11, -0.19, -0.305, -0.4, -0.5025, -0.5975, -0.7075]),
        )
        assert command.left_gripper_target == pytest.approx(0.55)
        assert command.right_gripper_target == pytest.approx(0.45)
        assert command.duration_s == 0.02
        np.testing.assert_array_equal(info["policy_action"], expected_policy_action)
        assert info["policy_action"].dtype == np.float32
        assert info["command_id"] == command.command_id
        assert info["command_status"] is CommandStatus.FINISHED_DISPATCH
        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert env.observation_space.contains(observation)
        assert controller.last_receipt is not None
        assert (
            controller.read_requests[-1]["newer_than"]
            == controller.last_receipt.accepted_at_monotonic_s
        )
        assert controller.submitted_at_s is not None
        assert controller.post_submit_read_at_s is not None
        assert controller.post_submit_read_at_s - controller.submitted_at_s >= 0.015
    finally:
        env.close()


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(15, dtype=np.float32),
        np.array(["bad"] * 16, dtype=object),
        np.full(16, np.nan, dtype=np.float32),
        np.full(16, 1.01, dtype=np.float32),
    ],
    ids=["wrong-shape", "non-numeric", "non-finite", "not-normalized"],
)
def test_step_rejects_invalid_policy_actions_before_submission(
    monkeypatch: pytest.MonkeyPatch,
    action: np.ndarray,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(F1RobotConfig())
    try:
        with pytest.raises(ValueError, match="action"):
            env.step(action)
        assert controller.policy_commands == []
    finally:
        env.close()


def test_step_returns_copies_of_controller_observation_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(F1RobotConfig(control_period_s=0.001))
    try:
        observation, *_ = env.step(np.zeros(16, dtype=np.float32))
        controller_observation = controller.observation

        observation["state"]["left_joint_position"].fill(99.0)
        observation["frames"]["head_color"].fill(255)

        assert not np.any(controller_observation.left_joint_position_rad == 99.0)
        assert not np.any(controller_observation.head_color_rgb == 255)
    finally:
        env.close()


def test_step_starts_the_control_period_after_command_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    clock_values = iter([10.0, 20.0, 20.001])
    clock_calls: list[float] = []

    def fake_monotonic() -> float:
        value = next(clock_values)
        clock_calls.append(value)
        return value

    original_submit = controller.submit_command
    clock_call_count_at_submit: list[int] = []

    def recording_submit(command: DualArmCommand) -> CommandReceipt:
        clock_call_count_at_submit.append(len(clock_calls))
        return original_submit(command)

    monkeypatch.setattr(F1_ENV_MODULE, "monotonic", fake_monotonic)
    monkeypatch.setattr(controller, "submit_command", recording_submit)
    env = F1RobotEnv(F1RobotConfig(control_period_s=0.02))
    try:
        env.step(np.zeros(16, dtype=np.float32))

        assert clock_call_count_at_submit == [1]
    finally:
        env.close()


def test_step_uses_unique_command_ids_and_truncates_at_the_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(F1RobotConfig(control_period_s=0.001, max_num_steps=4))
    try:
        truncations = [env.step(np.zeros(16, dtype=np.float32))[3] for _ in range(4)]

        assert truncations == [False, False, False, True]
        assert [command.command_id for command in controller.policy_commands] == [
            0,
            1,
            2,
            3,
        ]
    finally:
        env.close()


def test_reset_submits_a_separate_command_and_verifies_measured_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    joint_target = tuple(np.linspace(-0.3, 0.3, 14))
    config = F1RobotConfig(
        control_period_s=0.001,
        reset_joint_target_rad=joint_target,
        reset_gripper_target=(0.2, 0.8),
        reset_duration_s=0.004,
        reset_timeout_s=0.05,
        reset_tolerance_rad=0.002,
    )
    env = F1RobotEnv(config)
    try:
        observation, info = env.reset(seed=7)

        assert controller.policy_commands == []
        assert len(controller.reset_commands) == 1
        command = controller.reset_commands[0]
        assert isinstance(command, DualArmResetCommand)
        np.testing.assert_allclose(command.left_joint_target_rad, joint_target[:7])
        np.testing.assert_allclose(command.right_joint_target_rad, joint_target[7:])
        assert command.left_gripper_target == 0.2
        assert command.right_gripper_target == 0.8
        assert command.duration_s == 0.004
        assert command.timeout_s == 0.05
        assert command.tolerance_rad == 0.002
        assert controller.last_receipt is not None
        assert (
            controller.read_requests[-1]["newer_than"]
            == controller.last_receipt.accepted_at_monotonic_s
        )
        np.testing.assert_allclose(
            observation["state"]["left_joint_position"], joint_target[:7]
        )
        np.testing.assert_allclose(
            observation["state"]["right_joint_position"], joint_target[7:]
        )
        np.testing.assert_allclose(observation["state"]["left_gripper"], [0.2])
        np.testing.assert_allclose(observation["state"]["right_gripper"], [0.8])
        assert env.observation_space.contains(observation)
        assert info == {
            "reset_command_id": command.command_id,
            "command_status": CommandStatus.FINISHED_DISPATCH,
        }
    finally:
        env.close()


def test_reset_is_not_a_transition_resets_horizon_and_keeps_ids_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(
        F1RobotConfig(
            control_period_s=0.001,
            max_num_steps=2,
            reset_timeout_s=0.05,
        )
    )
    try:
        assert env.step(np.zeros(16, dtype=np.float32))[3] is False
        assert env.step(np.zeros(16, dtype=np.float32))[3] is True

        reset_result = env.reset()
        first_step_after_reset = env.step(np.zeros(16, dtype=np.float32))

        assert len(reset_result) == 2
        assert first_step_after_reset[3] is False
        ids = [command.command_id for command in controller.policy_commands]
        ids += [controller.reset_commands[0].command_id]
        assert sorted(ids) == [0, 1, 2, 3]
    finally:
        env.close()


def test_reset_has_a_bounded_monotonic_deadline_when_state_does_not_converge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController(converge_reset=False)
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(
        F1RobotConfig(
            control_period_s=0.001,
            reset_joint_target_rad=(0.9,) * 14,
            reset_timeout_s=0.01,
        )
    )
    started_at_s = monotonic()
    try:
        with pytest.raises(TimeoutError, match="reset"):
            env.reset()
        assert monotonic() - started_at_s < 0.5
        assert len(controller.reset_commands) == 1
        assert controller.policy_commands == []
    finally:
        env.close()


def test_close_is_idempotent_and_constructor_failure_closes_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(F1RobotConfig())

    env.close()
    env.close()

    assert controller.close_calls == 1

    failed_controller = RecordingController(
        ready_error=ControllerNotReadyError("not ready")
    )
    _install_recording_controller(monkeypatch, failed_controller)
    with pytest.raises(ControllerNotReadyError, match="not ready"):
        F1RobotEnv(F1RobotConfig())
    assert failed_controller.close_calls == 1
