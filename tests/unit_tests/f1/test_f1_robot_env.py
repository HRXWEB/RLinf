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
from types import ModuleType, SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from f1_robot_controller import (
    BackendUnavailableError,
    CommandReceipt,
    CommandStatus,
    ControllerError,
    ControllerHealth,
    ControllerLifecycleError,
    ControllerNotReadyError,
    DualArmCommand,
    DualArmResetCommand,
    F1RobotController,
    ObservationUnavailableError,
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
        self.policy_commands: list[DualArmCommand] = []
        self.reset_commands: list[DualArmResetCommand] = []
        self.statuses: dict[int, CommandStatus] = {}
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
    ) -> RobotObservation:
        self.events.append(("read_observation", newer_than))
        self.read_requests.append(
            {
                "max_age_s": max_age_s,
                "max_skew_s": max_skew_s,
                "newer_than": newer_than,
            }
        )
        if newer_than is not None:
            self.post_submit_read_at_s = monotonic()
        if self.read_error is not None:
            raise self.read_error
        return self.observation

    def submit_command(self, command: DualArmCommand) -> CommandReceipt:
        self.events.append(("submit_command", command.command_id))
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
        self.statuses[command.command_id] = self.policy_status
        return receipt

    def submit_reset_command(self, command: DualArmResetCommand) -> CommandReceipt:
        self.events.append(("submit_reset_command", command.command_id))
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
        self.statuses[command.command_id] = self.reset_status
        return receipt

    def get_command_status(self, command_id: int) -> CommandStatus:
        self.events.append(("get_command_status", command_id))
        return self.statuses[command_id]

    def health(self) -> ControllerHealth:
        self.events.append(("health", None))
        if self.health_error_once is not None:
            error = self.health_error_once
            self.health_error_once = None
            raise error
        return self.controller_health

    def stop_experiment_motion(self, reason: str) -> None:
        self.events.append(("stop_experiment_motion", reason))
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

    def factory(*, is_dummy: bool, config: object) -> RecordingController:
        factory_call.update(is_dummy=is_dummy, config=config)
        return controller

    monkeypatch.setattr(F1_ENV_MODULE, "create_controller", factory)
    return factory_call


def _load_realworld_env_class(
    monkeypatch: pytest.MonkeyPatch,
) -> type:
    """Load RealWorldEnv without importing RLinf's unrelated heavy stack."""

    torch_stub = ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    psutil_stub = ModuleType("psutil")
    psutil_stub.process_iter = lambda: []
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)

    filelock_stub = ModuleType("filelock")
    filelock_stub.FileLock = type("FileLock", (), {})
    monkeypatch.setitem(sys.modules, "filelock", filelock_stub)

    class OmegaConfStub:
        @staticmethod
        def create(value: object) -> object:
            return value

        @staticmethod
        def to_container(value: object, *, resolve: bool) -> object:
            del resolve
            return value

    omegaconf_stub = ModuleType("omegaconf")
    omegaconf_stub.OmegaConf = OmegaConfStub
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf_stub)

    venv_stub = ModuleType("rlinf.envs.realworld.venv")
    venv_stub.NoAutoResetSyncVectorEnv = type("NoAutoResetSyncVectorEnv", (), {})
    monkeypatch.setitem(sys.modules, "rlinf.envs.realworld.venv", venv_stub)

    utils_stub = ModuleType("rlinf.envs.utils")
    utils_stub.to_tensor = lambda value: value
    monkeypatch.setitem(sys.modules, "rlinf.envs.utils", utils_stub)

    scheduler_stub = ModuleType("rlinf.scheduler")
    scheduler_stub.WorkerInfo = object
    monkeypatch.setitem(sys.modules, "rlinf.scheduler", scheduler_stub)

    module_path = ROOT / "rlinf" / "envs" / "realworld" / "realworld_env.py"
    spec = importlib.util.spec_from_file_location(
        "_realworld_env_under_test", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load RealWorldEnv")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.RealWorldEnv


def _raw_realworld_observation() -> dict[str, object]:
    return {
        "state": {
            "right_gripper": np.array([[16.0]], dtype=np.float32),
            "left_gripper": np.array([[8.0]], dtype=np.float32),
            "right_joint_position": np.arange(9, 16, dtype=np.float32)[None, :],
            "left_joint_position": np.arange(1, 8, dtype=np.float32)[None, :],
        },
        "frames": {
            "head_color": np.zeros((1, 128, 128, 3), dtype=np.uint8),
        },
    }


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
        {"max_joint_delta_rad": (0.01,) * 13},
        {"max_joint_delta_rad": (0.01,) * 13 + (-0.01,)},
        {"max_joint_delta_rad": (0.01,) * 13 + (np.nan,)},
        {"max_gripper_delta": -0.01},
        {"reset_joint_target_rad": (0.0,) * 13},
        {"reset_joint_target_rad": (0.0,) * 13 + (np.inf,)},
        {"reset_gripper_target": (0.5,)},
        {"reset_gripper_target": (-0.01, 0.5)},
        {"reset_gripper_target": (0.5, np.nan)},
        {"reset_duration_s": 0.0},
        {"reset_timeout_s": np.inf},
        {"reset_tolerance_rad": 0.0},
    ],
)
def test_f1_robot_config_rejects_invalid_safety_values(
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        F1RobotConfig(**overrides)


def test_f1_robot_config_copies_mutable_sequences_to_tuples() -> None:
    joint_delta = [0.01] * 14
    reset_target = [0.1] * 14
    gripper_target = [0.2, 0.8]

    config = F1RobotConfig(
        max_joint_delta_rad=joint_delta,
        reset_joint_target_rad=reset_target,
        reset_gripper_target=gripper_target,
    )
    joint_delta[0] = 99.0
    reset_target[0] = 99.0
    gripper_target[0] = 99.0

    assert config.max_joint_delta_rad == (0.01,) * 14
    assert config.reset_joint_target_rad == (0.1,) * 14
    assert config.reset_gripper_target == (0.2, 0.8)
    assert isinstance(config.max_joint_delta_rad, tuple)
    assert isinstance(config.reset_joint_target_rad, tuple)
    assert isinstance(config.reset_gripper_target, tuple)


def test_f1_exports_and_uses_the_canonical_state_order() -> None:
    assert F1_PACKAGE.F1_STATE_ORDER == EXPECTED_F1_STATE_ORDER

    env = F1RobotEnv(F1RobotConfig())
    try:
        assert tuple(env.observation_space["state"].spaces) == EXPECTED_F1_STATE_ORDER
    finally:
        env.close()


def test_realworld_wrap_obs_flattens_the_configured_canonical_state_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    realworld_env_class = _load_realworld_env_class(monkeypatch)
    env = realworld_env_class.__new__(realworld_env_class)
    env.main_image_key = "head_color"
    env.task_descriptions = ["task"]
    env.state_order = EXPECTED_F1_STATE_ORDER

    observation = env._wrap_obs(_raw_realworld_observation())

    np.testing.assert_array_equal(
        observation["states"],
        np.arange(1, 17, dtype=np.float32)[None, :],
    )


def test_realworld_wrap_obs_rejects_state_order_that_is_not_an_exact_key_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    realworld_env_class = _load_realworld_env_class(monkeypatch)
    env = realworld_env_class.__new__(realworld_env_class)
    env.main_image_key = "head_color"
    env.task_descriptions = ["task"]
    env.state_order = EXPECTED_F1_STATE_ORDER[:-1] + ("unexpected",)

    with pytest.raises(ValueError, match="state_order"):
        env._wrap_obs(_raw_realworld_observation())


def test_realworld_wrap_obs_without_state_order_keeps_sorted_legacy_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    realworld_env_class = _load_realworld_env_class(monkeypatch)
    env = realworld_env_class.__new__(realworld_env_class)
    env.main_image_key = "head_color"
    env.task_descriptions = ["task"]
    env.state_order = None

    observation = env._wrap_obs(_raw_realworld_observation())

    np.testing.assert_array_equal(
        observation["states"],
        np.array([[8.0, *range(1, 8), 16.0, *range(9, 16)]], dtype=np.float32),
    )


def test_realworld_env_reads_optional_state_order_from_top_level_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    realworld_env_class = _load_realworld_env_class(monkeypatch)

    def fake_init_env(env: object) -> None:
        state_spaces = {
            key: gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            for key in EXPECTED_F1_STATE_ORDER
        }
        env.env = SimpleNamespace(
            single_observation_space=gym.spaces.Dict(
                {"state": gym.spaces.Dict(state_spaces)}
            )
        )
        env.task_descriptions = ["task"]

    monkeypatch.setattr(realworld_env_class, "_init_env", fake_init_env)
    monkeypatch.setattr(realworld_env_class, "_init_metrics", lambda self: None)
    monkeypatch.setattr(
        realworld_env_class,
        "_init_reset_state_ids",
        lambda self: None,
    )
    cfg = SimpleNamespace(
        override_cfg={},
        video_cfg={},
        seed=1,
        use_fixed_reset_state_ids=False,
        auto_reset=False,
        ignore_terminations=False,
        group_size=1,
        main_image_key="head_color",
        state_order=list(EXPECTED_F1_STATE_ORDER),
    )
    cfg.get = lambda name, default=None: getattr(cfg, name, default)

    env = realworld_env_class(cfg, 1, 0, 1, None)

    assert env.state_order == EXPECTED_F1_STATE_ORDER


def test_realworld_env_rejects_configured_state_order_during_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    realworld_env_class = _load_realworld_env_class(monkeypatch)

    def fake_init_env(env: object) -> None:
        state_spaces = {
            key: gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            for key in EXPECTED_F1_STATE_ORDER
        }
        env.env = SimpleNamespace(
            single_observation_space=gym.spaces.Dict(
                {"state": gym.spaces.Dict(state_spaces)}
            )
        )
        env.task_descriptions = ["task"]

    monkeypatch.setattr(realworld_env_class, "_init_env", fake_init_env)
    monkeypatch.setattr(realworld_env_class, "_init_metrics", lambda self: None)
    monkeypatch.setattr(
        realworld_env_class,
        "_init_reset_state_ids",
        lambda self: None,
    )
    cfg = SimpleNamespace(
        override_cfg={},
        video_cfg={},
        seed=1,
        use_fixed_reset_state_ids=False,
        auto_reset=False,
        ignore_terminations=False,
        group_size=1,
        main_image_key="head_color",
        state_order=["left_joint_position"],
    )
    cfg.get = lambda name, default=None: getattr(cfg, name, default)

    with pytest.raises(ValueError, match="state_order"):
        realworld_env_class(cfg, 1, 0, 1, None)


def test_f1_robot_env_exposes_16d_action_state_and_three_rgb_frames() -> None:
    env = F1RobotEnv(F1RobotConfig())
    try:
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (16,)
        assert env.action_space.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(env.action_space.low, np.full(16, -1.0))
        np.testing.assert_array_equal(env.action_space.high, np.full(16, 1.0))

        state_space = env.observation_space["state"]
        assert tuple(state_space.spaces) == EXPECTED_F1_STATE_ORDER
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
        np.testing.assert_array_equal(
            info["absolute_left_joint_target_rad"],
            command.left_joint_target_rad,
        )
        np.testing.assert_array_equal(
            info["absolute_right_joint_target_rad"],
            command.right_joint_target_rad,
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
        info["absolute_left_joint_target_rad"].fill(99.0)
        assert not np.any(command.left_joint_target_rad == 99.0)
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


def test_step_initial_observation_failure_stops_without_masking_or_submitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    stale_error = ObservationUnavailableError("initial observation is stale")
    controller.read_error = stale_error
    controller.stop_error = ControllerLifecycleError("stop cleanup failed")
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(F1RobotConfig())
    try:
        with pytest.raises(ObservationUnavailableError) as captured:
            env.step(np.zeros(16, dtype=np.float32))

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
    env = F1RobotEnv(F1RobotConfig(control_period_s=0.001))
    try:
        with pytest.raises(ControllerError, match=status.value):
            env.step(np.zeros(16, dtype=np.float32))

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
    env = F1RobotEnv(F1RobotConfig(control_period_s=0.001))
    try:
        with pytest.raises(ControllerError, match=health.reason):
            env.step(np.zeros(16, dtype=np.float32))

        assert len(controller.stop_reasons) == 1
        assert env._num_steps == 0
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
        controller.events.clear()
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
        assert [event[0] for event in controller.events[:6]] == [
            "health",
            "stop_experiment_motion",
            "open",
            "wait_ready",
            "health",
            "submit_reset_command",
        ]
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
        assert len(controller.stop_reasons) == 2
    finally:
        env.close()


def test_reset_failed_status_stops_reset_motion_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController(reset_status=CommandStatus.REJECTED)
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(F1RobotConfig(reset_timeout_s=0.05))
    try:
        with pytest.raises(ControllerError, match="rejected"):
            env.reset()

        assert len(controller.stop_reasons) == 2
        assert len(controller.reset_commands) == 1
    finally:
        env.close()


def test_reset_reopens_stopped_controller_without_repeating_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(F1RobotConfig(reset_timeout_s=0.05))
    controller.health_error_once = ControllerNotReadyError("controller is stopped")
    controller.events.clear()
    try:
        observation, info = env.reset()

        assert env.observation_space.contains(observation)
        assert info["command_status"] is CommandStatus.FINISHED_DISPATCH
        assert controller.stop_reasons == []
        assert [event[0] for event in controller.events[:5]] == [
            "health",
            "open",
            "wait_ready",
            "health",
            "submit_reset_command",
        ]
    finally:
        env.close()


def test_reset_does_not_treat_other_health_errors_as_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(F1RobotConfig(reset_timeout_s=0.05))
    health_error = ControllerLifecycleError("health probe failed")
    controller.health_error_once = health_error
    controller.events.clear()
    try:
        with pytest.raises(ControllerLifecycleError) as captured:
            env.reset()

        assert captured.value is health_error
        assert controller.stop_reasons == []
        assert [event[0] for event in controller.events] == ["health"]
        assert controller.reset_commands == []
    finally:
        env.close()


def test_reset_preserves_active_controller_stop_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(F1RobotConfig(reset_timeout_s=0.05))
    stop_error = ControllerLifecycleError("previous motion could not stop")
    controller.stop_error = stop_error
    controller.events.clear()
    try:
        with pytest.raises(ControllerLifecycleError) as captured:
            env.reset()

        assert captured.value is stop_error
        assert len(controller.stop_reasons) == 1
        assert [event[0] for event in controller.events] == [
            "health",
            "stop_experiment_motion",
        ]
        assert controller.reset_commands == []
    finally:
        env.close()


def test_reset_unexpected_exception_stops_reset_motion_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RecordingController()
    _install_recording_controller(monkeypatch, controller)
    env = F1RobotEnv(F1RobotConfig(reset_timeout_s=0.05))
    controller.read_error = RuntimeError("read failed")
    try:
        with pytest.raises(RuntimeError, match="read failed"):
            env.reset()

        assert len(controller.stop_reasons) == 2
        assert len(controller.reset_commands) == 1
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
        F1RobotConfig(
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
    ) -> RobotObservation:
        newer_than_values.append(newer_than)
        return original_read(
            max_age_s=max_age_s,
            max_skew_s=max_skew_s,
            newer_than=newer_than,
        )

    monkeypatch.setattr(controller, "read_observation", recording_read_observation)
    try:
        reset_observation, reset_info = env.reset()
        action = np.full(16, 0.25, dtype=np.float32)
        step_observation, reward, terminated, truncated, step_info = env.step(action)

        assert env.observation_space.contains(reset_observation)
        assert env.observation_space.contains(step_observation)
        assert reset_info["command_status"] is CommandStatus.FINISHED_DISPATCH
        assert step_info["command_status"] is CommandStatus.FINISHED_DISPATCH
        assert reset_info["reset_command_id"] == 0
        assert step_info["command_id"] == 1
        assert newer_than_values[0] is not None
        assert newer_than_values[-1] == step_info["command_accepted_at_monotonic_s"]
        assert (
            step_info["command_created_at_monotonic_s"]
            <= step_info["command_accepted_at_monotonic_s"]
            < step_info["command_expires_at_monotonic_s"]
        )
        np.testing.assert_allclose(
            step_observation["state"]["left_joint_position"],
            step_info["absolute_left_joint_target_rad"],
        )
        np.testing.assert_allclose(
            step_observation["state"]["right_joint_position"],
            step_info["absolute_right_joint_target_rad"],
        )
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


def test_installed_fake_recovers_from_step_read_failure_through_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = F1RobotEnv(
        F1RobotConfig(
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
    ) -> RobotObservation:
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
            env.step(np.zeros(16, dtype=np.float32))
        assert captured.value is stale_error
        with pytest.raises(ControllerNotReadyError):
            controller.health()

        reset_observation, reset_info = env.reset()
        step_result = env.step(np.zeros(16, dtype=np.float32))

        assert env.observation_space.contains(reset_observation)
        assert reset_info["reset_command_id"] == 0
        assert reset_info["command_status"] is CommandStatus.FINISHED_DISPATCH
        assert step_result[4]["command_id"] == 1
        assert step_result[4]["command_status"] is CommandStatus.FINISHED_DISPATCH
        health = controller.health()
        assert health.ready
        assert not health.faulted
    finally:
        env.close()
