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

"""Behavioral coverage for the F1 dual-arm peg-insertion task."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator

import gymnasium as gym
import numpy as np
import pytest
from f1_robot_controller import ObservationUnavailableError
from gymnasium.envs.registration import registry

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
F1_PACKAGE_DIR = ROOT / "rlinf" / "envs" / "realworld" / "f1"
TASKS_PACKAGE = "rlinf.envs.realworld.f1.tasks"
ENV_ID = "F1DualArmPegInsertionEnv-v1"
RIGHT_ARM_ENV_ID = "F1RightArmPegInsertionEnv-v0"
RIGHT_ARM_REACH_ENV_ID = "F1RightArmFixedReachEnv-v0"
LEGACY_ENV_ID = "F1DualArmPegInsertionEnv-v0"
EXPECTED_TASK_DESCRIPTION = (
    "Use both arms cooperatively to insert the peg into the matching hole."
)


def _namespace_package(name: str, path: Path) -> ModuleType:
    package = ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    return package


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


def _fake_controller_config() -> dict[str, object]:
    return {
        "backend": "fake",
        "control_period_s": 0.001,
        "max_observation_age_s": 0.25,
        "max_observation_skew_s": 0.05,
        "fake": {
            "seed": 7,
            "image_height": 32,
            "image_width": 48,
            "command_latency_s": 0.0,
            "command_history_limit": 32,
        },
    }


def _action_scale() -> dict[str, float]:
    return {
        "tcp_position_m": 0.005,
        "tcp_orientation_deg": 1.0,
        "gripper_percent_closed": 10.0,
    }


@pytest.fixture
def task_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Load only the F1 task package, avoiding unrelated heavyweight imports."""

    package_paths = {
        "rlinf": ROOT / "rlinf",
        "rlinf.envs": ROOT / "rlinf" / "envs",
        "rlinf.envs.realworld": ROOT / "rlinf" / "envs" / "realworld",
        "rlinf.envs.realworld.f1": F1_PACKAGE_DIR,
    }
    for name, path in package_paths.items():
        monkeypatch.setitem(sys.modules, name, _namespace_package(name, path))

    tasks_init = F1_PACKAGE_DIR / "tasks" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        TASKS_PACKAGE,
        tasks_init,
        submodule_search_locations=[str(tasks_init.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the F1 task package")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, TASKS_PACKAGE, module)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        registry.pop(ENV_ID, None)
        registry.pop(RIGHT_ARM_ENV_ID, None)
        registry.pop(RIGHT_ARM_REACH_ENV_ID, None)


def _make_task_env(**overrides: object) -> gym.Env:
    return gym.make(
        ENV_ID,
        override_cfg={
            "controller": _fake_controller_config(),
            "action_scale": _action_scale(),
            "motion_envelope": _approved_motion_envelope(),
            **overrides,
        },
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg={"source": "realworld-wrapper"},
    )


def _make_right_arm_env(**overrides: object) -> gym.Env:
    return gym.make(
        RIGHT_ARM_ENV_ID,
        override_cfg={
            "controller": _fake_controller_config(),
            "action_scale": _action_scale(),
            "motion_envelope": _approved_motion_envelope(),
            "target_tcp_pose_m_deg": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            "reset_left_joint_pose_deg": [90, -90, -90, -90, 0, 0, 0],
            "reset_right_joint_pose_deg": [-90, -90, 90, -90, 0, 0, 0],
            "joint_reset_tolerance_deg": 2.0,
            "tcp_reference_frame": "right_arm_tcp_pose",
            "position_reward_scale_m": 0.02,
            "orientation_reward_scale_deg": 10.0,
            "position_weight": 0.7,
            "orientation_weight": 0.3,
            "step_penalty": 0.01,
            "position_tolerance_m": 0.002,
            "orientation_tolerance_deg": 2.0,
            "success_hold_steps": 1,
            **overrides,
        },
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg={"source": "realworld-wrapper"},
    )


def _make_right_arm_reach_env(**overrides: object) -> gym.Env:
    motion_envelope = _approved_motion_envelope()
    motion_envelope["right_arm"]["tcp"]["max_delta"]["orientation_deg"] = 5.0
    return gym.make(
        RIGHT_ARM_REACH_ENV_ID,
        override_cfg={
            "controller": _fake_controller_config(),
            "action_scale": _action_scale(),
            "motion_envelope": motion_envelope,
            "target_tcp_pose_m_deg": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            "reset_left_joint_pose_deg": [90, -90, -90, -90, 0, 0, 0],
            "reset_right_joint_pose_deg": [-90, -90, 90, -90, 0, 0, 0],
            "joint_reset_tolerance_deg": 2.0,
            "tcp_reference_frame": "right_arm_tcp_pose",
            "fixed_orientation_deg": [0.0, 0.0, 0.0],
            "vendor_to_base_rotation": np.eye(3).tolist(),
            "workspace_lower_offset_m": [-0.05, -0.05, -0.01],
            "workspace_upper_offset_m": [0.05, 0.05, 0.05],
            "position_reward_scale_m": 0.01,
            "step_penalty": 0.01,
            "xy_tolerance_m": 0.002,
            "z_tolerance_m": 0.002,
            "success_hold_steps": 1,
            "policy_image_shape": [128, 224],
            **overrides,
        },
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg={"source": "realworld-wrapper"},
    )


def test_registration_exposes_only_canonical_v1_and_is_reload_safe(
    task_module: ModuleType,
) -> None:
    specification = gym.spec(ENV_ID)

    assert specification.entry_point == (
        "rlinf.envs.realworld.f1.tasks:DualArmPegInsertionEnv"
    )
    with pytest.raises(gym.error.Error):
        gym.spec(LEGACY_ENV_ID)

    reloaded = importlib.reload(task_module)
    assert reloaded is task_module
    assert gym.spec(ENV_ID) is specification


def test_registration_exposes_right_arm_fixed_target_task(
    task_module: ModuleType,
) -> None:
    """Catch the single-arm task disappearing from the public Gym registry."""

    assert RIGHT_ARM_ENV_ID in registry
    specification = gym.spec(RIGHT_ARM_ENV_ID)
    assert specification.entry_point == (
        "rlinf.envs.realworld.f1.tasks:create_right_arm_peg_insertion_env"
    )


def test_registration_exposes_right_arm_fixed_reach_task(
    task_module: ModuleType,
) -> None:
    """Catch the curriculum base task disappearing from the Gym registry."""

    assert RIGHT_ARM_REACH_ENV_ID in registry
    specification = gym.spec(RIGHT_ARM_REACH_ENV_ID)
    assert specification.entry_point == (
        "rlinf.envs.realworld.f1.tasks:create_right_arm_fixed_reach_env"
    )


def test_fixed_reach_uses_full_right_wrist_image_xyz_state_and_three_actions(
    task_module: ModuleType,
) -> None:
    """Catch the reach policy receiving a cropped/head frame or 6D action."""

    env = _make_right_arm_reach_env()
    try:
        observation, _ = env.reset()

        assert env.action_space.shape == (3,)
        assert set(observation["frames"]) == {"right_wrist_color"}
        assert observation["frames"]["right_wrist_color"].shape == (128, 224, 3)
        assert observation["state"]["proprioception"].shape == (3,)
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_fixed_reach_action_holds_orientation_and_inactive_actuators(
    task_module: ModuleType,
) -> None:
    """Catch XYZ policy actions leaking into orientation, grippers, or left arm."""

    env = _make_right_arm_reach_env()
    try:
        env.reset()
        _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))

        np.testing.assert_allclose(info["absolute_left_tcp_target_m_deg"], 0.0)
        assert info["absolute_left_gripper_target"] == pytest.approx(50.0)
        np.testing.assert_allclose(
            info["absolute_right_tcp_target_m_deg"],
            [0.005, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        assert info["absolute_right_gripper_target"] == pytest.approx(50.0)
    finally:
        env.close()


def test_fixed_reach_clips_candidate_position_to_base_workspace(
    task_module: ModuleType,
) -> None:
    """Catch normalized actions escaping the calibrated base-frame box."""

    env = _make_right_arm_reach_env(
        target_tcp_pose_m_deg=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        workspace_lower_offset_m=[-0.001, -0.001, -0.001],
        workspace_upper_offset_m=[0.001, 0.001, 0.001],
        workspace_command_margin_m=0.0005,
    )
    try:
        env.reset()
        _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))

        np.testing.assert_allclose(
            info["absolute_right_tcp_target_m_deg"],
            [0.0005, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
    finally:
        env.close()


def test_fixed_reach_clips_in_base_frame_with_non_identity_rotation(
    task_module: ModuleType,
) -> None:
    """Catch transposed or sign-flipped vendor-to-base workspace transforms."""

    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    env = _make_right_arm_reach_env(
        target_tcp_pose_m_deg=[0.0] * 6,
        vendor_to_base_rotation=rotation.tolist(),
        workspace_lower_offset_m=[-0.01, -0.001, -0.01],
        workspace_upper_offset_m=[0.01, 0.001, 0.01],
    )
    try:
        env.reset()
        _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))

        commanded_vendor = info["absolute_right_tcp_target_m_deg"][:3]
        np.testing.assert_allclose(commanded_vendor, [0.001, 0.0, 0.0])
        np.testing.assert_allclose(rotation @ commanded_vendor, [0.0, 0.001, 0.0])
    finally:
        env.close()


def test_fixed_reach_rejects_unsafe_fixed_orientation_jump(
    task_module: ModuleType,
) -> None:
    """Catch fixed-orientation commands bypassing the production angular limit."""

    env = _make_right_arm_reach_env(fixed_orientation_deg=[90.0, 0.0, 90.0])
    try:
        with pytest.raises(ValueError, match="fixed orientation delta"):
            env.unwrapped._bounded_tcp_target(
                arm_name="right_arm",
                current=np.zeros(6, dtype=np.float64),
                action=np.zeros(6, dtype=np.float64),
            )
    finally:
        env.close()


def test_fixed_reach_rejects_margin_that_collapses_command_workspace(
    task_module: ModuleType,
) -> None:
    """Catch a safety margin eliminating an axis of the command workspace."""

    with pytest.raises(ValueError, match="nonempty command workspace"):
        _make_right_arm_reach_env(
            workspace_lower_offset_m=[-0.005, -0.005, -0.005],
            workspace_upper_offset_m=[0.005, 0.005, 0.005],
            workspace_command_margin_m=0.005,
        )


def test_fixed_reach_requires_three_consecutive_success_steps_and_bonuses_once(
    task_module: ModuleType,
) -> None:
    """Catch early success or a stale hold counter after leaving tolerance."""

    env = _make_right_arm_reach_env(
        target_tcp_pose_m_deg=[0.005, 0.0, 0.0, 0.0, 0.0, 0.0],
        success_hold_steps=3,
    )
    try:
        env.reset()

        _, reward_1, terminated_1, _, _ = env.step(
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        _, _, terminated_away, _, _ = env.step(
            np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        )
        _, reward_2, terminated_2, _, _ = env.step(
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        _, reward_3, terminated_3, _, _ = env.step(np.zeros(3, dtype=np.float32))
        _, reward_4, terminated_4, _, _ = env.step(np.zeros(3, dtype=np.float32))

        assert reward_1 < 5.0
        assert reward_2 < 5.0
        assert reward_3 < 5.0
        assert reward_4 == pytest.approx(4.99)
        assert not terminated_1
        assert not terminated_away
        assert not terminated_2
        assert not terminated_3
        assert terminated_4
    finally:
        env.close()


def test_right_arm_task_exposes_only_head_image_tcp_state_and_six_actions(
    task_module: ModuleType,
) -> None:
    """Catch dual-arm observations or actions leaking into the single-arm API."""

    env = _make_right_arm_env()
    try:
        observation, info = env.reset()

        assert env.action_space.shape == (6,)
        assert set(observation["frames"]) == {"head_color"}
        assert observation["state"]["proprioception"].shape == (6,)
        assert env.observation_space.contains(observation)
        assert info["tcp_reference_frame"] == "right_arm_tcp_pose"
    finally:
        env.close()


def test_right_arm_reset_moves_both_arms_to_configured_joint_pose(
    task_module: ModuleType,
) -> None:
    env = _make_right_arm_env(reset_duration_s=0.001, reset_timeout_s=0.1)
    try:
        _, info = env.reset()
        measured = env.unwrapped._active_controller.read_observation(
            max_age_s=0.25,
            max_skew_s=0.05,
        )

        np.testing.assert_allclose(
            np.rad2deg(measured.left_joint_position_rad),
            [90, -90, -90, -90, 0, 0, 0],
        )
        np.testing.assert_allclose(
            np.rad2deg(measured.right_joint_position_rad),
            [-90, -90, 90, -90, 0, 0, 0],
        )
        assert info["reset_mode"] == "dual_arm_joint_pose"
        assert env.unwrapped._next_command_id == 1
    finally:
        env.close()


def test_right_arm_reset_requires_a_post_acceptance_observation(
    task_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_right_arm_env(reset_duration_s=0.001, reset_timeout_s=0.1)
    controller = env.unwrapped._active_controller
    original_read = controller.read_observation
    newer_than_values: list[float | None] = []

    def record_read(**kwargs: float | None) -> object:
        newer_than_values.append(kwargs.get("newer_than"))
        return original_read(**kwargs)

    monkeypatch.setattr(controller, "read_observation", record_read)
    try:
        env.reset()

        assert any(value is not None for value in newer_than_values)
    finally:
        env.close()


def test_right_arm_reset_stops_when_post_acceptance_state_never_arrives(
    task_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_right_arm_env(reset_duration_s=0.001, reset_timeout_s=0.01)
    controller = env.unwrapped._active_controller
    original_read = controller.read_observation
    original_stop = controller.stop_command_dispatch
    stop_reasons: list[str] = []

    def reject_post_acceptance(**kwargs: float | None) -> object:
        if kwargs.get("newer_than") is not None:
            raise ObservationUnavailableError("no post-reset observation")
        return original_read(**kwargs)

    def record_stop(reason: str) -> None:
        stop_reasons.append(reason)
        original_stop(reason)

    monkeypatch.setattr(controller, "read_observation", reject_post_acceptance)
    monkeypatch.setattr(controller, "stop_command_dispatch", record_stop)
    try:
        with pytest.raises(TimeoutError, match="did not converge"):
            env.reset()

        assert len(stop_reasons) == 1
        assert stop_reasons[0].startswith("reset failed")
    finally:
        env.close()


def test_right_arm_action_holds_left_arm_and_both_grippers(
    task_module: ModuleType,
) -> None:
    """Catch the six-dimensional policy action moving an inactive actuator."""

    env = _make_right_arm_env()
    try:
        env.reset()
        _, reward, terminated, truncated, info = env.step(
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )

        np.testing.assert_allclose(info["absolute_left_tcp_target_m_deg"], 0.0)
        assert info["absolute_left_gripper_target"] == pytest.approx(50.0)
        np.testing.assert_allclose(
            info["absolute_right_tcp_target_m_deg"],
            [0.005, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        assert info["absolute_right_gripper_target"] == pytest.approx(50.0)
        assert reward > 0.0
        assert terminated is False
        assert truncated is False
    finally:
        env.close()


def test_right_arm_dense_reward_approaches_the_only_target_pose(
    task_module: ModuleType,
) -> None:
    """Catch reward depending on a pre-insert pose or approach direction."""

    env = _make_right_arm_env()
    try:
        env.reset()
        _, first_reward, first_terminated, _, _ = env.step(
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        _, second_reward, second_terminated, _, info = env.step(
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )

        assert first_reward > 0.0
        assert first_terminated is False
        assert second_reward > first_reward
        assert second_terminated is True
        assert info["is_success"] is True
        assert info["position_error_m"] == pytest.approx(0.0, abs=1e-8)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"target_tcp_pose_m_deg": [0.0] * 5},
            "target_tcp_pose_m_deg",
        ),
        (
            {"tcp_reference_frame": ""},
            "tcp_reference_frame",
        ),
        (
            {"success_hold_steps": 0},
            "success_hold_steps",
        ),
    ],
)
def test_right_arm_task_rejects_ambiguous_calibration(
    task_module: ModuleType,
    overrides: dict[str, object],
    message: str,
) -> None:
    """Catch invalid target calibration reaching the robot controller."""

    with pytest.raises((TypeError, ValueError), match=message):
        _make_right_arm_env(**overrides)


def test_right_arm_stationary_action_cannot_farm_dense_reward(
    task_module: ModuleType,
) -> None:
    """A stationary policy away from the target receives only the step cost."""

    env = _make_right_arm_env()
    try:
        env.reset()
        _, first_reward, first_terminated, _, _ = env.step(
            np.zeros(6, dtype=np.float32)
        )
        _, second_reward, second_terminated, _, _ = env.step(
            np.zeros(6, dtype=np.float32)
        )

        assert first_reward == pytest.approx(-0.01)
        assert second_reward == pytest.approx(-0.01)
        assert first_terminated is False
        assert second_terminated is False
    finally:
        env.close()


def test_config_defaults_to_ten_steps(task_module: ModuleType) -> None:
    config = task_module.DualArmPegInsertionConfig(
        controller=_fake_controller_config(),
        action_scale=_action_scale(),
        motion_envelope=_approved_motion_envelope(),
    )

    assert config.max_num_steps == 10


def test_task_accepts_a_custom_positive_horizon(task_module: ModuleType) -> None:
    env = _make_task_env(max_num_steps=7)
    try:
        assert isinstance(env.unwrapped, task_module.DualArmPegInsertionEnv)
        assert env.unwrapped.config.max_num_steps == 7
    finally:
        env.close()


def test_nonpositive_horizon_fails_before_controller_creation(
    task_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller_factory_calls: list[object] = []
    env_module = sys.modules["rlinf.envs.realworld.f1.f1_robot_env"]

    def fail_if_controller_is_created(**kwargs: object) -> object:
        controller_factory_calls.append(kwargs)
        raise AssertionError("invalid task config must not create a controller")

    monkeypatch.setattr(env_module, "create_controller", fail_if_controller_is_created)

    with pytest.raises(ValueError, match="max_num_steps"):
        _make_task_env(max_num_steps=0)

    assert controller_factory_calls == []


def test_direct_gym_make_has_no_operator_wrapper_and_uses_current_state_origin(
    task_module: ModuleType,
) -> None:
    env = _make_task_env(max_num_steps=10)
    try:
        assert isinstance(env.unwrapped, task_module.DualArmPegInsertionEnv)
        assert env.unwrapped.__class__ is task_module.DualArmPegInsertionEnv
        assert env.unwrapped.config.max_num_steps == 10
        assert env.action_space.low.tolist() == [-1.0] * 14
        assert env.action_space.high.tolist() == [1.0] * 14

        observation, info = env.reset()
        assert env.observation_space.contains(observation)
        assert info["reset_mode"] == "current_state_origin"
        assert info["session_origin_state"].shape == (16,)
        assert env.unwrapped._next_command_id == 0
    finally:
        env.close()


def test_explicit_default_task_horizon_is_accepted(
    task_module: ModuleType,
) -> None:
    env = _make_task_env(
        max_num_steps=10,
    )
    try:
        assert isinstance(env.unwrapped, task_module.DualArmPegInsertionEnv)
        assert env.unwrapped.config.max_num_steps == 10
    finally:
        env.close()


def test_unknown_task_override_fails_closed(task_module: ModuleType) -> None:
    del task_module

    with pytest.raises(TypeError, match="unexpected_task_field"):
        _make_task_env(unexpected_task_field=True)


def test_gym_make_runs_installed_fake_reset_reward_success_and_horizon(
    task_module: ModuleType,
) -> None:
    forbidden_before = {
        name
        for name in sys.modules
        if name == "rclpy"
        or name.startswith("rclpy.")
        or name == "cv_bridge"
        or name.startswith("cv_bridge.")
    }
    env = _make_task_env()
    try:
        assert isinstance(env.unwrapped, task_module.DualArmPegInsertionEnv)
        assert env.get_wrapper_attr("task_description") == EXPECTED_TASK_DESCRIPTION

        observation, reset_info = env.reset(seed=7)
        assert env.observation_space.contains(observation)
        assert reset_info["reset_mode"] == "current_state_origin"
        assert reset_info["session_origin_state"].shape == (16,)

        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        for _ in range(10):
            observation, reward, terminated, truncated, _ = env.step(
                np.zeros(14, dtype=np.float32)
            )
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)

        assert rewards == [0.0] * 10
        assert terminations == [False] * 10
        assert truncations == [False] * 9 + [True]
        assert env.unwrapped._calc_step_reward(observation) == 0.0
        assert env.unwrapped._is_success(observation) is False
    finally:
        env.close()

    forbidden_after = {
        name
        for name in sys.modules
        if name == "rclpy"
        or name.startswith("rclpy.")
        or name == "cv_bridge"
        or name.startswith("cv_bridge.")
    }
    assert forbidden_after == forbidden_before
