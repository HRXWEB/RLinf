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
from gymnasium.envs.registration import registry

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
F1_PACKAGE_DIR = ROOT / "rlinf" / "envs" / "realworld" / "f1"
TASKS_PACKAGE = "rlinf.envs.realworld.f1.tasks"
ENV_ID = "F1DualArmPegInsertionEnv-v1"
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
