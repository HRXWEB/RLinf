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
        override_cfg={"control_period_s": 0.001, **overrides},
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg={
            "source": "realworld-wrapper",
            "operator_control": {"mode": "automatic", "timeout_s": 0.0},
        },
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


def test_config_has_fixed_dual_arm_reset_targets(task_module: ModuleType) -> None:
    config = task_module.DualArmPegInsertionConfig()

    assert config.reset_joint_target_rad == (0.0,) * 14
    assert config.reset_gripper_target == (0.5, 0.5)
    assert config.max_num_steps == 4


@pytest.mark.parametrize(
    ("overrides", "expected_field"),
    [
        (
            {"reset_joint_target_rad": (0.1,) + (0.0,) * 13},
            "reset_joint_target_rad",
        ),
        ({"reset_gripper_target": (0.4, 0.5)}, "reset_gripper_target"),
        ({"max_num_steps": 5}, "max_num_steps"),
    ],
    ids=["noncanonical-joints", "noncanonical-grippers", "noncanonical-horizon"],
)
def test_noncanonical_task_contract_fails_before_controller_creation(
    task_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    expected_field: str,
) -> None:
    controller_factory_calls: list[object] = []
    env_module = sys.modules["rlinf.envs.realworld.f1.f1_robot_env"]

    def fail_if_controller_is_created(**kwargs: object) -> object:
        controller_factory_calls.append(kwargs)
        raise AssertionError("invalid task config must not create a controller")

    monkeypatch.setattr(env_module, "create_controller", fail_if_controller_is_created)

    with pytest.raises(ValueError, match=expected_field):
        _make_task_env(**overrides)

    assert controller_factory_calls == []


def test_explicit_canonical_task_contract_is_accepted(
    task_module: ModuleType,
) -> None:
    env = _make_task_env(
        reset_joint_target_rad=[0.0] * 14,
        reset_gripper_target=[0.5, 0.5],
        max_num_steps=4,
    )
    try:
        assert isinstance(env.unwrapped, task_module.DualArmPegInsertionEnv)
        assert env.unwrapped.config.reset_joint_target_rad == (0.0,) * 14
        assert env.unwrapped.config.reset_gripper_target == (0.5, 0.5)
        assert env.unwrapped.config.max_num_steps == 4
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
        assert reset_info["reset_command_id"] == 0

        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        for _ in range(4):
            observation, reward, terminated, truncated, _ = env.step(
                np.zeros(16, dtype=np.float32)
            )
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)

        assert rewards == [0.0, 0.0, 0.0, 0.0]
        assert terminations == [False, False, False, False]
        assert truncations == [False, False, False, True]
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
