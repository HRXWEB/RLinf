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

"""Integration coverage for the F1 Gym registration and Hydra config."""

import importlib.util
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from f1_robot_controller import ControllerConfig, create_controller
from gymnasium.envs.registration import registry
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "examples" / "embodiment" / "config"
CONFIG_NAME = "realworld_dummy_f1_peg_sac_cnn_async"
ENV_ID = "F1DualArmPegInsertionEnv-v1"
LEGACY_ENV_ID = "F1DualArmPegInsertionEnv-v0"
F1_STATE_ORDER = [
    "left_joint_position",
    "left_gripper",
    "right_joint_position",
    "right_gripper",
]


def _namespace_package(name: str, path: Path) -> ModuleType:
    package = ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    return package


def _stub_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *,
    package: bool = False,
    **attributes: object,
) -> ModuleType:
    module = ModuleType(name)
    module.__package__ = name if package else name.rpartition(".")[0]
    if package:
        module.__path__ = []
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _StubRealWorldEnv:
    @staticmethod
    def realworld_setup() -> None:
        pass


def _load_realworld_package(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the real package init while stubbing unrelated robot stacks."""

    realworld_dir = ROOT / "rlinf" / "envs" / "realworld"
    monkeypatch.setitem(
        sys.modules, "rlinf", _namespace_package("rlinf", ROOT / "rlinf")
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs",
        _namespace_package("rlinf.envs", ROOT / "rlinf" / "envs"),
    )
    for name in tuple(sys.modules):
        if name.startswith("rlinf.envs.realworld"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    placeholder = type("StubRobot", (), {})
    dosw1_tasks = _stub_module(
        monkeypatch,
        "rlinf.envs.realworld.dosw1.tasks",
        package=True,
    )
    _stub_module(
        monkeypatch,
        "rlinf.envs.realworld.dosw1",
        package=True,
        DOSW1Config=placeholder,
        DOSW1Env=placeholder,
        tasks=dosw1_tasks,
    )
    franka_tasks = _stub_module(
        monkeypatch,
        "rlinf.envs.realworld.franka.tasks",
        package=True,
    )
    _stub_module(
        monkeypatch,
        "rlinf.envs.realworld.franka",
        package=True,
        FrankaEnv=placeholder,
        FrankaRobotConfig=placeholder,
        FrankaRobotState=placeholder,
        tasks=franka_tasks,
    )
    _stub_module(
        monkeypatch,
        "rlinf.envs.realworld.franka.dual_franka_env",
        DualFrankaEnv=placeholder,
        DualFrankaRobotConfig=placeholder,
    )
    _stub_module(
        monkeypatch,
        "rlinf.envs.realworld.franka.tasks.dual_franka_joint_env",
        DualFrankaJointEnv=placeholder,
        DualFrankaJointRobotConfig=placeholder,
    )
    _stub_module(
        monkeypatch,
        "rlinf.envs.realworld.franka.tasks.dual_franka_tcp_env",
        DualFrankaTCPEnv=placeholder,
        DualFrankaTCPRobotConfig=placeholder,
    )
    gim_tasks = _stub_module(
        monkeypatch,
        "rlinf.envs.realworld.gim_arm.tasks",
        package=True,
    )
    _stub_module(
        monkeypatch,
        "rlinf.envs.realworld.gim_arm",
        package=True,
        GimArmEnv=placeholder,
        GimArmRobotConfig=placeholder,
        GimArmRobotState=placeholder,
        tasks=gim_tasks,
    )
    xsquare_tasks = _stub_module(
        monkeypatch,
        "rlinf.envs.realworld.xsquare.tasks",
        package=True,
    )
    _stub_module(
        monkeypatch,
        "rlinf.envs.realworld.xsquare",
        package=True,
        Turtle2Env=placeholder,
        Turtle2RobotConfig=placeholder,
        Turtle2RobotState=placeholder,
        tasks=xsquare_tasks,
    )
    _stub_module(
        monkeypatch,
        "rlinf.envs.realworld.realworld_env",
        RealWorldEnv=_StubRealWorldEnv,
    )

    package_init = realworld_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "rlinf.envs.realworld",
        package_init,
        submodule_search_locations=[str(realworld_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the real-world package")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "rlinf.envs.realworld", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def registered_f1(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    registry.pop(ENV_ID, None)
    registry.pop(LEGACY_ENV_ID, None)
    module = _load_realworld_package(monkeypatch)
    try:
        yield module
    finally:
        registry.pop(ENV_ID, None)
        registry.pop(LEGACY_ENV_ID, None)


def _make_env(
    *,
    mode: str,
    is_dummy: bool = True,
    operator_control: Mapping[str, object] | None = None,
    **task_overrides: object,
) -> gym.Env:
    control = (
        dict(operator_control)
        if operator_control is not None
        else {"mode": mode, "timeout_s": 0.0}
    )
    return gym.make(
        ENV_ID,
        override_cfg={
            "is_dummy": is_dummy,
            "control_period_s": 0.001,
            **task_overrides,
        },
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg={"operator_control": control},
    )


def _record_controller_motion(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_is_dummy: bool,
) -> tuple[list[str], list[str]]:
    env_module = sys.modules["rlinf.envs.realworld.f1.f1_robot_env"]
    motion_calls: list[str] = []
    close_calls: list[str] = []

    def controller_factory(
        *,
        is_dummy: bool,
        config: ControllerConfig,
    ) -> object:
        assert is_dummy is expected_is_dummy
        controller = create_controller(is_dummy=True, config=config)
        original_submit = controller.submit_command
        original_reset = controller.submit_reset_command
        original_close = controller.close

        def submit_command(command: Any) -> Any:
            motion_calls.append("policy")
            return original_submit(command)

        def submit_reset_command(command: Any) -> Any:
            motion_calls.append("reset")
            return original_reset(command)

        def close() -> None:
            close_calls.append("close")
            original_close()

        monkeypatch.setattr(controller, "submit_command", submit_command)
        monkeypatch.setattr(controller, "submit_reset_command", submit_reset_command)
        monkeypatch.setattr(controller, "close", close)
        return controller

    monkeypatch.setattr(env_module, "create_controller", controller_factory)
    return motion_calls, close_calls


def test_hydra_composes_the_f1_train_and_eval_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples" / "embodiment"))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name=CONFIG_NAME)

    assert cfg.cluster.num_nodes == 1
    assert dict(cfg.cluster.component_placement) == {
        "actor": 0,
        "env": 0,
        "rollout": 0,
    }
    assert cfg.runner.max_steps == 2
    assert cfg.algorithm.adv_type == "embodied_sac"
    assert cfg.algorithm.loss_type == "embodied_sac"
    assert "demo_buffer" not in cfg.algorithm
    assert cfg.algorithm.replay_buffer.min_buffer_size == 1
    assert cfg.algorithm.entropy_tuning.target_entropy == -16
    assert cfg.actor.micro_batch_size == 8
    assert cfg.actor.global_batch_size == 8
    assert cfg.actor.model.state_dim == 16
    assert cfg.actor.model.action_dim == 16

    for section in (cfg.env.train, cfg.env.eval):
        assert section.env_type == "realworld"
        assert section.init_params.id == ENV_ID
        assert section.video_cfg.save_video is False
        assert section.auto_reset is True
        assert section.max_episode_steps == 4
        assert section.main_image_key == "head_color"
        assert list(section.state_order) == F1_STATE_ORDER
        assert section.operator_control.mode == "automatic"
        assert section.override_cfg.is_dummy is True


def test_importing_realworld_registers_only_v1_without_ros(
    registered_f1: ModuleType,
) -> None:
    del registered_f1
    forbidden_before = {
        name
        for name in sys.modules
        if name == "rclpy"
        or name.startswith("rclpy.")
        or name == "cv_bridge"
        or name.startswith("cv_bridge.")
    }

    assert gym.spec(ENV_ID).entry_point == (
        "rlinf.envs.realworld.f1.tasks:DualArmPegInsertionEnv"
    )
    with pytest.raises(gym.error.Error):
        gym.spec(LEGACY_ENV_ID)

    forbidden_after = {
        name
        for name in sys.modules
        if name == "rclpy"
        or name.startswith("rclpy.")
        or name == "cv_bridge"
        or name.startswith("cv_bridge.")
    }
    assert forbidden_after == forbidden_before


def test_each_gym_make_uses_its_own_dynamic_operator_config(
    registered_f1: ModuleType,
) -> None:
    del registered_f1
    automatic = _make_env(mode="automatic")
    manual = _make_env(mode="manual")
    try:
        assert (
            automatic.unwrapped.spec.kwargs["env_cfg"]["operator_control"]["mode"]
            == "automatic"
        )
        assert manual.unwrapped.spec.kwargs["env_cfg"]["operator_control"]["mode"] == (
            "manual"
        )
        assert gym.spec(ENV_ID).kwargs == {}
        assert automatic.__class__.__name__ == "SupervisedEpisodeControlWrapper"
        assert manual.__class__.__name__ == "SupervisedEpisodeControlWrapper"
    finally:
        automatic.close()
        manual.close()


def test_automatic_fake_wrapper_runs_reset_and_four_step_horizon_without_ros(
    registered_f1: ModuleType,
) -> None:
    del registered_f1
    forbidden_before = {
        name
        for name in sys.modules
        if name == "rclpy"
        or name.startswith("rclpy.")
        or name == "cv_bridge"
        or name.startswith("cv_bridge.")
    }
    env = _make_env(mode="automatic")
    try:
        observation, _ = env.reset(seed=7)
        assert env.observation_space.contains(observation)

        truncations = []
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step(np.zeros(16, dtype=np.float32))
            assert terminated is False
            truncations.append(truncated)
        assert truncations == [False, False, False, True]
        assert env.get_wrapper_attr("state").value == "waiting_reset"
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


def test_manual_mode_never_approves_robot_reset_automatically(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del registered_f1
    motion_calls, _ = _record_controller_motion(
        monkeypatch,
        expected_is_dummy=True,
    )
    env = _make_env(mode="manual")
    try:
        with pytest.raises(TimeoutError, match="reset_approved"):
            env.reset()
        assert motion_calls == []
        assert env.get_wrapper_attr("state").value == "waiting_reset"
    finally:
        env.close()


def test_automatic_mode_uses_constructed_backend_truth_and_cannot_be_spoofed(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del registered_f1
    motion_calls, close_calls = _record_controller_motion(
        monkeypatch,
        expected_is_dummy=False,
    )

    with pytest.raises(ValueError, match="Fake backend"):
        gym.make(
            ENV_ID,
            override_cfg={"is_dummy": False},
            env_cfg={
                "backend_is_fake": True,
                "operator_control": {"mode": "automatic", "timeout_s": 0.0},
            },
        )

    assert motion_calls == []
    assert close_calls == ["close"]


@pytest.mark.parametrize(
    "operator_control",
    [
        {"mode": "unknown", "timeout_s": 0.0},
        {"mode": "automatic", "timeout_s": 0.0, "unexpected": True},
    ],
    ids=["unknown-mode", "unknown-field"],
)
def test_unknown_operator_control_fails_closed_before_motion(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    operator_control: dict[str, object],
) -> None:
    del registered_f1
    motion_calls, close_calls = _record_controller_motion(
        monkeypatch,
        expected_is_dummy=True,
    )

    with pytest.raises(ValueError, match="operator_control"):
        _make_env(
            mode="automatic",
            operator_control=operator_control,
        )

    assert motion_calls == []
    assert close_calls == ["close"]


def test_unknown_task_override_still_fails_closed(
    registered_f1: ModuleType,
) -> None:
    del registered_f1
    with pytest.raises(TypeError, match="unexpected_task_field"):
        _make_env(mode="automatic", unexpected_task_field=True)
