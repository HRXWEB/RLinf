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

import importlib
import inspect
import json
import subprocess
import sys
import textwrap
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
from omegaconf import open_dict

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "examples" / "embodiment" / "config"
CONFIG_NAME = "realworld_dummy_f1_peg_sac_cnn_async"
ENV_ID = "F1DualArmPegInsertionEnv-v1"
LEGACY_ENV_ID = "F1DualArmPegInsertionEnv-v0"
LEGACY_SPECS = {
    "DOSW1PickEnv-v1": "rlinf.envs.realworld.dosw1.tasks:create_dosw1_pick_env",
    "FrankaEnv-v1": "rlinf.envs.realworld.franka.tasks:create_franka_env",
    "DualFrankaJointEnv-v1": (
        "rlinf.envs.realworld.franka.tasks:create_dual_franka_joint_env"
    ),
    "DualFrankaTCPEnv-v1": (
        "rlinf.envs.realworld.franka.tasks:create_dual_franka_tcp_env"
    ),
    "PegInsertionEnv-v1": (
        "rlinf.envs.realworld.franka.tasks:create_peg_insertion_env"
    ),
    "FrankaBinRelocationEnv-v1": (
        "rlinf.envs.realworld.franka.tasks:create_franka_bin_relocation_env"
    ),
    "BottleEnv-v1": "rlinf.envs.realworld.franka.tasks:create_bottle_env",
    "DexpnpEnv-v1": "rlinf.envs.realworld.franka.tasks:create_dexpnp_env",
    "GimArmPegInsertionEnv-v1": (
        "rlinf.envs.realworld.gim_arm.tasks:GimArmPegInsertionEnv"
    ),
    "ButtonEnv-v1": "rlinf.envs.realworld.xsquare.tasks:create_button_env",
}
F1_STATE_ORDER = [
    "left_joint_position",
    "left_gripper",
    "right_joint_position",
    "right_gripper",
]


def _gym_registration_metadata_conflicts() -> dict[str, object]:
    """Return one non-default value for every supported registration option."""

    candidates: dict[str, object] = {
        "reward_threshold": 1.0,
        "nondeterministic": True,
        "max_episode_steps": 1,
        "order_enforce": False,
        "disable_env_checker": True,
        "kwargs": {"conflict": True},
        "vector_entry_point": "conflicting.module:create_vector_env",
        "apply_api_compatibility": True,
        "autoreset": True,
    }
    supported = inspect.signature(gym.register).parameters
    return {name: value for name, value in candidates.items() if name in supported}


def _load_realworld_package(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Reload the real package and its F1 registration without dependency stubs."""

    monkeypatch.syspath_prepend(str(ROOT))
    envs_package = importlib.import_module("rlinf.envs")
    monkeypatch.delattr(envs_package, "realworld", raising=False)
    for name in tuple(sys.modules):
        if name.startswith("rlinf.envs.realworld"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("rlinf.envs.realworld")


def _compose_config(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples" / "embodiment"))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        return compose(config_name=CONFIG_NAME)


@pytest.fixture
def registered_f1(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    for env_id in (ENV_ID, LEGACY_ENV_ID, *LEGACY_SPECS):
        registry.pop(env_id, None)
    module = _load_realworld_package(monkeypatch)
    try:
        yield module
    finally:
        for env_id in (ENV_ID, LEGACY_ENV_ID, *LEGACY_SPECS):
            registry.pop(env_id, None)


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
    cfg = _compose_config(monkeypatch)

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
    assert cfg.actor.model.image_num == 3
    assert cfg.runner.only_eval is False
    assert cfg.rollout.model.precision == cfg.actor.model.precision

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


def test_importing_realworld_is_f1_only_and_has_no_process_side_effects() -> None:
    script = textwrap.dedent(
        f"""
        import json
        import sys
        from types import ModuleType

        process_calls = []
        psutil = ModuleType("psutil")
        psutil.process_iter = lambda: process_calls.append("scan")
        sys.modules["psutil"] = psutil

        import gymnasium as gym
        import rlinf.envs.realworld as realworld

        forbidden_prefixes = (
            "cv2",
            "cv_bridge",
            "rclpy",
            "rospy",
            "turtle2_basic",
            "rlinf.envs.realworld.dosw1",
            "rlinf.envs.realworld.franka",
            "rlinf.envs.realworld.gim_arm",
            "rlinf.envs.realworld.xsquare",
        )
        forbidden = sorted(
            name
            for name in sys.modules
            if any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in forbidden_prefixes
            )
        )
        print(json.dumps({{
            "entry_point": gym.spec({ENV_ID!r}).entry_point,
            "legacy_ids": sorted(
                env_id
                for env_id in {sorted(LEGACY_SPECS)!r}
                if gym.spec(env_id) is not None
            ),
            "forbidden": forbidden,
            "process_calls": process_calls,
            "exports": list(realworld.__all__),
        }}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    payload = json.loads(result.stdout)

    assert payload["entry_point"] == (
        "rlinf.envs.realworld.f1.tasks:DualArmPegInsertionEnv"
    )
    assert payload["legacy_ids"] == sorted(LEGACY_SPECS)
    assert payload["forbidden"] == []
    assert payload["process_calls"] == []
    assert payload["exports"] == [
        "DualFrankaEnv",
        "DualFrankaJointEnv",
        "DualFrankaJointRobotConfig",
        "DualFrankaTCPEnv",
        "DualFrankaTCPRobotConfig",
        "DualFrankaRobotConfig",
        "DOSW1Config",
        "DOSW1Env",
        "dosw1_tasks",
        "FrankaEnv",
        "FrankaRobotConfig",
        "FrankaRobotState",
        "f1_tasks",
        "franka_tasks",
        "GimArmEnv",
        "GimArmRobotConfig",
        "GimArmRobotState",
        "gim_arm_tasks",
        "Turtle2Env",
        "Turtle2RobotConfig",
        "Turtle2RobotState",
        "xsquare_tasks",
        "RealWorldEnv",
    ]

    with pytest.raises(gym.error.Error):
        gym.spec(LEGACY_ENV_ID)


@pytest.mark.parametrize(
    ("option_name", "option_value"),
    _gym_registration_metadata_conflicts().items(),
)
def test_parent_import_rejects_conflicting_f1_registration_metadata(
    option_name: str,
    option_value: object,
) -> None:
    registration_option = {option_name: option_value}
    script = textwrap.dedent(
        f"""
        import gymnasium as gym
        from gymnasium.envs.registration import WrapperSpec

        wrapper = WrapperSpec(
            name="F1SupervisedEpisodeControl",
            entry_point=(
                "rlinf.envs.realworld.f1.tasks:"
                "_make_supervised_episode_control"
            ),
            kwargs={{}},
        )
        gym.register(
            id={ENV_ID!r},
            entry_point=(
                "rlinf.envs.realworld.f1.tasks:DualArmPegInsertionEnv"
            ),
            additional_wrappers=(wrapper,),
            **{registration_option!r},
        )
        import rlinf.envs.realworld  # noqa: F401
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "RuntimeError" in result.stderr
    assert ENV_ID in result.stderr


def test_parent_import_accepts_only_the_exact_canonical_f1_spec() -> None:
    script = textwrap.dedent(
        f"""
        import importlib
        import json

        import gymnasium as gym
        from gymnasium.envs.registration import WrapperSpec

        wrapper = WrapperSpec(
            name="F1SupervisedEpisodeControl",
            entry_point=(
                "rlinf.envs.realworld.f1.tasks:"
                "_make_supervised_episode_control"
            ),
            kwargs={{}},
        )
        gym.register(
            id={ENV_ID!r},
            entry_point=(
                "rlinf.envs.realworld.f1.tasks:DualArmPegInsertionEnv"
            ),
            additional_wrappers=(wrapper,),
        )
        before = gym.spec({ENV_ID!r})
        import rlinf.envs.realworld.f1.tasks as tasks
        importlib.reload(tasks)
        after = gym.spec({ENV_ID!r})
        print(json.dumps({{
            "same_spec": before is after,
            "wrappers": [
                {{
                    "name": item.name,
                    "entry_point": item.entry_point,
                    "kwargs": item.kwargs,
                }}
                for item in after.additional_wrappers
            ],
        }}))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    payload = json.loads(result.stdout)

    assert payload == {
        "same_spec": True,
        "wrappers": [
            {
                "name": "F1SupervisedEpisodeControl",
                "entry_point": (
                    "rlinf.envs.realworld.f1.tasks:_make_supervised_episode_control"
                ),
                "kwargs": {},
            }
        ],
    }


def test_parent_import_rejects_conflicting_f1_additional_wrappers() -> None:
    script = textwrap.dedent(
        f"""
        import gymnasium as gym
        from gymnasium.envs.registration import WrapperSpec

        wrapper = WrapperSpec(
            name="ConflictingWrapper",
            entry_point="conflicting.module:wrap",
            kwargs={{}},
        )
        gym.register(
            id={ENV_ID!r},
            entry_point=(
                "rlinf.envs.realworld.f1.tasks:DualArmPegInsertionEnv"
            ),
            additional_wrappers=(wrapper,),
        )
        import rlinf.envs.realworld  # noqa: F401
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "RuntimeError" in result.stderr
    assert ENV_ID in result.stderr


def test_direct_legacy_gym_factory_runs_setup_once_before_construction(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del registered_f1
    order: list[str] = []
    received_kwargs: list[dict[str, object]] = []

    class ControlledLegacyEnv(gym.Env):
        metadata = {"render_modes": []}
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        observation_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def create_franka_env(**kwargs: object) -> gym.Env:
        order.append("constructor")
        assert order[0] == "setup"
        received_kwargs.append(kwargs)
        return ControlledLegacyEnv()

    tasks_module = ModuleType("rlinf.envs.realworld.franka.tasks")
    tasks_module.create_franka_env = create_franka_env
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.realworld.franka.tasks",
        tasks_module,
    )

    import psutil

    def process_iter() -> list[object]:
        order.append("setup")
        return []

    monkeypatch.setattr(psutil, "process_iter", process_iter)
    factory_kwargs = {
        "override_cfg": {"is_dummy": True},
        "worker_info": None,
        "hardware_info": None,
        "env_idx": 3,
        "env_cfg": {"sentinel": "preserved"},
    }

    first = gym.make("FrankaEnv-v1", **factory_kwargs)
    second = gym.make("FrankaEnv-v1", **factory_kwargs)
    try:
        assert order == ["setup", "constructor", "constructor"]
        assert received_kwargs == [factory_kwargs, factory_kwargs]
    finally:
        first.close()
        second.close()


def test_explicit_legacy_tasks_import_keeps_the_lightweight_proxy_registration(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy_entry_point = gym.spec("GimArmPegInsertionEnv-v1").entry_point
    gim_arm_package = ModuleType("rlinf.envs.realworld.gim_arm")
    gim_arm_package.__package__ = "rlinf.envs.realworld.gim_arm"
    gim_arm_package.__path__ = [str(ROOT / "rlinf" / "envs" / "realworld" / "gim_arm")]
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.realworld.gim_arm",
        gim_arm_package,
    )
    peg_module = ModuleType("rlinf.envs.realworld.gim_arm.tasks.peg_insertion")
    peg_module.GimArmPegInsertionEnv = type("ControlledGimArmEnv", (), {})
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.realworld.gim_arm.tasks.peg_insertion",
        peg_module,
    )

    tasks_module = registered_f1.gim_arm_tasks

    assert tasks_module.__name__ == "rlinf.envs.realworld.gim_arm.tasks"
    assert gym.spec("GimArmPegInsertionEnv-v1").entry_point == proxy_entry_point
    for export_name, module_name in (
        ("dosw1_tasks", "rlinf.envs.realworld.dosw1.tasks"),
        ("franka_tasks", "rlinf.envs.realworld.franka.tasks"),
        ("xsquare_tasks", "rlinf.envs.realworld.xsquare.tasks"),
    ):
        controlled_tasks = ModuleType(module_name)
        monkeypatch.setitem(sys.modules, module_name, controlled_tasks)
        assert getattr(registered_f1, export_name) is controlled_tasks


def test_parent_import_rejects_a_conflicting_legacy_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_id in (ENV_ID, LEGACY_ENV_ID, *LEGACY_SPECS):
        registry.pop(env_id, None)
    gym.register("FrankaEnv-v1", entry_point="conflicting.module:factory")
    try:
        with pytest.raises(RuntimeError, match="FrankaEnv-v1"):
            _load_realworld_package(monkeypatch)
    finally:
        for env_id in (ENV_ID, LEGACY_ENV_ID, *LEGACY_SPECS):
            registry.pop(env_id, None)


def test_parent_import_normalizes_an_exact_legacy_registration_to_its_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_id in (ENV_ID, LEGACY_ENV_ID, *LEGACY_SPECS):
        registry.pop(env_id, None)
    gym.register("FrankaEnv-v1", entry_point=LEGACY_SPECS["FrankaEnv-v1"])
    try:
        _load_realworld_package(monkeypatch)

        assert gym.spec("FrankaEnv-v1").entry_point == (
            "rlinf.envs.realworld.registration:create_franka_env"
        )
    finally:
        for env_id in (ENV_ID, LEGACY_ENV_ID, *LEGACY_SPECS):
            registry.pop(env_id, None)


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


@pytest.mark.parametrize(
    "make_kwargs",
    [
        {},
        {"env_cfg": {}},
        {"env_cfg": {"operator_control": None}},
    ],
    ids=["no-env-cfg", "missing-operator-control", "null-operator-control"],
)
def test_direct_gym_make_without_operator_control_fails_closed(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    make_kwargs: dict[str, object],
) -> None:
    del registered_f1
    motion_calls, close_calls = _record_controller_motion(
        monkeypatch,
        expected_is_dummy=True,
    )

    with pytest.raises(ValueError, match="operator_control"):
        gym.make(ENV_ID, override_cfg={"is_dummy": True}, **make_kwargs)

    assert motion_calls == []
    assert close_calls == ["close"]


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


def test_composed_config_runs_realworld_env_through_the_four_step_horizon(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del registered_f1
    cfg = _compose_config(monkeypatch)
    from rlinf.envs import get_env_cls

    env_cls = get_env_cls("realworld", cfg.env.train)

    def fail_if_legacy_setup_runs() -> None:
        raise AssertionError("F1 construction must not run legacy node setup")

    monkeypatch.setattr(
        env_cls,
        "realworld_setup",
        staticmethod(fail_if_legacy_setup_runs),
    )
    env = env_cls(
        cfg.env.train,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        observation, _ = env.reset(seed=11)
        assert tuple(observation["states"].shape) == (1, 16)
        assert tuple(observation["main_images"].shape) == (1, 128, 128, 3)
        assert tuple(observation["extra_view_images"].shape) == (
            1,
            2,
            128,
            128,
            3,
        )

        truncations = []
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step(
                np.zeros((1, 16), dtype=np.float32)
            )
            assert terminated.tolist() == [False]
            truncations.append(truncated.tolist())
        assert truncations == [[False], [False], [False], [True]]
        assert env.elapsed_steps.tolist() == [0]
        assert not any(
            name.startswith(
                (
                    "rlinf.envs.realworld.dosw1",
                    "rlinf.envs.realworld.franka",
                    "rlinf.envs.realworld.gim_arm",
                    "rlinf.envs.realworld.xsquare",
                )
            )
            for name in sys.modules
        )
    finally:
        env.close()


def test_custom_realworld_id_runs_legacy_setup_once_before_each_constructor(
    registered_f1: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del registered_f1
    custom_env_id = "Task5ControlledRealWorld-v0"
    order: list[str] = []

    class ControlledRealWorldTask(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            override_cfg: Mapping[str, object],
            worker_info: object,
            hardware_info: object,
            env_idx: int,
            env_cfg: Mapping[str, object],
        ) -> None:
            del override_cfg, worker_info, hardware_info, env_idx, env_cfg
            order.append("constructor")
            assert order[0] == "setup"
            self.action_space = gym.spaces.Box(
                -1.0,
                1.0,
                shape=(16,),
                dtype=np.float32,
            )
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "left_joint_position": gym.spaces.Box(
                                -np.inf,
                                np.inf,
                                shape=(7,),
                                dtype=np.float32,
                            ),
                            "left_gripper": gym.spaces.Box(
                                0.0,
                                1.0,
                                shape=(1,),
                                dtype=np.float32,
                            ),
                            "right_joint_position": gym.spaces.Box(
                                -np.inf,
                                np.inf,
                                shape=(7,),
                                dtype=np.float32,
                            ),
                            "right_gripper": gym.spaces.Box(
                                0.0,
                                1.0,
                                shape=(1,),
                                dtype=np.float32,
                            ),
                        }
                    ),
                    "frames": gym.spaces.Dict(
                        {
                            key: gym.spaces.Box(
                                0,
                                255,
                                shape=(128, 128, 3),
                                dtype=np.uint8,
                            )
                            for key in (
                                "head_color",
                                "left_wrist_color",
                                "right_wrist_color",
                            )
                        }
                    ),
                }
            )
            self.task_description = "controlled legacy task"

    import psutil

    def process_iter() -> list[object]:
        order.append("setup")
        return []

    monkeypatch.setattr(psutil, "process_iter", process_iter)
    gym.register(custom_env_id, entry_point=ControlledRealWorldTask)
    cfg = _compose_config(monkeypatch)
    with open_dict(cfg.env.train.init_params):
        cfg.env.train.init_params.id = custom_env_id
    from rlinf.envs import get_env_cls

    env_cls = get_env_cls("realworld", cfg.env.train)
    first = env_cls(cfg.env.train, 1, 0, 1, None)
    second = env_cls(cfg.env.train, 1, 1, 1, None)
    try:
        assert order == ["setup", "constructor", "constructor"]
    finally:
        first.env.close()
        second.env.close()
        registry.pop(custom_env_id, None)


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
