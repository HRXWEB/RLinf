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

"""Fault-boundary integration coverage for the configured F1 environment."""

import asyncio
import importlib
import importlib.util
import queue
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from f1_robot_controller import create_controller
from gymnasium.envs.registration import registry
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CONFIG_ROOT = ROOT / "examples" / "embodiment" / "config"
CONFIG_NAME = "realworld_dummy_f1_peg_sac_cnn_async"
ENV_ID = "F1DualArmPegInsertionEnv-v1"
FORBIDDEN_MODULE_PREFIXES = ("rclpy", "cv_bridge")


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


def _stub_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    **attributes: object,
) -> ModuleType:
    module = ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _WorkerImportBoundary:
    """Minimal scheduler surface needed to load worker production methods."""

    @staticmethod
    def timer(*_args: object, **_kwargs: object) -> Callable:
        return lambda function: function


def _load_env_worker_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load EnvWorker while isolating unavailable Ray-only import surfaces."""

    def no_op(*_args: object, **_kwargs: object) -> None:
        return None

    _stub_module(
        monkeypatch,
        "rlinf.algorithms.registry",
        calculate_adv_and_returns=lambda **_kwargs: {},
    )
    _stub_module(
        monkeypatch,
        "rlinf.algorithms.rlt.transition",
        update_rlt_transitions=no_op,
    )
    _stub_module(
        monkeypatch,
        "rlinf.scheduler",
        Channel=object,
        Cluster=object,
        CommMapper=object,
        Worker=_WorkerImportBoundary,
    )
    _stub_module(monkeypatch, "rlinf.config", SupportedModel=object)
    action_utils_path = ROOT / "rlinf" / "envs" / "action_utils.py"
    action_utils_spec = importlib.util.spec_from_file_location(
        "_f1_action_utils_under_test",
        action_utils_path,
    )
    assert action_utils_spec is not None and action_utils_spec.loader is not None
    action_utils_module = importlib.util.module_from_spec(action_utils_spec)
    action_utils_spec.loader.exec_module(action_utils_module)
    _stub_module(
        monkeypatch,
        "rlinf.envs.action_utils",
        prepare_actions=action_utils_module.prepare_actions,
    )
    _stub_module(monkeypatch, "rlinf.envs.wrappers", RecordVideo=object)
    _stub_module(monkeypatch, "rlinf.utils.data_iter_utils", split_list=no_op)
    _stub_module(
        monkeypatch,
        "rlinf.utils.distributed",
        masked_stats=no_op,
        normalize_from_stats=no_op,
    )
    _stub_module(monkeypatch, "rlinf.utils.metric_utils", compute_split_num=no_op)
    _stub_module(
        monkeypatch,
        "rlinf.utils.nested_dict_process",
        clone_nested_to_cpu=no_op,
        copy_dict_tensor=no_op,
        split_dict_to_chunk=no_op,
        update_nested_cfg=no_op,
    )
    _stub_module(
        monkeypatch,
        "rlinf.utils.placement",
        HybridComponentPlacement=object,
    )
    _stub_module(
        monkeypatch,
        "rlinf.utils.utils",
        flatten_embodied_batch=no_op,
        pack_batch=no_op,
        preprocess_embodied_batch=no_op,
    )
    _stub_module(
        monkeypatch,
        "rlinf.workers.env.history_manager",
        HistoryManager=object,
    )
    module_path = ROOT / "rlinf" / "workers" / "env" / "env_worker.py"
    spec = importlib.util.spec_from_file_location(
        "_f1_env_worker_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_async_sac_actor_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load async SAC replay admission without FSDP/Ray construction."""

    base_module = _stub_module(
        monkeypatch,
        "rlinf.workers.actor.fsdp_sac_policy_worker",
        EmbodiedSACFSDPPolicy=object,
    )
    del base_module
    _stub_module(monkeypatch, "rlinf.scheduler", Worker=_WorkerImportBoundary)
    _stub_module(
        monkeypatch,
        "rlinf.utils.metric_utils",
        append_to_dict=lambda *_args, **_kwargs: None,
        compute_split_num=lambda *_args, **_kwargs: 1,
    )
    module_path = (
        ROOT / "rlinf" / "workers" / "actor" / "async_fsdp_sac_policy_worker.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_f1_async_sac_actor_under_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configured_f1_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Any, Any, Any]]:
    """Build the composed RealWorld/F1/wrapper stack with installed Fake."""

    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples" / "embodiment"))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name=CONFIG_NAME)
    cfg.env.train.max_episode_steps = 10
    cfg.env.train.override_cfg = {
        "controller": _fake_controller_config(),
        "action_scale": _action_scale(),
        "motion_envelope": _approved_motion_envelope(),
        "max_num_steps": 10,
    }

    realworld = importlib.import_module("rlinf.envs.realworld")
    tasks_module = importlib.import_module("rlinf.envs.realworld.f1.tasks")
    if ENV_ID not in registry:
        importlib.reload(tasks_module)
    env_module = importlib.import_module("rlinf.envs.realworld.f1.f1_robot_env")
    controllers = []

    def capture_installed_fake(config: dict[str, object]) -> Any:
        controller = create_controller(config)
        controllers.append(controller)
        return controller

    monkeypatch.setattr(env_module, "create_controller", capture_installed_fake)
    env = realworld.RealWorldEnv(
        cfg.env.train,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        env.reset()
        assert len(controllers) == 1
        f1_env = env.env.envs[0]
        yield env, f1_env, controllers[0]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("inject_fault", "expected_cause"),
    [
        (
            lambda controller: controller.fault_injector.inject_stale_sensor(
                "head_color"
            ),
            "ObservationUnavailableError",
        ),
        (
            lambda controller: controller.fault_injector.inject_missing_sensor(
                "left_joint_position"
            ),
            "ObservationUnavailableError",
        ),
        (
            lambda controller: controller.fault_injector.reject_next_command(
                "injected command rejection"
            ),
            "CommandRejectedError",
        ),
        (
            lambda controller: controller.fault_injector.fail_executor_once(
                "injected executor fault"
            ),
            "ControllerError",
        ),
    ],
    ids=("stale-camera", "missing-joint-state", "command-rejection", "executor-fault"),
)
def test_installed_fake_faults_fail_closed_without_exposing_a_transition(
    configured_f1_env: tuple[Any, Any, Any],
    inject_fault: Callable[[Any], None],
    expected_cause: str,
) -> None:
    """Catch a fault being downgraded into a replayable Gym transition."""

    env, f1_env, controller = configured_f1_env
    inject_fault(controller)
    transition_sentinel = object()
    transition: object = transition_sentinel

    with pytest.raises(Exception) as raised:
        transition = env.step(np.zeros((1, 14), dtype=np.float32))

    error = raised.value
    assert transition is transition_sentinel
    assert f1_env.unwrapped._num_steps == 0
    causes = []
    cause = error
    while cause is not None:
        causes.append(type(cause).__name__)
        cause = cause.__cause__
    assert expected_cause in causes
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in FORBIDDEN_MODULE_PREFIXES
    )


def test_async_worker_fault_never_sends_or_admits_the_partial_trajectory(
    configured_f1_env: tuple[Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a staged faulting action reaching the ordinary SAC replay buffer."""

    from rlinf.data.embodied_io_struct import EnvOutput, RolloutResult

    env, _f1_env, controller = configured_f1_env
    controller.fault_injector.inject_stale_sensor("head_color")
    env_worker_module = _load_env_worker_module(monkeypatch)
    worker = object.__new__(env_worker_module.EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {"dagger": {"online_lerobot": {"enabled": False}}},
            "actor": {
                "model": {
                    "action_dim": 14,
                    "model_type": "cnn_policy",
                    "num_action_chunks": 1,
                }
            },
            "env": {
                "train": {
                    "auto_reset": True,
                    "env_type": "realworld",
                    "group_name": "EnvGroup",
                    "ignore_terminations": True,
                    "max_episode_steps": 4,
                }
            },
            "rollout": {"group_name": "RolloutGroup"},
        }
    )
    worker.enable_online_lerobot = False
    worker.stage_num = 1
    worker.rollout_epoch = 1
    worker.n_train_chunk_steps = 1
    worker.train_batch_size = 1
    worker.model_cfg = worker.cfg.actor.model
    worker.collect_prev_infos = False
    worker.collect_transitions = True
    worker.enable_rlt = False
    worker.reward_mode = "raw"
    worker.history_reward_assign = False
    worker.use_training_pipeline = False
    worker.env_decoupled_mode = False
    worker.env_list = [env]
    worker.rollout_results = None
    worker._prefetched_train_bootstrap = [
        EnvOutput(
            obs={"states": torch.zeros((1, 16), dtype=torch.float32)},
            dones=torch.zeros((1, 1), dtype=torch.bool),
            terminations=torch.zeros((1, 1), dtype=torch.bool),
            truncations=torch.zeros((1, 1), dtype=torch.bool),
        )
    ]
    rollout_result = RolloutResult(
        actions=torch.zeros((1, 1, 14), dtype=torch.float32),
        forward_inputs={
            "action": torch.zeros((1, 1, 14), dtype=torch.float32),
        },
    )
    worker.recv_from = lambda **_kwargs: rollout_result
    worker.send_to = lambda **_kwargs: None
    trajectory_send_calls: list[object] = []
    production_send = worker.send_rollout_trajectories

    async def record_trajectory_send(rollout: object, channel: object) -> None:
        trajectory_send_calls.append(rollout)
        await production_send(rollout, channel)

    worker.send_rollout_trajectories = record_trajectory_send

    class RecordingActorChannel:
        def __init__(self) -> None:
            self.items: list[object] = []

        def put(self, item: object, **_kwargs: object) -> None:
            self.items.append(item)

    actor_channel = RecordingActorChannel()

    with pytest.raises(Exception):
        asyncio.run(
            worker._run_interact_once(
                input_channel=object(),
                rollout_channel=object(),
                reward_channel=None,
                actor_channel=actor_channel,
                cooperative_yield=False,
            )
        )

    assert len(worker.rollout_results[0].actions) == 1
    assert trajectory_send_calls == []
    assert actor_channel.items == []

    actor_module = _load_async_sac_actor_module(monkeypatch)
    actor = object.__new__(actor_module.AsyncEmbodiedSACFSDPPolicy)
    actor._recv_queue = queue.Queue()
    for item in actor_channel.items:
        actor._recv_queue.put(item)
    replay_admissions: list[list[object]] = []
    actor.replay_buffer = SimpleNamespace(
        add_trajectories=lambda items: replay_admissions.append(items)
    )
    actor.demo_buffer = None

    actor._drain_received_trajectories()

    assert replay_admissions == []


def test_dummy_config_accepts_negative_tcp_z_from_the_fake_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch the Fake zero pose sitting on a one-sided workspace boundary."""

    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples" / "embodiment"))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name=CONFIG_NAME)

    realworld = importlib.import_module("rlinf.envs.realworld")
    env = realworld.RealWorldEnv(
        cfg.env.train,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        env.reset()
        action = np.zeros((1, 14), dtype=np.float32)
        action[0, 2] = -1.0
        action[0, 9] = -1.0

        observation, *_ = env.step(action)

        assert observation["states"].shape == (1, 16)
    finally:
        env.close()


def test_gpu_e2e_config_freezes_the_two_step_f1_sac_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Catch the fake F1 e2e config losing a training or schema observable."""

    e2e_config_root = ROOT / "tests" / "e2e_tests" / "embodied"
    del tmp_path
    monkeypatch.setenv("REPO_PATH", str(ROOT))
    with initialize_config_dir(config_dir=str(e2e_config_root)):
        cfg = compose(config_name="realworld_f1_dummy_sac_cnn")
    container = OmegaConf.to_container(cfg, resolve=False)

    assert cfg.runner.max_steps == 2
    assert cfg.runner.weight_sync_interval == 1
    assert cfg.algorithm.replay_buffer.min_buffer_size == 1
    assert cfg.algorithm.update_epoch >= 1
    assert cfg.actor.micro_batch_size == 8
    assert cfg.actor.global_batch_size == 8
    assert cfg.actor.model.state_dim == 16
    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.image_num == 3
    assert cfg.algorithm.entropy_tuning.target_entropy == -14
    assert cfg.env.train.init_params.id == ENV_ID
    assert cfg.env.train.action_dim == 14
    assert cfg.env.train.init_params.registration_module == (
        "rlinf.envs.realworld.f1.tasks"
    )
    assert cfg.env.train.max_episode_steps == 10
    assert cfg.env.train.max_steps_per_rollout_epoch == 10
    assert cfg.env.train.override_cfg.controller.backend == "fake"
    assert cfg.env.train.override_cfg.action_scale.tcp_position_m == pytest.approx(
        0.005
    )
    assert cfg.env.train.override_cfg.action_scale.tcp_orientation_deg == pytest.approx(
        1.0
    )
    assert cfg.env.train.override_cfg.max_num_steps == 10
    assert cfg.env.train.override_cfg.motion_envelope.gripper_percent_closed == [
        0.0,
        100.0,
    ]
    assert cfg.env.eval.init_params.registration_module == (
        "rlinf.envs.realworld.f1.tasks"
    )
    assert cfg.env.eval.max_episode_steps == 10
    assert cfg.env.eval.max_steps_per_rollout_epoch == 10
    assert cfg.env.eval.override_cfg.controller.backend == "fake"
    assert cfg.env.eval.override_cfg.max_num_steps == 10
    assert cfg.rollout.collect_transitions is True
    assert cfg.rollout.model.model_path == cfg.actor.model.model_path
    assert cfg.runner.logger.log_path == "./logs/f1-peg-fake-e2e"
    serialized = str(container)
    assert "F1_" not in serialized
    assert "is_dummy" not in serialized
    assert "architecture_smoke" not in serialized
