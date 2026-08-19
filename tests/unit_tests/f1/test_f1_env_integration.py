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
from f1_robot_controller import ControllerConfig, create_controller
from gymnasium.envs.registration import registry
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "examples" / "embodiment" / "config"
CONFIG_NAME = "realworld_dummy_f1_peg_sac_cnn_async"
ENV_ID = "F1DualArmPegInsertionEnv-v1"
FORBIDDEN_MODULE_PREFIXES = ("rclpy", "cv_bridge")


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


def _load_async_runner_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the real runner inheritance chain without Ray/logger dependencies."""

    _stub_module(
        monkeypatch,
        "rlinf.scheduler",
        Channel=object,
        WorkerGroupFuncResult=object,
    )
    _stub_module(monkeypatch, "rlinf.utils.distributed", ScopedTimer=object)
    _stub_module(monkeypatch, "rlinf.utils.logging", get_logger=lambda: None)
    _stub_module(monkeypatch, "rlinf.utils.metric_logger", MetricLogger=object)
    _stub_module(
        monkeypatch,
        "rlinf.utils.metric_utils",
        compute_evaluate_metrics=lambda results: results,
        print_metrics_table=lambda *_args, **_kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "rlinf.utils.runner_utils",
        check_progress=lambda *_args, **_kwargs: (False,) * 3,
    )
    _stub_module(monkeypatch, "rlinf.utils.timers", Timer=object)
    base_path = ROOT / "rlinf" / "runners" / "embodied_runner.py"
    base_spec = importlib.util.spec_from_file_location(
        "_f1_embodied_runner_under_test",
        base_path,
    )
    assert base_spec is not None and base_spec.loader is not None
    base_module = importlib.util.module_from_spec(base_spec)
    base_spec.loader.exec_module(base_module)
    monkeypatch.setitem(sys.modules, "rlinf.runners.embodied_runner", base_module)

    module_path = ROOT / "rlinf" / "runners" / "async_embodied_runner.py"
    spec = importlib.util.spec_from_file_location(
        "_f1_async_runner_under_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SyncHandle:
    """Controllable stand-in for the external worker-group handle boundary."""

    def __init__(self, *, done: bool = True, wait_error: Exception | None = None):
        self.is_done = done
        self.wait_error = wait_error
        self.wait_calls = 0

    def done(self) -> bool:
        return self.is_done

    def wait(self) -> None:
        self.wait_calls += 1
        if self.wait_error is not None:
            raise self.wait_error


class _SyncRollout:
    def __init__(self, handle: _SyncHandle) -> None:
        self.handle = handle
        self.request_calls = 0

    def sync_model_from_actor(self) -> _SyncHandle:
        self.request_calls += 1
        return self.handle

    def request_actor_sync_model(self) -> _SyncHandle:
        self.request_calls += 1
        return self.handle


class _SyncActor:
    def __init__(self, handle: _SyncHandle) -> None:
        self.handle = handle
        self.request_calls = 0

    def sync_model_to_rollout(self) -> _SyncHandle:
        self.request_calls += 1
        return self.handle


@pytest.fixture
def configured_f1_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Any, Any, Any]]:
    """Build the composed RealWorld/F1/wrapper stack with installed Fake."""

    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples" / "embodiment"))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name=CONFIG_NAME)

    realworld = importlib.import_module("rlinf.envs.realworld")
    tasks_module = importlib.import_module("rlinf.envs.realworld.f1.tasks")
    if ENV_ID not in registry:
        importlib.reload(tasks_module)
    env_module = importlib.import_module("rlinf.envs.realworld.f1.f1_robot_env")
    controllers = []

    def capture_installed_fake(
        *,
        is_dummy: bool,
        config: ControllerConfig,
    ) -> Any:
        controller = create_controller(is_dummy=is_dummy, config=config)
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
        wrapper = env.env.envs[0]
        yield env, wrapper, controllers[0]
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
            "ObservationUnavailableError",
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

    env, wrapper, controller = configured_f1_env
    wrappers = importlib.import_module("rlinf.envs.realworld.f1.wrappers")
    inject_fault(controller)
    transition_sentinel = object()
    transition: object = transition_sentinel

    with pytest.raises(wrappers.EpisodeFaultError) as raised:
        transition = env.step(np.zeros((1, 16), dtype=np.float32))

    error = raised.value
    assert transition is transition_sentinel
    assert error.transition_valid is False
    assert error.quarantine is True
    assert wrapper.state is wrappers.EpisodeState.FAULT
    causes = []
    cause = error.__cause__
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

    env, _wrapper, controller = configured_f1_env
    controller.fault_injector.inject_stale_sensor("head_color")
    wrappers = importlib.import_module("rlinf.envs.realworld.f1.wrappers")
    env_worker_module = _load_env_worker_module(monkeypatch)
    worker = object.__new__(env_worker_module.EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {"dagger": {"online_lerobot": {"enabled": False}}},
            "actor": {
                "model": {
                    "action_dim": 16,
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
        actions=torch.zeros((1, 1, 16), dtype=torch.float32),
        forward_inputs={
            "action": torch.zeros((1, 1, 16), dtype=torch.float32),
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

    with pytest.raises(wrappers.EpisodeFaultError):
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


def test_gpu_e2e_config_freezes_the_two_step_f1_sac_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Catch the GPU gate losing a required training or schema observable."""

    e2e_config_root = ROOT / "tests" / "e2e_tests" / "embodied"
    model_path = str(tmp_path / "model-artifact")
    run_dir = str(tmp_path / "run-artifact")
    monkeypatch.setenv("REPO_PATH", str(ROOT))
    monkeypatch.setenv("F1_MODEL_PATH", model_path)
    monkeypatch.setenv("F1_RUN_DIR", run_dir)
    with initialize_config_dir(config_dir=str(e2e_config_root)):
        cfg = compose(config_name="realworld_f1_dummy_sac_cnn")

    assert cfg.runner.max_steps == 2
    assert cfg.runner.weight_sync_interval == 1
    assert cfg.algorithm.replay_buffer.min_buffer_size == 1
    assert cfg.algorithm.update_epoch >= 1
    assert cfg.actor.micro_batch_size == 8
    assert cfg.actor.global_batch_size == 8
    assert cfg.actor.model.state_dim == 16
    assert cfg.actor.model.action_dim == 16
    assert cfg.actor.model.image_num == 3
    assert cfg.algorithm.entropy_tuning.target_entropy == -16
    assert cfg.env.train.init_params.id == ENV_ID
    assert cfg.env.train.operator_control.mode == "automatic"
    assert cfg.env.train.override_cfg.is_dummy is True
    assert cfg.rollout.collect_transitions is True
    assert cfg.actor.model.model_path == model_path
    assert cfg.rollout.model.model_path == model_path
    assert cfg.runner.logger.log_path == run_dir


def test_weight_sync_wait_failures_never_increment_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a failed or unfinished weight sync being reported as successful."""

    module = _load_async_runner_module(monkeypatch)
    runner = module.AsyncEmbodiedRunner.__new__(module.AsyncEmbodiedRunner)
    runner._weight_sync_success_total = 0
    runner._pending_rollout_weight_sync = None
    runner.logger = SimpleNamespace(info=lambda _message: None)
    actor_handle = _SyncHandle(wait_error=RuntimeError("actor wait failed"))
    rollout_handle = _SyncHandle()
    runner.actor = _SyncActor(actor_handle)
    runner.rollout = _SyncRollout(rollout_handle)

    with pytest.raises(RuntimeError, match="actor wait failed"):
        runner.update_rollout_weights(no_wait=False)
    assert runner._weight_sync_success_total == 0
    assert actor_handle.wait_calls == 1
    assert rollout_handle.wait_calls == 0

    pending_rollout = _SyncHandle()
    pending_actor = _SyncHandle(wait_error=RuntimeError("pending wait failed"))
    runner._pending_rollout_weight_sync = (pending_rollout, pending_actor)
    with pytest.raises(RuntimeError, match="pending wait failed"):
        runner._cleanup_pending_rollout_weight_sync(no_wait=False)
    assert runner._weight_sync_success_total == 0
    assert runner._pending_rollout_weight_sync == (pending_rollout, pending_actor)


def test_unfinished_weight_sync_is_coalesced_without_a_false_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a pending no-wait sync being replaced or counted prematurely."""

    module = _load_async_runner_module(monkeypatch)
    runner = module.AsyncEmbodiedRunner.__new__(module.AsyncEmbodiedRunner)
    pending_rollout = _SyncHandle(done=False)
    pending_actor = _SyncHandle(done=True)
    runner._pending_rollout_weight_sync = (pending_rollout, pending_actor)
    runner._weight_sync_request_total = 0
    runner._weight_sync_coalesced_total = 0
    runner._weight_sync_success_total = 0
    runner.logger = SimpleNamespace(info=lambda _message: None)
    new_rollout = _SyncRollout(_SyncHandle())
    new_actor = _SyncActor(_SyncHandle())
    runner.rollout = new_rollout
    runner.actor = new_actor

    runner.update_rollout_weights(no_wait=True)

    assert runner._weight_sync_request_total == 1
    assert runner._weight_sync_coalesced_total == 1
    assert runner._weight_sync_success_total == 0
    assert runner._pending_rollout_weight_sync == (pending_rollout, pending_actor)
    assert pending_rollout.wait_calls == 0
    assert pending_actor.wait_calls == 0
    assert new_rollout.request_calls == 0
    assert new_actor.request_calls == 0


def test_successful_weight_syncs_publish_a_monotonic_metric_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch successful syncs being omitted from per-step training metrics."""

    module = _load_async_runner_module(monkeypatch)
    runner = module.AsyncEmbodiedRunner.__new__(module.AsyncEmbodiedRunner)
    runner._weight_sync_success_total = 0
    runner._pending_rollout_weight_sync = None
    runner.logger = SimpleNamespace(info=lambda _message: None)
    runner.actor = _SyncActor(_SyncHandle())
    runner.rollout = _SyncRollout(_SyncHandle())

    observed_metrics = []
    for _step in range(2):
        runner.update_rollout_weights(no_wait=False)
        observed_metrics.append(runner._weight_sync_metrics())
    runner._pending_rollout_weight_sync = (_SyncHandle(), _SyncHandle())
    assert runner._cleanup_pending_rollout_weight_sync(no_wait=True) is True
    observed_metrics.append(runner._weight_sync_metrics())

    assert observed_metrics == [
        {"train/weight_sync_success_total": 1},
        {"train/weight_sync_success_total": 2},
        {"train/weight_sync_success_total": 3},
    ]
